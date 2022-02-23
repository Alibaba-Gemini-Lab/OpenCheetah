//  Authors: Wen-jie Lu on 2021/9/14.
#include "gemini/cheetah/hom_fc_ss.h"

#include <seal/seal.h>
#include <seal/secretkey.h>
#include <seal/util/polyarithsmallmod.h>

#include <functional>

#include "gemini/core/logging.h"
#include "gemini/core/util/ThreadPool.h"

namespace gemini {

static TensorShape GetOutShape(const HomFCSS::Meta &meta) {
  if (meta.weight_shape.dims() != 2 || meta.input_shape.dims() != 1 ||
      meta.input_shape.length() != meta.weight_shape.cols()) {
    LOG(WARNING) << "GetConv2DOutShape invalid meta";
  }
  TensorShape outshape({meta.weight_shape.rows()});
  return outshape;
}

static TensorShape getSplit(const HomFCSS::Meta &meta, size_t N) {
  if (meta.weight_shape.dims() != 2 || meta.input_shape.dims() != 1 ||
      meta.input_shape.length() != meta.weight_shape.cols()) {
    LOG(FATAL) << "getSplit invalid meta";
  }

  size_t nrows = meta.weight_shape.rows();
  size_t ncols = meta.weight_shape.cols();

  std::array<int64_t, 2> ret{0};
  size_t min_cost = -1;
  for (size_t d0 = 1; d0 <= std::min(N, nrows); ++d0) {
    for (size_t d1 = 1; d1 <= std::min(N, ncols); ++d1) {
      if (d0 * d1 > N) continue;
      size_t ct_in = CeilDiv(ncols, d1);
      size_t ct_out = CeilDiv(nrows, d0);
      size_t cost = ct_in + ct_out;
      if (cost < min_cost) {
        min_cost = cost;
        ret[0] = int64_t(d0);
        ret[1] = int64_t(d1);
      }
    }
  }
  return TensorShape({ret[0], ret[1]});
}

static Code LaunchWorks(
    ThreadPool &tpool, size_t num_works,
    std::function<Code(long wid, size_t start, size_t end)> program) {
  if (num_works == 0) return Code::OK;
  const long pool_sze = tpool.pool_size();
  if (pool_sze <= 1L) {
    return program(0, 0, num_works);
  } else {
    Code code;
    std::vector<std::future<Code>> futures;
    size_t work_load = (num_works + pool_sze - 1) / pool_sze;
    for (long wid = 0; wid < pool_sze; ++wid) {
      size_t start = wid * work_load;
      size_t end = std::min(start + work_load, num_works);
      futures.push_back(tpool.enqueue(program, wid, start, end));
    }

    code = Code::OK;
    for (auto &&work : futures) {
      Code c = work.get();
      if (code == Code::OK && c != Code::OK) {
        code = c;
      }
    }
    return code;
  }
}

void truncate_for_decryption(seal::Ciphertext &ct,
                             const seal::Evaluator &evaluator,
                             const seal::SEALContext &context);

seal::scheme_type HomFCSS::scheme() const {
  if (context_) {
    return context_->first_context_data()->parms().scheme();
  } else {
    return seal::scheme_type::none;
  }
}

size_t HomFCSS::poly_degree() const {
  if (context_) {
    return context_->first_context_data()->parms().poly_modulus_degree();
  } else {
    return 0;
  }
}

uint64_t HomFCSS::plain_modulus() const {
  if (context_) {
    return context_->first_context_data()->parms().plain_modulus().value();
  } else {
    return -1;
  }
}

Code HomFCSS::setUp(const seal::SEALContext &context,
                    std::optional<seal::SecretKey> sk,
					std::shared_ptr<seal::PublicKey> pk) {
  context_ = std::make_shared<seal::SEALContext>(context);
  ENSURE_OR_RETURN(context_, Code::ERR_NULL_POINTER);

  if (sk) {
    if (!seal::is_metadata_valid_for(*sk, *context_)) {
      LOG(WARNING) << "HomFCSS: invalid secret key for this SEALContext";
      return Code::ERR_INVALID_ARG;
    }

    sk_ = seal::SecretKey(*sk);
    encryptor_ = std::make_shared<seal::Encryptor>(*context_, *sk);
  }

  if (pk) {
    if (!seal::is_metadata_valid_for(*pk, *context_)) {
      LOG(WARNING) << "HomFCSS: invalid public key for this SEALContext";
      return Code::ERR_INVALID_ARG;
    }

    pk_encryptor_ = std::make_shared<seal::Encryptor>(*context_, *pk);
  }
  evaluator_ = std::make_shared<seal::Evaluator>(*context_);
  return Code::OK;
}

Code HomFCSS::initPtx(seal::Plaintext &pt, seal::parms_id_type pid) const {
  ENSURE_OR_RETURN(context_, Code::ERR_CONFIG);

  if (scheme() != seal::scheme_type::ckks) {
    // BFV or BGV
    pt.parms_id() = seal::parms_id_zero;  // foo SEAL when using BFV
    pt.resize(poly_degree());
    ENSURE_OR_RETURN(pt.data() != nullptr, Code::ERR_SEAL_MEMORY);
    return Code::OK;
  }

  if (pid == seal::parms_id_zero) {
    pid = context_->first_parms_id();
  }

  auto cntxt_data = context_->get_context_data(pid);
  ENSURE_OR_RETURN(cntxt_data != nullptr, Code::ERR_INVALID_ARG);
  const size_t num_moduli = cntxt_data->parms().coeff_modulus().size();
  const size_t num_elt = seal::util::mul_safe(num_moduli, poly_degree());
  pt.parms_id() = seal::parms_id_zero;  // foo SEAL when using BFV
  pt.resize(num_elt);
  pt.parms_id() = pid;
  ENSURE_OR_RETURN(pt.data() != nullptr, Code::ERR_SEAL_MEMORY);
  return Code::OK;
}

Code HomFCSS::vec2PolyBFV(const uint64_t *vec, size_t len, seal::Plaintext &pt,
                          bool is_to_ntt) const {
  if (scheme() != seal::scheme_type::bfv) {
    LOG(FATAL) << "A2HBFV: invalid scheme";
  }

  if (is_to_ntt) {
    LOG(WARNING) << "A2H: demand is_to_ntt = false for scheme bfv";
  }

  CHECK_ERR(initPtx(pt), "A2H: InitPtx");
  ENSURE_OR_RETURN(vec != nullptr, Code::ERR_NULL_POINTER);
  ENSURE_OR_RETURN(len > 0 && len <= poly_degree(), Code::ERR_OUT_BOUND);

  seal::util::modulo_poly_coeffs(vec, len, plain_modulus(), pt.data());
  std::fill_n(pt.data() + len, pt.coeff_count() - len, 0);

  return Code::OK;
}

Code HomFCSS::vec2Poly(const uint64_t *vec, size_t len, seal::Plaintext &pt,
                       const Role role, bool is_to_ntt) const {
  switch (scheme()) {
    case seal::scheme_type::bfv:
      return vec2PolyBFV(vec, len, pt, is_to_ntt);
    default:
      LOG(WARNING) << "A2H: shceme is not supported yet\n";
  }
  return Code::ERR_INTERNAL;
}

Code HomFCSS::encryptInputVector(
    const Tensor<uint64_t> &input_vector, const Meta &meta,
    std::vector<seal::Serializable<seal::Ciphertext>> &encrypted_share,
    size_t nthreads) const {
  ENSURE_OR_RETURN(context_ && encryptor_, Code::ERR_CONFIG);
  ENSURE_OR_RETURN(input_vector.shape().IsSameSize(meta.input_shape),
                   Code::ERR_DIM_MISMATCH);
  ENSURE_OR_RETURN(meta.input_shape.num_elements() > 0, Code::ERR_INVALID_ARG);

  const bool is_ckks = scheme() == seal::scheme_type::ckks;
  ENSURE_OR_RETURN(!is_ckks, Code::ERR_INTERNAL);

  auto split_shape = getSplit(meta, poly_degree());
  if (split_shape.num_elements() > poly_degree()) {
    LOG(FATAL) << "BUG";
  }
  auto nout = CeilDiv(input_vector.length(), split_shape.cols());

  Role encode_role = Role::none;  // BFV/BGV not use this role
  encrypted_share.resize(nout, encryptor_->encrypt_zero());

  seal::Plaintext pt;
  std::vector<uint64_t> tmp(poly_degree());
  const uint64_t plain = plain_modulus();
  bool is_failed = false;
  for (size_t i = 0; i < nout && !is_failed; ++i) {
    auto start = i * split_shape.cols();
    auto end =
        std::min<size_t>(input_vector.length(), start + split_shape.cols());
    auto len = end - start;
    // reversed ordering for the vector
    tmp[0] = input_vector(start);
    std::transform(input_vector.data() + start + 1, input_vector.data() + end,
                   tmp.rbegin(),
                   [plain](uint64_t u) { return u > 0 ? plain - u : 0; });
    if (len < tmp.size()) {
      std::fill_n(tmp.rbegin() + len, tmp.size() - len - 1, 0);
    }

    if (Code::OK != vec2Poly(tmp.data(), tmp.size(), pt, encode_role, false)) {
      is_failed = true;
    } else {
      try {
        encrypted_share.at(i) = encryptor_->encrypt_symmetric(pt);
      } catch (const std::logic_error &e) {
        is_failed = true;
      }
    }
  }

  // erase the sensitive data
  seal::util::seal_memzero(tmp.data(), sizeof(uint64_t) * tmp.size());
  seal::util::seal_memzero(pt.data(), sizeof(uint64_t) * pt.coeff_count());

  if (is_failed) {
    return Code::ERR_INTERNAL;
  } else {
    return Code::OK;
  }
}

Code HomFCSS::encodeInputVector(const Tensor<uint64_t> &input_vector,
                                const Meta &meta,
                                std::vector<seal::Plaintext> &encoded_share,
                                size_t nthreads) const {
  ENSURE_OR_RETURN(context_, Code::ERR_CONFIG);
  ENSURE_OR_RETURN(input_vector.shape().IsSameSize(meta.input_shape),
                   Code::ERR_DIM_MISMATCH);
  ENSURE_OR_RETURN(meta.input_shape.num_elements() > 0, Code::ERR_INVALID_ARG);

  const bool is_ckks = scheme() == seal::scheme_type::ckks;
  ENSURE_OR_RETURN(!is_ckks, Code::ERR_INTERNAL);

  auto split_shape = getSplit(meta, poly_degree());
  if (split_shape.num_elements() > poly_degree()) {
    LOG(FATAL) << "BUG";
  }
  auto nout = CeilDiv(input_vector.length(), split_shape.cols());

  Role encode_role = Role::none;  // BFV/BGV not use this role
  encoded_share.resize(nout);

  auto encode_prg = [&](long wid, size_t start, size_t end) {
    std::vector<uint64_t> tmp(poly_degree());
    const uint64_t plain = plain_modulus();
    bool is_failed = false;
    for (size_t i = start; i < end && !is_failed; ++i) {
      auto start = i * split_shape.cols();
      auto end =
          std::min<size_t>(input_vector.length(), start + split_shape.cols());
      auto len = end - start;

      // reversed ordering for the vector
      tmp[0] = input_vector(start);
      std::transform(input_vector.data() + start + 1, input_vector.data() + end,
                     tmp.rbegin(),
                     [plain](uint64_t u) { return u > 0 ? plain - u : 0; });
      if (len < tmp.size()) {
        std::fill_n(tmp.rbegin() + len, tmp.size() - len - 1, 0);
      }
      if (Code::OK != vec2Poly(tmp.data(), tmp.size(), encoded_share.at(i),
                               encode_role, false)) {
        is_failed = true;
      }
    }
    seal::util::seal_memzero(tmp.data(), sizeof(uint64_t) * tmp.size());
    return is_failed ? Code::ERR_INTERNAL : Code::OK;
  };

  gemini::ThreadPool tpool(nthreads);
  return LaunchWorks(tpool, nout, encode_prg);
}

Code HomFCSS::encodeWeightMatrix(
    const Tensor<uint64_t> &weight_matrix, const Meta &meta,
    std::vector<std::vector<seal::Plaintext>> &encoded_share,
    size_t nthreads) const {
  ENSURE_OR_RETURN(context_, Code::ERR_CONFIG);
  ENSURE_OR_RETURN(weight_matrix.shape().IsSameSize(meta.weight_shape),
                   Code::ERR_DIM_MISMATCH);
  const size_t nrows = meta.weight_shape.rows();
  const size_t ncols = meta.weight_shape.cols();

  auto split_shape = getSplit(meta, poly_degree());
  if (split_shape.num_elements() > poly_degree()) {
    LOG(FATAL) << "BUG";
  }

  const auto n_row_blks = CeilDiv<size_t>(nrows, split_shape.rows());
  const auto n_col_blks = CeilDiv<size_t>(ncols, split_shape.cols());
  encoded_share.resize(n_row_blks);

  auto encode_prg = [&](long wid, size_t start, size_t end) {
    bool is_failed = false;
    std::vector<uint64_t> tmp(poly_degree());
    for (size_t r_blk = start; r_blk < end && !is_failed; ++r_blk) {
      encoded_share[r_blk].resize(n_col_blks);
      auto top_left_row = r_blk * split_shape.rows();
      auto top_right_row =
          std::min<size_t>(top_left_row + split_shape.rows(), nrows);
      auto row_extent = top_right_row - top_left_row;

      for (size_t c_blk = 0; c_blk < n_col_blks; ++c_blk) {
        auto top_left_col = c_blk * split_shape.cols();
        auto col_extent =
            std::min<size_t>(top_left_col + split_shape.cols(), ncols) -
            top_left_col;
        // Encode the sub-matrix start ad (top_left_row, top_left_col) with
        // sizes (row_extent, col_extent) Matrix is stored in row-major
        auto src_ptr =
            weight_matrix.data() + top_left_row * ncols + top_left_col;
        auto dst_ptr = tmp.begin();
        for (size_t r = 0; r < row_extent; ++r) {
          std::copy_n(src_ptr, col_extent, dst_ptr);
          // For the right-most submatrtix, we might need zero-padding.
          size_t nzero_pad = split_shape.cols() - col_extent;
          if (nzero_pad > 0) {
            std::fill_n(dst_ptr + col_extent, nzero_pad, 0);
          }
          dst_ptr += split_shape.cols();
          src_ptr += ncols;
        }
        // zero-out the other coefficients
        std::fill(dst_ptr, tmp.end(), 0);
        if (Code::OK != vec2Poly(tmp.data(), tmp.size(),
                                 encoded_share.at(r_blk).at(c_blk), Role::none,
                                 false)) {
          is_failed = true;
          break;
        }
      }
    }
    seal::util::seal_memzero(tmp.data(), sizeof(uint64_t) * tmp.size());
    return is_failed ? Code::ERR_INTERNAL : Code::OK;
  };

  ThreadPool tpool(nthreads);
  return LaunchWorks(tpool, n_row_blks, encode_prg);
}

Code HomFCSS::matVecMul(const std::vector<std::vector<seal::Plaintext>> &matrix,
                        const std::vector<seal::Ciphertext> &vec_share0,
                        const std::vector<seal::Plaintext> &vec_share1,
                        const Meta &meta,
                        std::vector<seal::Ciphertext> &out_share0,
                        Tensor<uint64_t> &out_share1, size_t nthreads) const {
  ENSURE_OR_RETURN(context_ && evaluator_, Code::ERR_CONFIG);

  auto split_shape = getSplit(meta, poly_degree());
  const size_t n_ct_in =
      CeilDiv<size_t>(meta.input_shape.length(), split_shape.cols());
  const size_t n_ct_out =
      CeilDiv<size_t>(meta.weight_shape.rows(), split_shape.rows());

  ENSURE_OR_RETURN(vec_share0.size() == n_ct_in, Code::ERR_INVALID_ARG);
  ENSURE_OR_RETURN(matrix.size() == n_ct_out, Code::ERR_INVALID_ARG);
  for (const auto &c : matrix) {
    ENSURE_OR_RETURN(c.size() == n_ct_in, Code::ERR_INVALID_ARG);
  }

  if (meta.is_shared_input && vec_share1.size() != n_ct_in) {
    return Code::ERR_DIM_MISMATCH;
  }

  ThreadPool tpool(nthreads);

  std::vector<seal::Ciphertext> input;
  if (meta.is_shared_input) {
    input.resize(n_ct_in);
    auto add_prg = [&](long wid, size_t start, size_t end) {
      for (size_t i = start; i < end; ++i) {
        evaluator_->add_plain(vec_share0[i], vec_share1[i], input[i]);
      }
      return Code::OK;
    };
    (void)LaunchWorks(tpool, n_ct_in, add_prg);
  }

  out_share0.resize(n_ct_out);
  auto fma_prg = [&](long wid, size_t start, size_t end) {
    for (size_t j = start; j < end; ++j) {
      evaluator_->multiply_plain(
          meta.is_shared_input ? input[0] : vec_share0[0], matrix[j][0],
          out_share0[j]);
      // TODO(wen-jie): to implement FMA
      for (size_t i = 1; i < n_ct_in; ++i) {
        seal::Ciphertext tmp;
        evaluator_->multiply_plain(
            meta.is_shared_input ? input[i] : vec_share0[i], matrix[j][i], tmp);
        evaluator_->add_inplace(out_share0[j], tmp);
      }
    }
    return Code::OK;
  };
  (void)LaunchWorks(tpool, n_ct_out, fma_prg);

  addRandomMask(out_share0, out_share1, meta, tpool);

  if (scheme() == seal::scheme_type::bfv) {
    for (auto &c : out_share0) {
      truncate_for_decryption(c, *evaluator_, *context_);
    }
  }

  // Post-processing for compressing out_ct volume.
  removeUnusedCoeffs(out_share0, meta);
  return Code::OK;
}

Code HomFCSS::addRandomMask(std::vector<seal::Ciphertext> &cts,
                            Tensor<uint64_t> &mask_vector, const Meta &meta,
                            gemini::ThreadPool &tpool) const {
  ENSURE_OR_RETURN(pk_encryptor_, Code::ERR_CONFIG);
  TensorShape split_shape = getSplit(meta, poly_degree());
  const size_t n_ct_out =
      CeilDiv<size_t>(meta.weight_shape.rows(), split_shape.rows());
  ENSURE_OR_RETURN(cts.size() == n_ct_out, Code::ERR_INVALID_ARG);

  std::vector<size_t> targets(split_shape.rows());
  for (size_t i = 0; i < targets.size(); ++i) {
    targets[i] = i * split_shape.cols();
  }

  auto mask_prg = [&](long wid, size_t start, size_t end) {
    RLWECt zero;
    RLWEPt mask;
    std::vector<U64> coeffs(targets.size());
    mask_vector.Reshape(GetOutShape(meta));
    for (size_t r_blk = start; r_blk < end; ++r_blk) {
      auto &this_ct = cts.at(r_blk);
      CHECK_ERR(sampleRandomMask(targets, coeffs.data(), coeffs.size(), mask,
                                 this_ct.parms_id(), this_ct.is_ntt_form()),
                "RandomMaskPoly");
      evaluator_->sub_plain_inplace(this_ct, mask);

      pk_encryptor_->encrypt_zero(this_ct.parms_id(), zero);
      evaluator_->add_inplace(this_ct, zero);

      auto row_bgn = r_blk * split_shape.rows();
      auto row_end = std::min<size_t>(row_bgn + split_shape.rows(),
                                      meta.weight_shape.rows());
      auto coeffs_ptr = coeffs.data();
      for (size_t r = row_bgn; r < row_end; ++r) {
        mask_vector(r) = *coeffs_ptr++;
      }
    }

    seal::util::seal_memzero(coeffs.data(), sizeof(uint64_t) * coeffs.size());
    seal::util::seal_memzero(mask.data(),
                             sizeof(uint64_t) * mask.coeff_count());
    return Code::OK;
  };

  return LaunchWorks(tpool, n_ct_out, mask_prg);
}

// In our Cheetah paper, we export the needed coefficients using the Extract
// function. Indeed this semantic can be implemented by zero-out the un-used
// coefficients.
Code HomFCSS::removeUnusedCoeffs(std::vector<seal::Ciphertext> &cts,
                                 const Meta &meta, double *density) const {
  TensorShape out_shape = GetOutShape(meta);
  ENSURE_OR_RETURN(out_shape.num_elements() > 0, Code::ERR_INVALID_ARG);
  ENSURE_OR_RETURN(context_ && evaluator_, Code::ERR_CONFIG);

  TensorShape split_shape = getSplit(meta, poly_degree());
  const size_t n_ct_out =
      CeilDiv<size_t>(meta.weight_shape.rows(), split_shape.rows());
  ENSURE_OR_RETURN(cts.size() == n_ct_out, Code::ERR_INVALID_ARG);

  if (density) *density = 0.;

  for (size_t r_blk = 0; r_blk < n_ct_out; ++r_blk) {
    auto &this_ct = cts[r_blk];
    auto row_bgn = r_blk * split_shape.rows();
    auto row_end = std::min<size_t>(row_bgn + split_shape.rows(),
                                    meta.weight_shape.rows());
    auto upper = (row_end - row_bgn) * split_shape.cols();

    for (size_t index = 0; index < poly_degree(); ++index) {
      if (index < upper && index % split_shape.cols() == 0) {
        if (density) *density += 1;
        continue;
      }

      auto this_ct_ptr = this_ct.data();
      for (size_t L = 0; L < this_ct.coeff_modulus_size(); ++L) {
        this_ct_ptr[index] = 0;
        this_ct_ptr += poly_degree();
      }
    }
  }

  if (density) *density /= cts.size();
  return Code::OK;
}

Code HomFCSS::sampleRandomMask(const std::vector<size_t> &targets,
                               uint64_t *coeffs_buff, size_t buff_size,
                               seal::Plaintext &mask, seal::parms_id_type pid,
                               bool is_ntt) const {
  using namespace seal::util;
  ENSURE_OR_RETURN(context_, Code::ERR_CONFIG);

  auto cntxt_data = context_->get_context_data(pid);
  ENSURE_OR_RETURN(cntxt_data != nullptr, Code::ERR_INVALID_ARG);
  ENSURE_OR_RETURN(!targets.empty(), Code::ERR_INVALID_ARG);
  ENSURE_OR_RETURN(buff_size >= targets.size(), Code::ERR_OUT_BOUND);

  const size_t N = poly_degree();
  if (std::any_of(targets.begin(), targets.end(),
                  [N](size_t c) { return c >= N; })) {
    return Code::ERR_INVALID_ARG;
  }

  auto parms = cntxt_data->parms();
  mask.parms_id() = seal::parms_id_zero;  // foo SEAL when using BFV
  mask.resize(N);

  auto prng = parms.random_generator()->create();
  const size_t nbytes = mul_safe(mask.coeff_count(), sizeof(uint64_t));
  prng->generate(nbytes, reinterpret_cast<std::byte *>(mask.data()));

  if (IsTwoPower(plain_modulus())) {
    uint64_t mod_mask = plain_modulus() - 1;
    std::transform(mask.data(), mask.data() + mask.coeff_count(), mask.data(),
                   [mod_mask](uint64_t u) { return u & mod_mask; });
  } else {
    // TODO(wen-jie): to use reject sampling to obtain uniform in [0, t).
    modulo_poly_coeffs(mask.data(), mask.coeff_count(), parms.plain_modulus(),
                       mask.data());
  }

  auto coeff_ptr = coeffs_buff;
  for (size_t idx : targets) {
    *coeff_ptr++ = mask[idx];
  }
  return Code::OK;
}

Code HomFCSS::decryptToVector(const std::vector<seal::Ciphertext> &enc_vector,
                              const Meta &meta, Tensor<uint64_t> &out,
                              size_t nthreads) const {
  ENSURE_OR_RETURN(context_ && evaluator_ && sk_, Code::ERR_CONFIG);

  auto split_shape = getSplit(meta, poly_degree());
  auto n_ct_out = CeilDiv<size_t>(meta.weight_shape.rows(), split_shape.rows());

  if (n_ct_out != enc_vector.size()) {
    printf("expect %zd but got %zd\n", n_ct_out, enc_vector.size());
  }

  ENSURE_OR_RETURN(enc_vector.size() == n_ct_out, Code::ERR_INVALID_ARG);
  TensorShape out_shape = GetOutShape(meta);
  out.Reshape(out_shape);
  seal::Decryptor decryptor(*context_, *sk_);
  seal::Plaintext pt;
  for (size_t r_blk = 0; r_blk < n_ct_out; ++r_blk) {
    decryptor.decrypt(enc_vector.at(r_blk), pt);
    auto row_bgn = r_blk * split_shape.rows();
    auto row_end = std::min<size_t>(row_bgn + split_shape.rows(),
                                    meta.weight_shape.rows());

    for (size_t r = row_bgn; r < row_end; ++r) {
      size_t coeff_idx = (r - row_bgn) * split_shape.cols();
      out(r) = coeff_idx >= pt.coeff_count() ? 0 : pt[coeff_idx];
    }
  }

  return Code::OK;
}

Code HomFCSS::idealFunctionality(const Tensor<uint64_t> &input_matrix,
                                 const Tensor<uint64_t> &weight_matrix,
                                 const Meta &meta,
                                 Tensor<uint64_t> &out) const {
  return Code::OK;
}
}  // namespace gemini
