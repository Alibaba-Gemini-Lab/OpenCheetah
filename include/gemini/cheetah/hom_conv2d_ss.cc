//  Authors: Wen-jie Lu on 2021/9/11.
#include "gemini/cheetah/hom_conv2d_ss.h"

#include <seal/seal.h>
#include <seal/secretkey.h>
#include <seal/util/polyarithsmallmod.h>
#include <seal/util/rlwe.h>

#include <functional>

#include "gemini/cheetah/tensor_encoder.h"
#include "gemini/core/logging.h"
#include "gemini/core/util/ThreadPool.h"

#define BFV_TRUNCATE_LARGE 1
#define BFV_TRUNCATE_SMALL 1

namespace gemini {

static TensorShape GetConv2DOutShape(const HomConv2DSS::Meta &meta) {
  auto o = shape_inference::Conv2D(meta.ishape, meta.fshape, meta.padding,
                                   meta.stride);
  if (!o) {
    LOG(WARNING) << "GetConv2DOutShape failed";
    return TensorShape({0, 0, 0});
  }
  o->Update(0, meta.n_filters);
  return *o;
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

static inline uint64_t make_bits_mask(int n_low_zeros) {
  n_low_zeros = std::max(0, n_low_zeros);
  n_low_zeros = std::min(63, n_low_zeros);
  return (static_cast<uint64_t>(-1) >> n_low_zeros) << n_low_zeros;
}

namespace internal {
void sub_poly_inplace(seal::Ciphertext &ct, const seal::Plaintext &pt,
                      const seal::SEALContext &context,
                      const seal::Evaluator &evaluator) {
  if (ct.size() != 2) {
    LOG(FATAL) << "sub_poly_inplace: invalid ct.size()";
    return;
  }

  if (pt.parms_id() == seal::parms_id_zero) {
    if (pt.coeff_count() != ct.poly_modulus_degree()) {
      LOG(FATAL) << "sub_poly_inplace: invalid pt.coeff_count()";
      return;
    }
    evaluator.sub_plain_inplace(ct, pt);
    return;
  }

  auto n = ct.poly_modulus_degree();
  auto L = ct.coeff_modulus_size();
  if (pt.coeff_count() != n * L) {
    LOG(FATAL) << "sub_poly_inplace: invalid pt.coeff_count()";
    return;
  }

  auto cntxt = context.get_context_data(ct.parms_id());
  if (!cntxt) {
    LOG(FATAL) << "sub_poly_inplace: invalid ct.parms_id()";
  }

  auto &coeff_modulus = cntxt->parms().coeff_modulus();
  auto src_ptr = pt.data();
  auto dst_ptr = ct.data(0);
  for (size_t l = 0; l < L; ++l) {
    seal::util::sub_poly_coeffmod(dst_ptr, src_ptr, n, coeff_modulus[l],
                                  dst_ptr);
    dst_ptr += n;
    src_ptr += n;
  }
}

};  // namespace internal

void remove_unused_coeffs(seal::Ciphertext &ct,
                          const seal::Evaluator &evaluator,
                          std::vector<size_t> used_indices) {
  if (ct.size() == 0) return;

  if (ct.is_ntt_form()) {
    evaluator.transform_from_ntt_inplace(ct);
  }

  size_t N = ct.poly_modulus_degree();
  size_t L = ct.coeff_modulus_size();
  if (used_indices.empty() || used_indices.size() > N) {
    LOG(FATAL) << "remove_unused_coeffs: invalid used_indices";
  }
  if (std::any_of(used_indices.begin(), used_indices.end(),
                  [N](size_t c) { return c >= N; })) {
    LOG(FATAL) << "remove_unused_coeffs: invalid used_indices";
  }

  std::sort(used_indices.begin(), used_indices.end());
  for (size_t index = 0; index < N; ++index) {
    // skip the needed coefficients
    if (std::binary_search(used_indices.cbegin(), used_indices.cend(), index)) {
      continue;
    }
    // zero-out the un-used coefficients
    auto rns_ptr = ct.data(0);
    for (size_t l = 0; l < L; ++l) {
      rns_ptr[index] = 0;
      rns_ptr += N;
    }
  }
}

static void set_poly_coeffs_uniform(
    uint64_t *poly, int bitlen,
    std::shared_ptr<seal::UniformRandomGenerator> prng,
    const seal::EncryptionParameters &parms) {
  using namespace seal::util;
  if (bitlen < 0 || bitlen > 64) {
    LOG(WARNING) << "set_poly_coeffs_uniform invalid bitlen";
  }

  auto &coeff_modulus = parms.coeff_modulus();
  size_t coeff_count = parms.poly_modulus_degree();
  size_t coeff_mod_count = coeff_modulus.size();
  uint64_t bitlen_mask = (1ULL << (bitlen % 64)) - 1;

  // sample random in [0, 2^bitlen) then convert it to the RNS form
  const size_t nbytes = mul_safe(coeff_count, sizeof(uint64_t));
  if (prng) {
    prng->generate(nbytes, reinterpret_cast<seal::seal_byte *>(poly));
  } else {
    auto _prng = parms.random_generator()->create();
    _prng->generate(nbytes, reinterpret_cast<seal::seal_byte *>(poly));
  }

  uint64_t *dst_ptr = poly + coeff_count;
  for (size_t j = 1; j < coeff_mod_count; ++j) {
    const uint64_t *src_ptr = poly;
    for (size_t i = 0; i < coeff_count; ++i, ++src_ptr) {
      *dst_ptr++ = barrett_reduce_64(*src_ptr & bitlen_mask, coeff_modulus[j]);
    }
  }

  dst_ptr = poly;
  for (size_t i = 0; i < coeff_count; ++i, ++dst_ptr) {
    *dst_ptr = barrett_reduce_64(*dst_ptr & bitlen_mask, coeff_modulus[0]);
  }
}

#if USE_APPROX_RESHARE
Code sample_random_mask(const std::vector<size_t> &targets,
                        uint64_t *coeffs_buff, size_t buff_size,
                        seal::Plaintext &mask, seal::parms_id_type pid,
                        std::shared_ptr<seal::UniformRandomGenerator> prng,
                        const seal::SEALContext &context) {
  using namespace seal::util;

  auto cntxt_data = context.get_context_data(pid);
  ENSURE_OR_RETURN(cntxt_data != nullptr, Code::ERR_INVALID_ARG);
  ENSURE_OR_RETURN(!targets.empty(), Code::ERR_INVALID_ARG);
  ENSURE_OR_RETURN(buff_size >= targets.size(), Code::ERR_OUT_BOUND);
  ENSURE_OR_RETURN(coeffs_buff != nullptr, Code::ERR_NULL_POINTER);
  ENSURE_OR_RETURN(prng != nullptr, Code::ERR_NULL_POINTER);

  auto parms = cntxt_data->parms();
  auto &modulus = parms.coeff_modulus();
  const size_t N = parms.poly_modulus_degree();
  const size_t L = modulus.size();
  if (std::any_of(targets.begin(), targets.end(),
                  [N](size_t c) { return c >= N; })) {
    return Code::ERR_INVALID_ARG;
  }

  mask.parms_id() = seal::parms_id_zero;  // foo SEAL when using BFV
  mask.resize(N * L);
  // sample in [0, Q) in RNS form
  sample_poly_uniform(prng, parms, mask.data());
  mask.parms_id() = cntxt_data->parms_id();

  std::vector<uint64_t> coeffs_rns(targets.size() * L);
  auto src_ptr = mask.data();
  auto dst_ptr = coeffs_rns.data();
  for (size_t l = 0; l < L; ++l) {
    for (size_t idx : targets) {
      *dst_ptr++ = src_ptr[idx];
    }
    src_ptr += N;
  }
  // r in [0, Q) -> round(Q/t*r) mod t
  ConstRNSIter wrap(coeffs_rns.data(), targets.size());
  cntxt_data->rns_tool()->decrypt_scale_and_round(
      wrap, coeffs_buff, seal::MemoryManager::GetPool());
  return Code::OK;
}
#else
Code sample_random_mask(const std::vector<size_t> &targets,
                        uint64_t *coeffs_buff, size_t buff_size,
                        seal::Plaintext &mask, seal::parms_id_type pid,
                        std::shared_ptr<seal::UniformRandomGenerator> prng,
                        const seal::SEALContext &context) {
  using namespace seal::util;

  auto cntxt_data = context.get_context_data(pid);
  ENSURE_OR_RETURN(cntxt_data != nullptr, Code::ERR_INVALID_ARG);
  ENSURE_OR_RETURN(!targets.empty(), Code::ERR_INVALID_ARG);
  ENSURE_OR_RETURN(buff_size >= targets.size(), Code::ERR_OUT_BOUND);
  ENSURE_OR_RETURN(coeffs_buff != nullptr, Code::ERR_NULL_POINTER);
  ENSURE_OR_RETURN(prng != nullptr, Code::ERR_NULL_POINTER);

  auto parms = cntxt_data->parms();
  const size_t N = parms.poly_modulus_degree();
  if (std::any_of(targets.begin(), targets.end(),
                  [N](size_t c) { return c >= N; })) {
    return Code::ERR_INVALID_ARG;
  }

  mask.parms_id() = seal::parms_id_zero;  // foo SEAL when using BFV
  mask.resize(N);

  const size_t nbytes = mul_safe(mask.coeff_count(), sizeof(uint64_t));
  if (prng) {
    prng->generate(nbytes, reinterpret_cast<std::byte *>(mask.data()));
  } else {
    auto _prng = parms.random_generator()->create();
    _prng->generate(nbytes, reinterpret_cast<std::byte *>(mask.data()));
  }

  const auto &t = parms.plain_modulus();
  if (IsTwoPower(t.value())) {
    uint64_t mod_mask = t.value() - 1;
    std::transform(mask.data(), mask.data() + mask.coeff_count(), mask.data(),
                   [mod_mask](uint64_t u) { return u & mod_mask; });
  } else {
    // TODO(wen-jie): to use reject sampling to obtain uniform in [0, t).
    modulo_poly_coeffs(mask.data(), mask.coeff_count(), t, mask.data());
  }

  auto coeff_ptr = coeffs_buff;
  for (size_t idx : targets) {
    *coeff_ptr++ = mask[idx];
  }
  return Code::OK;
}
#endif

static void asymmetric_encrypt_zero(
    const seal::SEALContext &context, const seal::PublicKey &public_key,
    const seal::parms_id_type parms_id, bool is_ntt_form,
    std::shared_ptr<seal::UniformRandomGenerator> prng,
    seal::Ciphertext &destination) {
  using namespace seal;
  using namespace seal::util;
  // We use a fresh memory pool with `clear_on_destruction' enabled
  MemoryPoolHandle pool =
      MemoryManager::GetPool(mm_prof_opt::mm_force_new, true);

  auto &context_data = *context.get_context_data(parms_id);
  auto &parms = context_data.parms();
  auto &coeff_modulus = parms.coeff_modulus();
  size_t coeff_modulus_size = coeff_modulus.size();
  size_t coeff_count = parms.poly_modulus_degree();
  auto ntt_tables = context_data.small_ntt_tables();
  size_t encrypted_size = public_key.data().size();

  // Make destination have right size and parms_id
  // Ciphertext (c_0,c_1, ...)
  destination.resize(context, parms_id, encrypted_size);
  destination.is_ntt_form() = is_ntt_form;
  destination.scale() = 1.0;

  // Generate u <-- R_3
  auto u(allocate_poly(coeff_count, coeff_modulus_size, pool));
  sample_poly_ternary(prng, parms, u.get());

  // c[j] = u * public_key[j]
  for (size_t i = 0; i < coeff_modulus_size; i++) {
    ntt_negacyclic_harvey_lazy(u.get() + i * coeff_count, ntt_tables[i]);
    for (size_t j = 0; j < encrypted_size; j++) {
      dyadic_product_coeffmod(u.get() + i * coeff_count,
                              public_key.data().data(j) + i * coeff_count,
                              coeff_count, coeff_modulus[i],
                              destination.data(j) + i * coeff_count);

      // Addition with e_0, e_1 is in non-NTT form
      if (!is_ntt_form) {
        inverse_ntt_negacyclic_harvey(destination.data(j) + i * coeff_count,
                                      ntt_tables[i]);
      }
    }
  }

  // Generate e_j <-- chi
  // c[j] = public_key[j] * u + e[j]
#if USE_APPROX_RESHARE
  // NOTE(wen-jie) we skip e[0] here since e[0] is replaced by the secret sharing random.
  for (size_t j = 1; j < encrypted_size; j++) {
#else
  for (size_t j = 0; j < encrypted_size; j++) {
#endif
    SEAL_NOISE_SAMPLER(prng, parms, u.get());
    for (size_t i = 0; i < coeff_modulus_size; i++) {
      // Addition with e_0, e_1 is in NTT form
      if (is_ntt_form) {
        ntt_negacyclic_harvey(u.get() + i * coeff_count, ntt_tables[i]);
      }
      add_poly_coeffmod(
          u.get() + i * coeff_count, destination.data(j) + i * coeff_count,
          coeff_count, coeff_modulus[i], destination.data(j) + i * coeff_count);
    }
  }
}

void flood_ciphertext(seal::Ciphertext &ct,
                      std::shared_ptr<seal::UniformRandomGenerator> prng,
                      const seal::SEALContext &context,
                      const seal::PublicKey &pk,
                      const seal::Evaluator &evaluator) {
  if (ct.size() != 2) {
    LOG(WARNING) << "flood_ciphertext: demands more coeff_modulus";
    return;
  }

  auto cntxt_data = context.get_context_data(ct.parms_id());
  if (!cntxt_data) {
    LOG(WARNING) << "flood_ciphertext: invalid ct.parms_id()";
    return;
  }

#if !USE_APPROX_RESHARE
  // NOTE(wen-jie) adding encryption-zero with larger noise
  auto &parms = cntxt_data->parms();
  const int noise_len = cntxt_data->total_coeff_modulus_bit_count() -
                        parms.plain_modulus().bit_count() - 1;

  auto mempool = seal::MemoryManager::GetPool();
  auto random = seal::util::allocate_uint(
      ct.coeff_modulus_size() * ct.poly_modulus_degree(), mempool);

  set_poly_coeffs_uniform(random.get(), std::min(64, noise_len), prng, parms);

  size_t n = ct.poly_modulus_degree();
  auto dst_ptr = ct.data();
  auto rns_ptr = random.get();
  auto &coeff_modulus = parms.coeff_modulus();
  for (size_t i = 0; i < ct.coeff_modulus_size(); i++) {
    seal::util::add_poly_coeffmod(rns_ptr, dst_ptr, n, coeff_modulus[i],
                                  dst_ptr);
    rns_ptr += n;
    dst_ptr += n;
  }
#endif

  evaluator.mod_switch_to_inplace(ct, context.last_parms_id());

  seal::Ciphertext zero;
  asymmetric_encrypt_zero(context, pk, ct.parms_id(), ct.is_ntt_form(), prng, zero);
  evaluator.add_inplace(ct, zero);
  if (ct.is_ntt_form()) {
    evaluator.transform_from_ntt_inplace(ct);
  }
}

void truncate_for_decryption(seal::Ciphertext &ct,
                             const seal::Evaluator &evaluator,
                             const seal::SEALContext &context) {
  auto context_data = context.last_context_data();
  const auto &parms = context_data->parms();
  if (parms.scheme() != seal::scheme_type::bfv) {
    LOG(WARNING) << "truncate_for_decryption: shceme not supported";
    return;
  }

  if (ct.size() != 2) {
    LOG(WARNING) << "truncate_for_decryption: ct.size() should be 2";
    return;
  }

  // only keep the first modulus
  evaluator.mod_switch_to_inplace(ct, context_data->parms_id());
  if (ct.is_ntt_form()) {
    evaluator.transform_from_ntt_inplace(ct);
  }

  // Hack on BFV decryption formula: c0 + c1*s mod p0 = m' = Delta*m + e ->
  // round(m'/Delta) = m The low-end bits of c0, c1 are useless for decryption,
  // and thus we can truncate those bits
  const size_t poly_n = ct.poly_modulus_degree();
  // Delta := floor(p0/t)
  const int n_delta_bits =
      parms.coeff_modulus()[0].bit_count() - parms.plain_modulus().bit_count();
  const int one_more_bit = IsTwoPower(parms.plain_modulus().value()) ? 0 : 1;
  const uint64_t mask0 = make_bits_mask(n_delta_bits - 1 - one_more_bit);
#if !USE_APPROX_RESHARE || BFV_TRUNCATE_LARGE
  std::transform(ct.data(0), ct.data(0) + poly_n, ct.data(0),
                 [mask0](uint64_t u) { return u & mask0; });
#endif
  // Norm |c1 * s|_infty < |c1|_infty * |s|_infty.
  // The value of |c1|_infty * |s|_infty is heuristically bounded by 12. *
  // Std(|c1|_infty) * Std(|s|_infty) Assume |c1| < B is B-bounded uniform. Then
  // the variance Var(|c1|_infty) = B^2*N/12. We need to make sure the value |c1
  // * s|_infty is NOT overflow Delta.
  constexpr double heuristic_bound = 12.;  // P(|x| > Delta) < 2^−{40}
  int n_var_bits{0};
  // The variance Var(|s|_infty) = 2/3*N since the secret key s is uniform from
  // {-1, 0, 1}.
  n_var_bits = std::log2(heuristic_bound * poly_n * std::sqrt(1 / 18.));
  const uint64_t mask1 = make_bits_mask(n_delta_bits - n_var_bits);
#if !USE_APPROX_RESHARE || BFV_TRUNCATE_SMALL
  std::transform(ct.data(1), ct.data(1) + poly_n, ct.data(1),
                 [mask1](uint64_t u) { return u & mask1; });
#endif
}

seal::scheme_type HomConv2DSS::scheme() const {
  if (context_) {
    return context_->first_context_data()->parms().scheme();
  } else {
    return seal::scheme_type::none;
  }
}

size_t HomConv2DSS::poly_degree() const {
  if (context_) {
    return context_->first_context_data()->parms().poly_modulus_degree();
  } else {
    return 0;
  }
}

uint64_t HomConv2DSS::plain_modulus() const {
  if (context_) {
    return context_->first_context_data()->parms().plain_modulus().value();
  } else {
    return -1;
  }
}

Code HomConv2DSS::setUp(const seal::SEALContext &context,
                        std::optional<seal::SecretKey> sk,
                        std::shared_ptr<seal::PublicKey> pk) {
  context_ = std::make_shared<seal::SEALContext>(context);
  ENSURE_OR_RETURN(context_, Code::ERR_NULL_POINTER);

  if (sk) {
    if (!seal::is_metadata_valid_for(*sk, *context_)) {
      LOG(WARNING) << "HomConv2DSS: invalid secret key for this SEALContext";
      return Code::ERR_INVALID_ARG;
    }

    sk_ = seal::SecretKey(*sk);
    encryptor_ = std::make_shared<seal::Encryptor>(*context_, *sk);
  }

  if (pk) {
    if (!seal::is_metadata_valid_for(*pk, *context_)) {
      LOG(WARNING) << "HomConv2DSS: invalid public key for this SEALContext";
      return Code::ERR_INVALID_ARG;
    }

    pk_ = std::make_shared<seal::PublicKey>(*pk);
  }

  tencoder_ = std::make_shared<TensorEncoder>(*context_);
  evaluator_ = std::make_shared<seal::Evaluator>(*context_);
  return Code::OK;
}

Code HomConv2DSS::encryptImage(
    const Tensor<uint64_t> &img, const Meta &meta,
    std::vector<seal::Serializable<seal::Ciphertext>> &encrypted_img,
    size_t nthreads) const {
  ENSURE_OR_RETURN(context_ && encryptor_ && tencoder_, Code::ERR_CONFIG);
  ENSURE_OR_RETURN(img.shape().IsSameSize(meta.ishape), Code::ERR_DIM_MISMATCH);

  TensorEncoder::Role encode_role = TensorEncoder::Role::none;
  std::vector<seal::Plaintext> polys;
  CHECK_ERR(tencoder_->EncodeImageShare(encode_role, img, meta.fshape,
                                        meta.padding, meta.stride,
                                        /*to_ntt*/ false, polys),
            "encryptImage");

  ThreadPool tpool(std::min(std::max(1UL, nthreads), kMaxThreads));
  seal::Serializable<seal::Ciphertext> dummy = encryptor_->encrypt_zero();
  encrypted_img.resize(polys.size(), dummy);
  auto encrypt_program = [&](long wid, size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
      encrypted_img[i] = encryptor_->encrypt_symmetric(polys[i]);
    }
    return Code::OK;
  };

  return LaunchWorks(tpool, polys.size(), encrypt_program);
}

Code HomConv2DSS::encodeImage(const Tensor<uint64_t> &img, const Meta &meta,
                              std::vector<seal::Plaintext> &encoded_img,
                              size_t nthreads) const {
  ENSURE_OR_RETURN(context_ && tencoder_, Code::ERR_CONFIG);
  ENSURE_OR_RETURN(img.shape().IsSameSize(meta.ishape), Code::ERR_DIM_MISMATCH);

  CHECK_ERR(tencoder_->EncodeImageShare(TensorEncoder::Role::evaluator, img,
                                        meta.fshape, meta.padding, meta.stride,
                                        /*to_ntt*/ false, encoded_img),
            "HomConv2DSS::encryptImage: encode failed");
  return Code::OK;
}

Code HomConv2DSS::encodeFilters(
    const std::vector<Tensor<uint64_t>> &filters, const Meta &meta,
    std::vector<std::vector<seal::Plaintext>> &encoded_filters,
    size_t nthreads) const {
  const size_t M = filters.size();
  ENSURE_OR_RETURN(M > 0 && M == meta.n_filters, Code::ERR_INVALID_ARG);
  for (const auto &f : filters) {
    ENSURE_OR_RETURN(f.shape().IsSameSize(meta.fshape), Code::ERR_DIM_MISMATCH);
  }

  encoded_filters.resize(M);
  const bool to_ntt = scheme() == seal::scheme_type::ckks;
  auto encode_program = [&](long wid, size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
      CHECK_ERR(
          tencoder_->EncodeFilter(filters[i], meta.ishape, meta.padding,
                                  meta.stride, to_ntt, encoded_filters[i]),
          "EncodeFilter");
    }
    return Code::OK;
  };

  ThreadPool tpool(std::min(std::max(1UL, nthreads), kMaxThreads));
  return LaunchWorks(tpool, M, encode_program);
}

size_t HomConv2DSS::conv2DOneFilter(const std::vector<seal::Ciphertext> &image,
                                    const std::vector<seal::Plaintext> &filter,
                                    const Meta &meta,
                                    seal::Ciphertext *out_buff,
                                    size_t out_buff_sze) const {
  if (!evaluator_) {
    LOG(WARNING) << "conv2DOneFilter: evaluator is absent";
    return size_t(-1);
  }

  size_t nnz = std::accumulate(filter.cbegin(), filter.cend(), 0,
                               [](size_t nnz, const seal::Plaintext &f) {
                                 return nnz + (f.is_zero() ? 0 : 1);
                               });
  if (nnz == 0) {
    LOG(WARNING) << "conv2DOneFilter: filter with all zero is not supported";
    return size_t(-1);
  }

  const size_t out_size = image.size() / filter.size();
  if (out_size < 1) {
    return size_t(-1);
  }

  if (out_size > out_buff_sze || !out_buff) {
    LOG(WARNING) << "conv2DOneFilter: require a larger out_buff";
    return size_t(-1);
  }

  const size_t accum_cnt = filter.size();
  for (size_t i = 0; i < out_size; ++i) {
    out_buff[i].release();
  }

  for (size_t c = 0; c < accum_cnt; ++c) {
    // filter on the margin might be all-zero
    if (filter[c].is_zero()) {
      continue;
    }

    for (size_t i = 0; i < out_size; ++i) {
      size_t ii = c * out_size + i;
      size_t o = ii % out_size;

      if (out_buff[o].size() > 0) {
        // TODO Use FMA. out_buf[o] += tensor[ii] * filter[c];
        auto cpy_ct{image.at(ii)};
        evaluator_->multiply_plain_inplace(cpy_ct, filter.at(c));
        evaluator_->add_inplace(out_buff[o], cpy_ct);
      } else {
        evaluator_->multiply_plain(image.at(ii), filter.at(c), out_buff[o]);
      }
    }
  }

  return out_size;
}

Code HomConv2DSS::conv2DSS(
    const std::vector<seal::Ciphertext> &img_share0,
    const std::vector<seal::Plaintext> &img_share1,
    const std::vector<std::vector<seal::Plaintext>> &filters, const Meta &meta,
    std::vector<seal::Ciphertext> &out_share0, Tensor<uint64_t> &out_share1,
    size_t nthreads) const {
  if (filters.size() != meta.n_filters) {
    LOG(WARNING) << "conv2DSS: #filters " << filters.size()
                 << " != " << meta.n_filters << "\n";
    return Code::ERR_DIM_MISMATCH;
  }

  if (meta.is_shared_input && img_share0.size() != img_share1.size()) {
    LOG(WARNING) << "conv2DSS: #shares " << img_share0.size()
                 << " != " << img_share1.size() << "\n";
    return Code::ERR_DIM_MISMATCH;
  }

  ENSURE_OR_RETURN(filters.size() == meta.n_filters, Code::ERR_DIM_MISMATCH);
  if (meta.is_shared_input) {
    ENSURE_OR_RETURN(img_share0.size() == img_share1.size(),
                     Code::ERR_DIM_MISMATCH);
  }

  TensorShape out_shape = GetConv2DOutShape(meta);
  if (out_shape.num_elements() == 0) {
    LOG(WARNING) << "conv2DSS: empty out_shape";
    return Code::ERR_CONFIG;
  }

  auto tl_pool =
      seal::MemoryManager::GetPool(seal::mm_prof_opt::mm_force_thread_local);

  std::vector<seal::Ciphertext> image;
  auto add_program = [&](long wid, size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
      try {
        evaluator_->add_plain(img_share0[i], img_share1[i], image[i]);
      } catch (std::logic_error e) {
        LOG(WARNING) << "SEAL ERROR: " << e.what();
        return Code::ERR_INTERNAL;
      }
    }
    return Code::OK;
  };

  ThreadPool tpool(std::min(std::max(1UL, nthreads), kMaxThreads));
  if (meta.is_shared_input) {
    image.resize(img_share0.size(), seal::Ciphertext(tl_pool));
    CHECK_ERR(LaunchWorks(tpool, image.size(), add_program), "add");
  }

  const size_t N = poly_degree();
  ConvCoeffIndexCalculator indexer(N, meta.ishape, meta.fshape, meta.padding,
                                   meta.stride);
  const size_t n_one_channel = indexer.slice_size(1) * indexer.slice_size(2);
  const size_t n_out_ct = meta.n_filters * n_one_channel;
  out_share0.resize(n_out_ct);
  auto conv_program = [&](long wid, size_t start, size_t end) {
    for (size_t m = start; m < end; ++m) {
      seal::Ciphertext *ct_start = &out_share0.at(m * n_one_channel);
      size_t used = conv2DOneFilter(meta.is_shared_input ? image : img_share0,
                                    filters[m], meta, ct_start, n_one_channel);
      if (used == (size_t)-1 || used != n_one_channel) {
        return Code::ERR_INTERNAL;
      }
    };

    return Code::OK;
  };

  CHECK_ERR(LaunchWorks(tpool, meta.n_filters, conv_program), "conv2D");

  out_share1.Reshape(out_shape);
  addRandomMask(out_share0, out_share1, meta, nthreads);

  if (scheme() == seal::scheme_type::bfv) {
    auto truncate_program = [&](long wid, size_t start, size_t end) {
      for (size_t cid = start; cid < end; ++cid) {
        truncate_for_decryption(out_share0[cid], *evaluator_, *context_);
      }
      return Code::OK;
    };

    CHECK_ERR(LaunchWorks(tpool, out_share0.size(), truncate_program),
              "conv2D");
  }

  // Post-processing for compressing out_ct volume.
  removeUnusedCoeffs(out_share0, meta);
  return Code::OK;
}

Code HomConv2DSS::sampleRandomMask(
    const std::vector<size_t> &targets, uint64_t *coeffs_buff, size_t buff_size,
    seal::Plaintext &mask, seal::parms_id_type pid,
    std::shared_ptr<seal::UniformRandomGenerator> prng, bool is_ntt) const {
  ENSURE_OR_RETURN(context_, Code::ERR_CONFIG);
  return sample_random_mask(targets, coeffs_buff, buff_size, mask, pid, prng,
                            *context_);
}

Code HomConv2DSS::addRandomMask(std::vector<seal::Ciphertext> &enc_tensor,
                                Tensor<uint64_t> &mask_tensor, const Meta &meta,
                                size_t nthreads) const {
  ENSURE_OR_RETURN(pk_, Code::ERR_CONFIG);

  TensorShape strided_ishape;
  std::array<int, 2> pads{0};
  std::array<int, 3> slice_width{0};
  if (!shape_inference::Conv2D(meta.ishape, meta.fshape, poly_degree(),
                               meta.padding, meta.stride, strided_ishape, pads,
                               slice_width)) {
    LOG(WARNING) << "addRandomMask: shape inference failed";
    return Code::ERR_INTERNAL;
  }

  bool is_input_compressed =
      strided_ishape.num_elements() < meta.ishape.num_elements();

  ConvCoeffIndexCalculator indexer(
      poly_degree(), is_input_compressed ? strided_ishape : meta.ishape,
      meta.fshape, is_input_compressed ? Padding::VALID : meta.padding,
      is_input_compressed ? 1 : meta.stride);

  const size_t n_one_channel = indexer.slice_size(1) * indexer.slice_size(2);
  if (enc_tensor.size() != meta.n_filters * n_one_channel) {
    LOG(WARNING) << "addRandomMask: ct.size() mismtach";
    return Code::ERR_INTERNAL;
  }

  mask_tensor.Reshape(GetConv2DOutShape(meta));
  auto mask_program = [&](long wid, size_t start, size_t end) {
    RLWEPt mask;
    TensorShape slice_shape;
    std::vector<size_t> targets;
    std::vector<U64> coeffs(poly_degree());

    auto prng =
        context_->first_context_data()->parms().random_generator()->create();
    for (size_t m = start; m < end; ++m) {
      size_t cid = m * n_one_channel;
      for (int sh = 0, hoffset = 0; sh < indexer.slice_size(1); ++sh) {
        for (int sw = 0, woffset = 0; sw < indexer.slice_size(2); ++sw) {
          CHECK_ERR(indexer.Get({sh, sw}, slice_shape, targets), "Get");

          if (slice_shape.height() * slice_shape.width() != targets.size()) {
            return Code::ERR_DIM_MISMATCH;
          }
          auto &this_ct = enc_tensor.at(cid++);

          flood_ciphertext(this_ct, prng, *context_, *pk_, *evaluator_);
          CHECK_ERR(
              sampleRandomMask(targets, coeffs.data(), coeffs.size(), mask,
                               this_ct.parms_id(), prng, this_ct.is_ntt_form()),
              "RandomMaskPoly");
          internal::sub_poly_inplace(this_ct, mask, *context_, *evaluator_);

          auto coeff_ptr = coeffs.data();
          for (long h = 0; h < slice_shape.height(); ++h) {
            for (long w = 0; w < slice_shape.width(); ++w) {
              mask_tensor(m, hoffset + h, woffset + w) = *coeff_ptr++;
            }
          }
          woffset += slice_shape.width();
        }
        hoffset += slice_shape.height();
      }
    }
    return Code::OK;
  };

  ThreadPool tpool(std::min(std::max(1UL, nthreads), kMaxThreads));
  return LaunchWorks(tpool, meta.n_filters, mask_program);
}

// In our Cheetah paper, we export the needed coefficients using the Extract
// function. Indeed this semantic can be implemented by zero-out the un-used
// coefficients.
Code HomConv2DSS::removeUnusedCoeffs(std::vector<seal::Ciphertext> &ct,
                                     const Meta &meta, double *density) const {
  TensorShape out_shape = GetConv2DOutShape(meta);
  ENSURE_OR_RETURN(out_shape.num_elements() > 0, Code::ERR_INVALID_ARG);
  ENSURE_OR_RETURN(context_ && evaluator_, Code::ERR_CONFIG);

  const size_t N = poly_degree();
  TensorShape strided_ishape;
  std::array<int, 2> pads{0};
  std::array<int, 3> slice_width{0};
  if (!shape_inference::Conv2D(meta.ishape, meta.fshape, N, meta.padding,
                               meta.stride, strided_ishape, pads,
                               slice_width)) {
    LOG(WARNING) << "shape inference failed";
    return Code::ERR_INTERNAL;
  }

  // Treat the strided image with stride = 1
  ConvCoeffIndexCalculator indexer(N, strided_ishape, meta.fshape,
                                   Padding::VALID, 1);
  const size_t one_channel = indexer.slice_size(1) * indexer.slice_size(2);
  if (ct.size() != one_channel * meta.n_filters) {
    LOG(WARNING) << "shape #ct mismatch";
    return Code::ERR_INTERNAL;
  }

  if (density) *density = 0.;

  for (long m = 0; m < meta.n_filters; ++m) {
    size_t ct_idx = m * one_channel;
    TensorShape slice_shape;
    std::vector<size_t> indices;

    for (int sh = 0; sh < indexer.slice_size(1); ++sh) {
      for (int sw = 0; sw < indexer.slice_size(2); ++sw) {
        CHECK_ERR(indexer.Get({sh, sw}, slice_shape, indices), "Get");

        if (slice_shape.height() * slice_shape.width() != indices.size()) {
          LOG(WARNING) << "slice_shape.size != indices.size";
          return Code::ERR_INTERNAL;
        }

        auto &this_ct = ct.at(ct_idx++);
        if (density) *density += (indices.size() * 1. / N);
        remove_unused_coeffs(this_ct, *evaluator_, indices);
      }
    }
  }

  if (density) *density /= ct.size();
  return Code::OK;
}

Code HomConv2DSS::decryptToTensor(
    const std::vector<seal::Ciphertext> &enc_tensor, const Meta &meta,
    Tensor<uint64_t> &out_tensor, size_t nthreads) const {
  if (!sk_) {
    LOG(FATAL) << "decrypt without sk";
  }
  ENSURE_OR_RETURN(context_ && sk_ && evaluator_, Code::ERR_CONFIG);

  const size_t N = poly_degree();
  TensorShape out_shape = GetConv2DOutShape(meta);
  ENSURE_OR_RETURN(out_shape.num_elements() > 0, Code::ERR_INVALID_ARG);
  TensorShape strided_ishape;
  std::array<int, 2> pads{0};
  std::array<int, 3> slice_width{0};
  if (!shape_inference::Conv2D(meta.ishape, meta.fshape, N, meta.padding,
                               meta.stride, strided_ishape, pads,
                               slice_width)) {
    LOG(WARNING) << "decryptToTensor: shape inference failed";
    return Code::ERR_INTERNAL;
  }

  bool is_input_compressed =
      strided_ishape.num_elements() < meta.ishape.num_elements();

  ConvCoeffIndexCalculator indexer(
      N, is_input_compressed ? strided_ishape : meta.ishape, meta.fshape,
      is_input_compressed ? Padding::VALID : meta.padding,
      is_input_compressed ? 1 : meta.stride);

  const size_t n_one_channel = indexer.slice_size(1) * indexer.slice_size(2);
  if (enc_tensor.size() != meta.n_filters * n_one_channel) {
    return Code::ERR_INTERNAL;
  }

  out_tensor.Reshape(out_shape);
  const bool need_ntt_form_ct = scheme() == seal::scheme_type::ckks;
  seal::Decryptor decryptor(*context_, *sk_);
  auto decrypt_program = [&](long wid, size_t start, size_t end) {
    RLWEPt pt;
    TensorShape slice_shape;
    std::vector<size_t> indices;
    std::vector<U64> coeffs(N);

    for (size_t m = start; m < end; ++m) {
      size_t cid = m * n_one_channel;
      for (int sh = 0, hoffset = 0; sh < indexer.slice_size(1); ++sh) {
        for (int sw = 0, woffset = 0; sw < indexer.slice_size(2); ++sw) {
          CHECK_ERR(indexer.Get({sh, sw}, slice_shape, indices), "Get");
          if (slice_shape.height() * slice_shape.width() != indices.size()) {
            return Code::ERR_DIM_MISMATCH;
          }

          if (need_ntt_form_ct == enc_tensor[cid].is_ntt_form()) {
            decryptor.decrypt(enc_tensor[cid], pt);
          } else {
            RLWECt cpy{enc_tensor[cid]};
            if (need_ntt_form_ct) {
              evaluator_->transform_to_ntt_inplace(cpy);
            } else {
              evaluator_->transform_from_ntt_inplace(cpy);
            }
            decryptor.decrypt(cpy, pt);
          }
          ++cid;

          // decrypt then take the needed coefficients
          CHECK_ERR(
              postProcessInplace(pt, indices, coeffs.data(), coeffs.size()),
              "ConvertThenModSwitch");

          auto coeff_ptr = coeffs.cbegin();
          for (long h = 0; h < slice_shape.height(); ++h) {
            for (long w = 0; w < slice_shape.width(); ++w) {
              out_tensor(m, hoffset + h, woffset + w) = *coeff_ptr++;
            }
          }

          woffset += slice_shape.width();
        }
        hoffset += slice_shape.height();
      }
    }

    return Code::OK;
  };

  ThreadPool tpool(nthreads);
  return LaunchWorks(tpool, meta.n_filters, decrypt_program);
}

Code HomConv2DSS::postProcessInplace(seal::Plaintext &pt,
                                     std::vector<size_t> &targets,
                                     uint64_t *out_poly,
                                     size_t out_buff_size) const {
  using namespace seal::util;
  ENSURE_OR_RETURN(context_, Code::ERR_CONFIG);

  auto cntxt = context_->first_context_data();
  ENSURE_OR_RETURN(cntxt, Code::ERR_SEAL_MEMORY);

  const size_t N = cntxt->parms().poly_modulus_degree();
  const size_t L = cntxt->parms().coeff_modulus().size();
  const size_t len = targets.empty() ? N : targets.size();
  ENSURE_OR_RETURN(out_poly && out_buff_size >= len && len <= N,
                   Code::ERR_DIM_MISMATCH);

  if (scheme() != seal::scheme_type::ckks) {
    if (targets.empty()) {
      std::copy_n(pt.data(), pt.coeff_count(), out_poly);
    } else {
      const size_t n = pt.coeff_count();
      for (size_t idx : targets) {
        *out_poly++ = idx >= n ? 0 : pt[idx];
      }
    }
    return Code::OK;
  } else {
    std::cerr << "Not implemented yet." << std::endl;
    return Code::ERR_INTERNAL;
  }
}

Code HomConv2DSS::idealFunctionality(
    const Tensor<uint64_t> &in_tensor,
    const std::vector<Tensor<uint64_t>> &filters, const Meta &meta,
    Tensor<uint64_t> &out_tensor) const {
  ENSURE_OR_RETURN(meta.ishape.IsSameSize(in_tensor.shape()),
                   Code::ERR_DIM_MISMATCH);
  ENSURE_OR_RETURN(meta.n_filters == filters.size(), Code::ERR_DIM_MISMATCH);
  for (const auto &f : filters) {
    ENSURE_OR_RETURN(meta.fshape.IsSameSize(f.shape()), Code::ERR_DIM_MISMATCH);
  }

  TensorShape out_shape = GetConv2DOutShape(meta);
  ENSURE_OR_RETURN(out_shape.num_elements() > 0, Code::ERR_DIM_MISMATCH);

  out_tensor.Reshape(out_shape);
  const uint64_t base_mod = plain_modulus();
  ENSURE_OR_RETURN(base_mod != -1, Code::ERR_INTERNAL);

  for (size_t m = 0; m < meta.n_filters; ++m) {
    Tensor<uint64_t> one_channel;
    if (base_mod > 1) {
      // mod p
      in_tensor.Conv2D(filters[m], meta.stride, meta.padding, one_channel,
                       base_mod);
    } else {
      // mod 2^64
      in_tensor.Conv2D(filters[m], meta.stride, meta.padding, one_channel);
    }
    std::array<int64_t, 3> offset = {(int64_t)m, 0, 0};
    std::array<int64_t, 3> extent{0};
    for (int d : {0, 1, 2}) {
      extent[d] = one_channel.dim_size(d);
    }
    out_tensor.tensor().slice(offset, extent) = one_channel.tensor();
  }

  return Code::OK;
}

}  // namespace gemini
