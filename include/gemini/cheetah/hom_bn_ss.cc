//  Authors: Wen-jie Lu on 2021/9/15.
#include "gemini/cheetah/hom_bn_ss.h"

#include <seal/seal.h>
#include <seal/secretkey.h>
#include <seal/util/polyarithsmallmod.h>
#include <seal/util/rlwe.h>

#include "gemini/cheetah/sliced_3d_tensor.h"
#include "gemini/core/logging.h"
#include "gemini/core/util/ThreadPool.h"
#include "gemini/core/util/math.h"
#include "gemini/core/util/timer.h"

namespace gemini {

static TensorShape GetOutShape(const HomBNSS::Meta &meta) {
  if (meta.vec_shape.dims() != 1 || meta.vec_shape.num_elements() < 1) {
    LOG(WARNING) << "invalid meta for BN";
  }
  return meta.vec_shape;
}

// defined in hom_conv2d_ss.cc
void remove_unused_coeffs(seal::Ciphertext &ct,
                          const seal::Evaluator &evaluator,
                          std::vector<size_t> used_indices);

void truncate_for_decryption(seal::Ciphertext &ct,
                             const seal::Evaluator &evaluator,
                             const seal::SEALContext &context);

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

TensorShape getSplitBN(const TensorShape &ishape, size_t N) {
  // NOTE(wen-jie) current implementation does not split along the C-axis
  int64_t n = static_cast<int64_t>(std::sqrt(N));
  int64_t h = std::min(n, ishape.height());
  int64_t w = std::min(n, ishape.width());
  return TensorShape({1, h, w});
}

TensorShape HomBNSS::getSplit(const Meta &meta) const {
  int64_t nCRT = contexts_.size();
  int64_t sub_vec_len =
      std::min<int64_t>(meta.vec_shape.length(), poly_degree());
  return TensorShape({nCRT, sub_vec_len});
}

Code HomBNSS::setUp(uint64_t target_base_mod, const seal::SEALContext &context,
                    std::optional<seal::SecretKey> sk,
                    std::shared_ptr<seal::PublicKey> pk) {
  direct_context_ = std::make_shared<seal::SEALContext>(context);
  ENSURE_OR_RETURN(direct_context_, Code::ERR_NULL_POINTER);
  auto parms = direct_context_->first_context_data()->parms();
  scheme_ = parms.scheme();
  target_base_mod_ = target_base_mod;
  poly_degree_ = parms.poly_modulus_degree();
  if (target_base_mod_ != parms.plain_modulus().value()) {
    LOG(WARNING) << "HomBNSS: invalid target base mod this SEALContext";
    return Code::ERR_INVALID_ARG;
  }

  if (sk) {
    if (!seal::is_metadata_valid_for(*sk, *direct_context_)) {
      LOG(WARNING) << "HomBNSS: invalid secret key for this SEALContext";
      return Code::ERR_INVALID_ARG;
    }

    direct_sk_ = seal::SecretKey(*sk);
    direct_encryptor_ =
        std::make_shared<seal::Encryptor>(*direct_context_, *sk);
  }

  if (pk) {
    if (!seal::is_metadata_valid_for(*pk, *direct_context_)) {
      LOG(WARNING) << "HomBNSS: invalid public key for this SEALContext";
      return Code::ERR_INVALID_ARG;
    }

    direct_pk_encryptor_ =
        std::make_shared<seal::Encryptor>(*direct_context_, *pk);
  }

  direct_evaluator_ = std::make_shared<seal::Evaluator>(*direct_context_);
  return Code::OK;
}

Code HomBNSS::setUp(uint64_t target_base_mod,
                    const std::vector<seal::SEALContext> &contexts,
                    std::vector<std::optional<seal::SecretKey>> sks,
                    std::vector<std::shared_ptr<seal::PublicKey>> pks) {
  ENSURE_OR_RETURN(!contexts.empty(), Code::ERR_CONFIG);
  ENSURE_OR_RETURN(sks.empty() || contexts.size() == sks.size(),
                   Code::ERR_CONFIG);

  const bool is_mod_2k = IsTwoPower(target_base_mod);
  const size_t nCRT = contexts.size();

  std::vector<seal::Modulus> primes(nCRT);
  size_t degree = 0;
  seal::scheme_type scheme{seal::scheme_type::none};
  for (size_t i = 0; i < nCRT; ++i) {
    ENSURE_OR_RETURN(contexts[i].parameters_set(), Code::ERR_CONFIG);
    if (!sks.empty()) {
      ENSURE_OR_RETURN(sks[i] && seal::is_valid_for(*sks[i], contexts[i]),
                       Code::ERR_CONFIG);
    }
    auto &parms = contexts[i].first_context_data()->parms();
    if (i == 0) {
      scheme = parms.scheme();
      degree = parms.poly_modulus_degree();
    } else {
      ENSURE_OR_RETURN(scheme == parms.scheme(), Code::ERR_CONFIG);
      ENSURE_OR_RETURN(degree == parms.poly_modulus_degree(), Code::ERR_CONFIG);
    }
    primes[i] = parms.plain_modulus();
  }

  seal::EncryptionParameters crt_parms(seal::scheme_type::bfv);
  crt_parms.set_plain_modulus(2);  // dummy
  crt_parms.set_poly_modulus_degree(degree);
  crt_parms.set_coeff_modulus(primes);
  // crt_parms.set_use_special_modulus(false);
  seal::SEALContext crt_context(crt_parms, true, seal::sec_level_type::none);
  if (!crt_context.parameters_set()) {
    LOG(FATAL) << "HomBNSS failed to init CRT context. "
               << crt_context.parameter_error_message();
    return Code::ERR_INTERNAL;
  }

  if (!is_mod_2k) {
    if (nCRT != 1 || primes[0].value() != target_base_mod) {
      LOG(FATAL) << "HomBNSS prime field base_mod mismatch "
                 << primes[0].value() << " != " << target_base_mod << " nCRT "
                 << nCRT;
      return Code::ERR_INVALID_ARG;
    }
  } else {
    const int nbits_crt_plain =
        crt_context.key_context_data()->total_coeff_modulus_bit_count() - 1;
    const int n_base_mod_bits = Log2(target_base_mod);
    if (nbits_crt_plain < 2 * n_base_mod_bits + 1 + kStatBits) {
      LOG(FATAL)
          << "HomBNSS require more primes to convert the target nbits of "
             "shares. Require "
          << 2 * n_base_mod_bits + 1 + kStatBits
          << " bits of plaintext but got only " << nbits_crt_plain;
      return Code::ERR_INVALID_ARG;
    }
  }

  scheme_ = scheme;
  poly_degree_ = degree;
  target_base_mod_ = target_base_mod;
  n_base_mod_bits_ = Log2(target_base_mod);
  crt_context_ = std::make_shared<seal::SEALContext>(crt_context);

  contexts_.resize(nCRT);
  encoders_.resize(nCRT);
  evaluators_.resize(nCRT);
  for (size_t i = 0; i < nCRT; ++i) {
    contexts_[i] = std::make_shared<seal::SEALContext>(contexts[i]);
    encoders_[i] = std::make_shared<seal::BatchEncoder>(*contexts_[i]);
    evaluators_[i] = std::make_shared<seal::Evaluator>(*contexts_[i]);
  }

  if (!sks.empty()) {
    sks_.resize(nCRT);
    encryptors_.resize(nCRT);
    for (size_t i = 0; i < nCRT; ++i) {
      sks_[i] = seal::SecretKey(*sks[i]);
      encryptors_[i] =
          std::make_shared<seal::Encryptor>(*contexts_[i], *sks_[i]);
    }
  }

  if (!pks.empty()) {
    pk_encryptors_.resize(nCRT);
    for (size_t i = 0; i < nCRT; ++i) {
      pk_encryptors_[i] =
          std::make_shared<seal::Encryptor>(*contexts_[i], *pks[i]);
    }
  }

  return Code::OK;
}

Code HomBNSS::encryptVector(
    const Tensor<uint64_t> &in_vec, const Meta &meta,
    std::vector<seal::Serializable<seal::Ciphertext>> &out,
    size_t nthreads) const {
  ENSURE_OR_RETURN(
      !contexts_.empty() && !encryptors_.empty() && !encoders_.empty(),
      Code::ERR_CONFIG);
  ENSURE_OR_RETURN(
      in_vec.dims() == 1 && in_vec.shape().IsSameSize(meta.vec_shape),
      Code::ERR_CONFIG);
  ENSURE_OR_RETURN(meta.target_base_mod == target_base_mod_,
                   Code::ERR_OUT_BOUND);

  TensorShape split_shape = getSplit(meta);
  const size_t nCRT = split_shape.dim_size(0);
  const size_t sub_vec_len = split_shape.dim_size(1);
  const size_t n_sub_vecs =
      CeilDiv<size_t>(meta.vec_shape.length(), sub_vec_len);

  seal::Plaintext pt;
  std::vector<uint64_t> tmp(poly_degree());

  seal::Serializable<seal::Ciphertext> dummy = encryptors_[0]->encrypt_zero();
  out.resize(nCRT * n_sub_vecs, dummy);

  auto encrypt_prg = [&](long wid, size_t start, size_t end) {
    seal::Plaintext pt;
    std::vector<uint64_t> tmp(poly_degree());
    for (size_t k = start; k < end; ++k) {
      size_t crt_idx = k / n_sub_vecs;

      auto vec_pos_bgn = (k % n_sub_vecs) * sub_vec_len;
      auto vec_pos_end =
          std::min<size_t>(vec_pos_bgn + sub_vec_len, meta.vec_shape.length());
      auto len = vec_pos_end - vec_pos_bgn;

      std::copy(in_vec.data() + vec_pos_bgn, in_vec.data() + vec_pos_end,
                tmp.data());
      std::fill(tmp.begin() + len, tmp.end(), 0);

      encoders_[crt_idx]->encode(tmp, pt);
      out.at(k) = encryptors_[crt_idx]->encrypt_symmetric(pt);
    }
    seal::util::seal_memzero(tmp.data(), sizeof(uint64_t) * tmp.size());
    seal::util::seal_memzero(pt.data(), sizeof(uint64_t) * pt.coeff_count());
    return Code::OK;
  };

  ThreadPool tpool(nthreads);
  return LaunchWorks(tpool, out.size(), encrypt_prg);

  /// Single thread version
  //  for (size_t i = 0; i < nCRT; ++i) {
  //    for (size_t j = 0; j < n_sub_vecs; ++j) {
  //      auto start = j * sub_vec_len;
  //      auto end = std::min<size_t>(start + sub_vec_len,
  //      meta.vec_shape.length()); std::copy(in_vec.data() + start,
  //      in_vec.data() + end, tmp.data()); std::fill(tmp.begin() + end - start,
  //      tmp.end(), 0); encoders_[i]->encode(tmp, pt); out.at(i * n_sub_vecs +
  //      j) = encryptors_[i]->encrypt_symmetric(pt);
  //    }
  //  }

  return Code::OK;
}

Code HomBNSS::encodeVector(const Tensor<uint64_t> &in_vec, const Meta &meta,
                           std::vector<seal::Plaintext> &out,
                           size_t nthreads) const {
  if (contexts_.empty() || encoders_.empty()) {
    LOG(WARNING) << "encodeVector empty context or encoders";
  }
  if (in_vec.dims() != 1 || !in_vec.shape().IsSameSize(meta.vec_shape)) {
    LOG(WARNING) << "encodeVector invalid in_vec shape " << in_vec.shape();
  }
  ENSURE_OR_RETURN(!contexts_.empty() && !encoders_.empty(), Code::ERR_CONFIG);
  ENSURE_OR_RETURN(
      in_vec.dims() == 1 && in_vec.shape().IsSameSize(meta.vec_shape),
      Code::ERR_CONFIG);
  ENSURE_OR_RETURN(meta.target_base_mod == target_base_mod_,
                   Code::ERR_OUT_BOUND);

  TensorShape split_shape = getSplit(meta);
  const size_t nCRT = split_shape.dim_size(0);
  const size_t sub_vec_len = split_shape.dim_size(1);
  const size_t n_sub_vecs =
      CeilDiv<size_t>(meta.vec_shape.length(), sub_vec_len);
  out.resize(nCRT * n_sub_vecs);

  auto encode_prg = [&](long wid, size_t start, size_t end) {
    std::vector<uint64_t> tmp(poly_degree());
    for (size_t k = start; k < end; ++k) {
      size_t crt_idx = k / n_sub_vecs;
      auto vec_pos_bgn = (k % n_sub_vecs) * sub_vec_len;
      auto vec_pos_end =
          std::min<size_t>(vec_pos_bgn + sub_vec_len, meta.vec_shape.length());
      auto len = vec_pos_end - vec_pos_bgn;

      std::copy(in_vec.data() + vec_pos_bgn, in_vec.data() + vec_pos_end,
                tmp.data());
      std::fill(tmp.begin() + len, tmp.end(), 0);
      encoders_[crt_idx]->encode(tmp, out.at(k));
    }
    seal::util::seal_memzero(tmp.data(), sizeof(uint64_t) * tmp.size());
    return Code::OK;
  };

  ThreadPool tpool(nthreads);
  return LaunchWorks(tpool, out.size(), encode_prg);
}

Code HomBNSS::encodeScales(const Tensor<uint64_t> &scales, const Meta &meta,
                           std::vector<seal::Plaintext> &out,
                           size_t nthreads) const {
  return encodeVector(scales, meta, out, nthreads);
}

Code HomBNSS::bn(const std::vector<seal::Ciphertext> &vec_share0,
                 const std::vector<seal::Plaintext> &vec_share1,
                 const std::vector<seal::Plaintext> &scales, const Meta &meta,
                 std::vector<seal::Ciphertext> &out_share0,
                 Tensor<uint64_t> &out_share1, size_t nthreads) const {
  ENSURE_OR_RETURN(meta.target_base_mod == target_base_mod_,
                   Code::ERR_OUT_BOUND);
  ENSURE_OR_RETURN(!contexts_.empty() && !evaluators_.empty(),
                   Code::ERR_CONFIG);

  TensorShape split_shape = getSplit(meta);
  const size_t nCRT = split_shape.dim_size(0);
  const size_t sub_vec_len = split_shape.dim_size(1);
  const size_t n_sub_vecs =
      CeilDiv<size_t>(meta.vec_shape.length(), sub_vec_len);
  const size_t n_ct = nCRT * n_sub_vecs;
  ENSURE_OR_RETURN(vec_share0.size() == n_ct, Code::ERR_INVALID_ARG);
  ENSURE_OR_RETURN(scales.size() == n_ct, Code::ERR_INVALID_ARG);
  if (meta.is_shared_input) {
    ENSURE_OR_RETURN(vec_share1.size() == n_ct, Code::ERR_INVALID_ARG);
  }

  out_share0.resize(n_ct);

  auto bn_prg = [&](long wid, size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
      size_t crt_idx = i / n_sub_vecs;
      if (meta.is_shared_input) {
        evaluators_[crt_idx]->add_plain(vec_share0[i], vec_share1[i],
                                        out_share0[i]);
        evaluators_[crt_idx]->multiply_plain_inplace(out_share0[i], scales[i]);
      } else {
        evaluators_[crt_idx]->multiply_plain(vec_share0[i], scales[i],
                                             out_share0[i]);
      }
      evaluators_[crt_idx]->mod_switch_to_inplace(
          out_share0[i], contexts_[crt_idx]->last_parms_id());
    }
    return Code::OK;
  };

  ThreadPool tpool(nthreads);
  (void)LaunchWorks(tpool, n_ct, bn_prg);
  return addMask(out_share0, out_share1, meta, tpool);
}

Code HomBNSS::addMaskPrimeField(std::vector<seal::Ciphertext> &cts,
                                Tensor<uint64_t> &mask, const Meta &meta,
                                ThreadPool &tpool) const {
  // TODO Not implemented yet.
  return Code::ERR_INTERNAL;
}

Code HomBNSS::addMaskRing(std::vector<seal::Ciphertext> &cts,
                          Tensor<uint64_t> &mask, const Meta &meta,
                          ThreadPool &tpool) const {
  TensorShape split_shape = getSplit(meta);
  const size_t nCRT = split_shape.dim_size(0);
  const size_t sub_vec_len = split_shape.dim_size(1);
  const size_t n_sub_vecs =
      CeilDiv<size_t>(meta.vec_shape.length(), sub_vec_len);
  const size_t n_ct = nCRT * n_sub_vecs;
  ENSURE_OR_RETURN(cts.size() == n_ct, Code::ERR_INVALID_ARG);

  auto crt_prng =
      crt_context_->key_context_data()->parms().random_generator()->create();
  // sample uniform random in r \in [0, 2^{2k+1+sigma})
  const int n_random_bits = 2 * n_base_mod_bits_ + 1 + kStatBits;
  const int n_random_limbs = CeilDiv(n_random_bits, 64);
  if (n_random_limbs > 2) {
    LOG(FATAL) << "Not implement yet";
  }
  const uint64_t msb_mask = [&]() -> uint64_t {
    int nlow = n_random_limbs * 64 - n_random_bits;
    return nlow == 0 ? static_cast<uint64_t>(-1) : (1ULL << nlow) - 1;
  }();

  std::vector<uint64_t> random(meta.vec_shape.length() * n_random_limbs);
  crt_prng->generate(random.size() * sizeof(uint64_t),
                     reinterpret_cast<seal::seal_byte *>(random.data()));
  for (size_t i = 0; i < random.size(); i += n_random_limbs) {
    random[i + n_random_limbs - 1] &= msb_mask;
  }

  // -r mod 2^k
  mask.Reshape(meta.vec_shape);
  const uint64_t mod_mask = (1ULL << n_base_mod_bits_) - 1;
  for (size_t i = 0, j = 0; i < random.size(); i += n_random_limbs, ++j) {
    mask(j) = (-random[i]) & mod_mask;
  }

  auto crt_primes = crt_context_->key_context_data()->parms().coeff_modulus();
  auto mask_prg = [&](long wid, size_t start, size_t end) {
    // Encode r in [0, 2^{2k + 1 + sigma}) to plaintext
    seal::Plaintext pt;
    std::vector<uint64_t> slots(poly_degree());
    for (size_t i = start; i < end; ++i) {
      size_t crt_idx = i / n_sub_vecs;
      size_t j = (i % n_sub_vecs);
      // nlimbs = 2
      auto rnd_ptr = random.cbegin() + 2 * j * sub_vec_len;
      auto vec_pos_bgn = (i % n_sub_vecs) * sub_vec_len;
      auto vec_pos_end =
          std::min<size_t>(vec_pos_bgn + sub_vec_len, meta.vec_shape.length());
      auto slot_dst = slots.begin();
      for (size_t d = vec_pos_bgn; d < vec_pos_end; ++d) {
        uint64_t u128[2]{*rnd_ptr++, *rnd_ptr++};
        *slot_dst++ = seal::util::barrett_reduce_128(u128, crt_primes[crt_idx]);
      }
      // zero-padding the un-used slots.
      std::fill(slot_dst, slots.end(), 0);
      encoders_[crt_idx]->encode(slots, pt);
      evaluators_[crt_idx]->add_plain_inplace(cts[i], pt);
      truncate_for_decryption(cts[i], *evaluators_[crt_idx],
                              *contexts_[crt_idx]);
    }
    return Code::OK;
  };

  return LaunchWorks(tpool, n_ct, mask_prg);
}

Code HomBNSS::addMask(std::vector<seal::Ciphertext> &cts,
                      Tensor<uint64_t> &mask, const Meta &meta,
                      ThreadPool &tpool) const {
  if (IsTwoPower(target_base_mod_)) {
    return addMaskRing(cts, mask, meta, tpool);
  } else {
    return addMaskPrimeField(cts, mask, meta, tpool);
  }
}

Code HomBNSS::decryptToVector(const std::vector<seal::Ciphertext> &in_vec,
                              const Meta &meta, Tensor<uint64_t> &out_vec,
                              size_t nthreads) const {
  ENSURE_OR_RETURN(meta.target_base_mod == target_base_mod_,
                   Code::ERR_OUT_BOUND);
  ENSURE_OR_RETURN(
      !contexts_.empty() && !sks_.empty() && !encoders_.empty() && crt_context_,
      Code::ERR_CONFIG);
  if (meta.vec_shape.length() == 0) {
    LOG(FATAL) << "empty vec_shape";
  }

  TensorShape split_shape = getSplit(meta);
  const size_t nCRT = split_shape.dim_size(0);
  if (nCRT != contexts_.size()) {
    LOG(FATAL) << "BUG nCRT!";
  }

  const size_t sub_vec_len = split_shape.dim_size(1);
  const size_t n_sub_vecs =
      CeilDiv<size_t>(meta.vec_shape.length(), sub_vec_len);
  const size_t n_ct = nCRT * n_sub_vecs;
  ENSURE_OR_RETURN(in_vec.size() == n_ct, Code::ERR_INVALID_ARG);

  std::vector<std::shared_ptr<seal::Decryptor>> decryptors(nCRT);
  for (size_t crt = 0; crt < nCRT; ++crt) {
    decryptors[crt] =
        std::make_shared<seal::Decryptor>(*contexts_.at(crt), *sks_.at(crt));
  }

  std::vector<uint64_t> tmp(meta.vec_shape.length() * nCRT);
  auto decrypt_prg = [&](long wid, size_t start, size_t end) {
    seal::Plaintext pt;
    std::vector<uint64_t> slots;
    for (size_t cid = start; cid < end; ++cid) {
      size_t crt_idx = cid / n_sub_vecs;
      auto vec_pos_bgn = (cid % n_sub_vecs) * sub_vec_len;
      auto vec_pos_end =
          std::min<size_t>(vec_pos_bgn + sub_vec_len, meta.vec_shape.length());
      decryptors[crt_idx]->decrypt(in_vec.at(cid), pt);
      encoders_[crt_idx]->decode(pt, slots);
      std::copy_n(
          slots.cbegin(), vec_pos_end - vec_pos_bgn,
          tmp.begin() + vec_pos_bgn + crt_idx * meta.vec_shape.length());
    }
    seal::util::seal_memzero(slots.data(), slots.size() * sizeof(uint64_t));
    seal::util::seal_memzero(pt.data(), pt.coeff_count() * sizeof(uint64_t));
    return Code::OK;
  };
  ThreadPool tpool(nthreads);
  (void)LaunchWorks(tpool, n_ct, decrypt_prg);

  auto kcontext = crt_context_->key_context_data();
  ENSURE_OR_RETURN(kcontext != nullptr, Code::ERR_INTERNAL);
  auto rns_tool = kcontext->rns_tool();
  ENSURE_OR_RETURN(rns_tool != nullptr, Code::ERR_INTERNAL);

  out_vec.Reshape(meta.vec_shape);
  if (nCRT > 1) {
    auto tl = seal::MemoryManager::GetPool(seal::mm_force_thread_local);
    rns_tool->base_q()->compose_array(tmp.data(), meta.vec_shape.length(), tl);
    const uint64_t mod_mask = (1UL << n_base_mod_bits_) - 1;
    for (long i = 0, j = 0; i < meta.vec_shape.length(); ++i, j += nCRT) {
      out_vec(i) = tmp[j] & mod_mask;
    }
  } else {
    std::copy_n(tmp.cbegin(), out_vec.length(), out_vec.data());
  }

  seal::util::seal_memzero(tmp.data(), tmp.size() * sizeof(uint64_t));

  return Code::OK;
}

uint64_t HomBNSS::plain_modulus() const {
  if (direct_context_) {
    return direct_context_->first_context_data()
        ->parms()
        .plain_modulus()
        .value();
  }
  return 0;
}

Code HomBNSS::initPtx(seal::Plaintext &pt, seal::parms_id_type pid) const {
  ENSURE_OR_RETURN(direct_context_, Code::ERR_CONFIG);

  if (scheme() != seal::scheme_type::ckks) {
    // BFV or BGV
    pt.parms_id() = seal::parms_id_zero;  // foo SEAL when using BFV
    pt.resize(poly_degree());
    ENSURE_OR_RETURN(pt.data() != nullptr, Code::ERR_SEAL_MEMORY);
    return Code::OK;
  }

  if (pid == seal::parms_id_zero) {
    pid = direct_context_->first_parms_id();
  }

  auto cntxt_data = direct_context_->get_context_data(pid);
  ENSURE_OR_RETURN(cntxt_data != nullptr, Code::ERR_INVALID_ARG);
  const size_t num_moduli = cntxt_data->parms().coeff_modulus().size();
  const size_t num_elt = seal::util::mul_safe(num_moduli, poly_degree());
  pt.parms_id() = seal::parms_id_zero;  // foo SEAL when using BFV
  pt.resize(num_elt);
  pt.parms_id() = pid;
  ENSURE_OR_RETURN(pt.data() != nullptr, Code::ERR_SEAL_MEMORY);
  return Code::OK;
}

Code HomBNSS::vec2PolyBFV(const uint64_t *vec, size_t len, seal::Plaintext &pt,
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

Code HomBNSS::encodeTensor(const Tensor<uint64_t> &in_tensor, const Meta &meta,
                           std::vector<seal::Plaintext> &out,
                           size_t nthreads) const {
  ENSURE_OR_RETURN(direct_context_, Code::ERR_CONFIG);
  ENSURE_OR_RETURN(in_tensor.shape().IsSameSize(meta.ishape), Code::ERR_CONFIG);
  TensorShape split_shape = getSplitBN(meta.ishape, poly_degree());

  const int dC = CeilDiv(meta.ishape.channels(), split_shape.channels());
  const int dH = CeilDiv(meta.ishape.height(), split_shape.height());
  const int dW = CeilDiv(meta.ishape.width(), split_shape.width());
  const size_t n_pt = dC * dH * dW;
  out.resize(n_pt);

  auto encode_prg = [&](long wid, size_t start, size_t end) {
    std::vector<uint64_t> tmp(poly_degree());
    std::array<size_t, 3> indices{0};

    for (size_t cid = start; cid < end; ++cid) {
      indices[0] = cid / (dH * dW);
      indices[1] = (cid / dW) % dH;
      indices[2] = cid % dW;

      std::array<int, 3> offsets{0};
      for (int d = 0; d < 3; ++d) {
        offsets[d] = static_cast<int>(indices[d] * split_shape.dim_size(d));
      }

      SlicedPaddedTensor<Tensor<uint64_t>> tensor_slice(&in_tensor, offsets,
                                                        split_shape);
      auto tmp_ptr = tmp.begin();
      for (int c = 0; c < split_shape.channels(); ++c) {
        for (int h = 0; h < split_shape.height(); ++h) {
          for (int w = 0; w < split_shape.width(); ++w) {
            *tmp_ptr++ = tensor_slice(c, h, w);
          }
        }
      }
      std::fill(tmp_ptr, tmp.end(), 0);
      auto code = vec2PolyBFV(tmp.data(), tmp.size(), out.at(cid), false);
      if (code != Code::OK) {
        LOG(FATAL) << "vec2PolyBFV: " << CodeMessage(code) << std::endl;
      }
    }
    seal::util::seal_memzero(tmp.data(), tmp.size() * sizeof(uint64_t));
    return Code::OK;
  };

  ThreadPool tpool(nthreads);
  return LaunchWorks(tpool, n_pt, encode_prg);
}

Code HomBNSS::encryptTensor(
    const Tensor<uint64_t> &in_tensor, const Meta &meta,
    std::vector<seal::Serializable<seal::Ciphertext>> &out,
    size_t nthreads) const {
  ENSURE_OR_RETURN(direct_context_ && direct_encryptor_, Code::ERR_CONFIG);
  ENSURE_OR_RETURN(in_tensor.shape().IsSameSize(meta.ishape), Code::ERR_CONFIG);
  TensorShape split_shape = getSplitBN(meta.ishape, poly_degree());

  const int dC = CeilDiv(meta.ishape.channels(), split_shape.channels());
  const int dH = CeilDiv(meta.ishape.height(), split_shape.height());
  const int dW = CeilDiv(meta.ishape.width(), split_shape.width());
  const size_t n_pt = dC * dH * dW;

  out.resize(n_pt, direct_encryptor_->encrypt_zero_symmetric());

  auto encrypt_prg = [&](long wid, size_t start, size_t end) {
    seal::Plaintext pt;
    std::array<size_t, 3> indices{0};
    std::vector<uint64_t> tmp(poly_degree());
    for (size_t cid = start; cid < end; ++cid) {
      indices[0] = cid / (dH * dW);
      indices[1] = (cid / dW) % dH;
      indices[2] = cid % dW;

      std::array<int, 3> offsets{0};
      for (int d = 0; d < 3; ++d) {
        offsets[d] = static_cast<int>(indices[d] * split_shape.dim_size(d));
      }

      SlicedPaddedTensor<Tensor<uint64_t>> tensor_slice(&in_tensor, offsets,
                                                        split_shape);
      auto tmp_ptr = tmp.begin();
      for (int c = 0; c < split_shape.channels(); ++c) {
        for (int h = 0; h < split_shape.height(); ++h) {
          for (int w = 0; w < split_shape.width(); ++w) {
            *tmp_ptr++ = tensor_slice(c, h, w);
          }
        }
      }
      std::fill(tmp_ptr, tmp.end(), 0);
      auto code = vec2PolyBFV(tmp.data(), tmp.size(), pt, false);
      if (code != Code::OK) {
        LOG(FATAL) << "vec2PolyBFV: " << CodeMessage(code) << std::endl;
      }
      out.at(cid) = direct_encryptor_->encrypt_symmetric(pt);
    }

    seal::util::seal_memzero(pt.data(), pt.coeff_count() * sizeof(uint64_t));
    seal::util::seal_memzero(tmp.data(), tmp.size() * sizeof(uint64_t));
    return Code::OK;
  };

  ThreadPool tpool(nthreads);
  return LaunchWorks(tpool, n_pt, encrypt_prg);
}

Code HomBNSS::bn_direct(const std::vector<seal::Ciphertext> &tensor_share0,
                        const std::vector<seal::Plaintext> &tensor_share1,
                        const Tensor<uint64_t> &scales, const Meta &meta,
                        std::vector<seal::Ciphertext> &out_share0,
                        Tensor<uint64_t> &out_share1, size_t nthreads) const {
  using namespace seal::util;
  ENSURE_OR_RETURN(direct_context_ && direct_evaluator_, Code::ERR_CONFIG);
  // multiply each channel `c` by scales[c].
  ENSURE_OR_RETURN(
      scales.dims() == 1 && scales.length() == meta.ishape.channels(),
      Code::ERR_CONFIG);
  TensorShape split_shape = getSplitBN(meta.ishape, poly_degree());

  const int dC = CeilDiv(meta.ishape.channels(), split_shape.channels());
  const int dH = CeilDiv(meta.ishape.height(), split_shape.height());
  const int dW = CeilDiv(meta.ishape.width(), split_shape.width());
  const size_t n_ct = dC * dH * dW;
  ENSURE_OR_RETURN(tensor_share0.size() == n_ct, Code::ERR_DIM_MISMATCH);

  out_share0.resize(n_ct);
  auto add_prg = [&](long wid, size_t start, size_t end) {
    if (meta.is_shared_input) {
      for (size_t i = start; i < end; ++i) {
        direct_evaluator_->add_plain(tensor_share0[i], tensor_share1[i],
                                     out_share0[i]);
      }
    } else {
      for (size_t i = start; i < end; ++i) {
        out_share0[i] = tensor_share0[i];
      }
    }
    return Code::OK;
  };

  seal::Modulus mod_plain(plain_modulus());
  auto mul_prog = [&](long wid, size_t start, size_t end) {
    for (size_t c = start; c < end; ++c) {
      int64_t s = static_cast<int64_t>(scales(c));
      uint64_t v =
          barrett_reduce_64(static_cast<uint64_t>(std::abs(s)), mod_plain);
      bool sign = std::signbit(s);

      size_t cid_start = c * dH * dW;
      size_t cid_end = std::min(n_ct, cid_start + dH * dW);
      for (size_t cid = cid_start; cid < cid_end; ++cid) {
        auto cntxt =
            direct_context_->get_context_data(out_share0[cid].parms_id());
        if (!cntxt) {
          LOG(FATAL) << "bn_direct invalid ciphertext";
        }

        const auto &modulus = cntxt->parms().coeff_modulus();
        for (size_t k = 0; k < out_share0[cid].size(); ++k) {
          auto rns_ptr = out_share0[cid].data(k);
          for (size_t l = 0; l < modulus.size(); ++l) {
            uint64_t scalar = barrett_reduce_64(v, modulus[l]);
            if (sign) {
              scalar = negate_uint_mod(scalar, modulus[l]);
            }
            multiply_poly_scalar_coeffmod(rns_ptr, poly_degree(), scalar,
                                          modulus[l], rns_ptr);
            rns_ptr += poly_degree();
          }
        }
      }
    }
    return Code::OK;
  };

  std::vector<seal::Plaintext> rnd;
  auto mask_prog = [&](long wid, size_t start, size_t end) {
    RLWECt zero;
    for (size_t cid = start; cid < end; ++cid) {
      direct_evaluator_->mod_switch_to_inplace(
          out_share0[cid], direct_context_->last_parms_id());

      direct_evaluator_->sub_plain_inplace(out_share0[cid], rnd[cid]);
      direct_pk_encryptor_->encrypt_zero(out_share0[cid].parms_id(), zero);
      direct_evaluator_->add_inplace(out_share0[cid], zero);

      std::array<size_t, 3> indices;
      indices[0] = cid / (dH * dW);
      indices[1] = (cid / dW) % dH;
      indices[2] = cid % dW;
      std::array<int, 3> offsets{0};
      for (int d = 0; d < 3; ++d) {
        offsets[d] = static_cast<int>(indices[d] * split_shape.dim_size(d));
      }
      std::vector<size_t> used_coeff_indices;
      size_t used_index = 0;
      for (int c = 0; c < split_shape.channels(); ++c) {
        for (int h = 0; h < split_shape.height(); ++h) {
          for (int w = 0; w < split_shape.width(); ++w) {
            if (offsets[0] + c >= meta.ishape.channels() ||
                offsets[1] + h >= meta.ishape.height() ||
                offsets[2] + w >= meta.ishape.width()) {
              continue;
            }
            used_coeff_indices.push_back(used_index++);
          }
        }
      }
      remove_unused_coeffs(out_share0[cid], *direct_evaluator_,
                           used_coeff_indices);

      truncate_for_decryption(out_share0[cid], *direct_evaluator_,
                              *direct_context_);
      seal::util::seal_memzero(rnd[cid].data(),
                               rnd[cid].coeff_count() * sizeof(uint64_t));
    }
    return Code::OK;
  };

  Code code;
  ThreadPool tpool(nthreads);
  // Step 1: add over mod 2^k to reconstruct the encrypted shares
  code = LaunchWorks(tpool, n_ct, add_prg);
  if (code != Code::OK) {
    LOG(WARNING) << CodeMessage(code);
    return Code::ERR_INTERNAL;
  }

  // Step 2: multiply with the weight
  code = LaunchWorks(tpool, scales.length(), mul_prog);
  if (code != Code::OK) {
    LOG(WARNING) << CodeMessage(code);
    return Code::ERR_INTERNAL;
  }

  // Step 3: re-sharing
  out_share1.Reshape(meta.ishape);

  auto &parms =
      direct_context_->get_context_data(out_share0[0].parms_id())->parms();
  auto prng = parms.random_generator()->create();
  const size_t nbytes = mul_safe<size_t>(meta.ishape.num_elements(), sizeof(uint64_t));
  prng->generate(nbytes, reinterpret_cast<std::byte *>(out_share1.data()));
  const uint64_t mod_mask = target_base_mod_ - 1;
  if (IsTwoPower(target_base_mod_)) {
    std::transform(
        out_share1.data(), out_share1.data() + out_share1.NumElements(),
        out_share1.data(), [mod_mask](uint64_t u) { return u & mod_mask; });
  } else {
    modulo_poly_coeffs(out_share1.data(), out_share1.NumElements(), mod_plain,
                       out_share1.data());
  }

  code = encodeTensor(out_share1, meta, rnd, nthreads);
  if (code != Code::OK) {
    LOG(WARNING) << CodeMessage(code);
    return Code::ERR_INTERNAL;
  }

  code = LaunchWorks(tpool, rnd.size(), mask_prog);
  if (code != Code::OK) {
    LOG(WARNING) << CodeMessage(code);
    return Code::ERR_INTERNAL;
  }

  return Code::OK;
}

Code HomBNSS::decryptToTensor(const std::vector<seal::Ciphertext> &cts,
                              const Meta &meta, Tensor<uint64_t> &out_tensor,
                              size_t nthreads) const {
  ENSURE_OR_RETURN(direct_context_ && direct_sk_, Code::ERR_CONFIG);
  TensorShape split_shape = getSplitBN(meta.ishape, poly_degree());

  const int dC = CeilDiv(meta.ishape.channels(), split_shape.channels());
  const int dH = CeilDiv(meta.ishape.height(), split_shape.height());
  const int dW = CeilDiv(meta.ishape.width(), split_shape.width());
  const size_t n_ct = dC * dH * dW;
  ENSURE_OR_RETURN(cts.size() == n_ct, Code::ERR_DIM_MISMATCH);

  seal::Decryptor decryptor(*direct_context_, *direct_sk_);
  out_tensor.Reshape(meta.ishape);

  auto decrypt_prg = [&](long wid, size_t start, size_t end) {
    seal::Plaintext pt;
    std::array<size_t, 3> indices;
    for (size_t cid = start; cid < end; ++cid) {
      indices[0] = cid / (dH * dW);
      indices[1] = (cid / dW) % dH;
      indices[2] = cid % dW;

      std::array<int, 3> offsets{0};
      for (int d = 0; d < 3; ++d) {
        offsets[d] = static_cast<int>(indices[d] * split_shape.dim_size(d));
      }

      decryptor.decrypt(cts.at(cid), pt);
      auto pt_ptr = pt.data();
      for (int c = 0; c < split_shape.channels(); ++c) {
        for (int h = 0; h < split_shape.height(); ++h) {
          for (int w = 0; w < split_shape.width(); ++w) {
            if (offsets[0] + c >= meta.ishape.channels() ||
                offsets[1] + h >= meta.ishape.height() ||
                offsets[2] + w >= meta.ishape.width()) {
              continue;
            }
            out_tensor(offsets[0] + c, offsets[1] + h, offsets[2] + w) =
                *pt_ptr++;
          }
        }
      }
    }
    seal::util::seal_memzero(pt.data(), pt.coeff_count() * sizeof(uint64_t));
    return Code::OK;
  };

  ThreadPool tpool(nthreads);
  return LaunchWorks(tpool, n_ct, decrypt_prg);
}

}  // namespace gemini
