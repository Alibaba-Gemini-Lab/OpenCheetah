//  Authors: Wen-jie Lu on 2021/9/15.
#ifndef GEMINI_CHEETAH_HOM_BN_SS_H_
#include <seal/secretkey.h>
#include <seal/serializable.h>

#include <optional>
#include <vector>

#include "gemini/cheetah/tensor.h"
#include "gemini/cheetah/tensor_shape.h"

// Forward
namespace seal {
class SEALContext;
class PublicKey;

class Plaintext;
class Ciphertext;
class Evaluator;
class BatchEncoder;
}  // namespace seal

namespace gemini {
class ThreadPool;
class HomBNSS {
 public:
#ifdef HOM_CONV2D_SS_MAX_THREADS
  static constexpr size_t kMaxThreads = HOM_CONV2D_SS_MAX_THREADS;
#else
  static constexpr size_t kMaxThreads = 16;
#endif
  static constexpr int64_t kStatBits = 40;  // statistical distance

  struct Meta {
    TensorShape ishape;
    TensorShape vec_shape;
    uint64_t target_base_mod;
    bool is_shared_input;
  };

  explicit HomBNSS() = default;

  ~HomBNSS() = default;

  Code setUp(uint64_t target_base_mod,
             const std::vector<seal::SEALContext> &contexts,
             std::vector<std::optional<seal::SecretKey>> sks);

  Code setUp(uint64_t target_base_mod, const seal::SEALContext &context,
             std::optional<seal::SecretKey> sk);

  inline seal::scheme_type scheme() const { return scheme_; }

  inline size_t poly_degree() const { return poly_degree_; }

  uint64_t plain_modulus() const;

  Code encryptVector(const Tensor<uint64_t> &in_vec, const Meta &meta,
                     std::vector<seal::Serializable<seal::Ciphertext>> &out,
                     size_t nthreads = 1) const;

  Code encodeVector(const Tensor<uint64_t> &in_vec, const Meta &meta,
                    std::vector<seal::Plaintext> &out,
                    size_t nthreads = 1) const;

  Code encodeScales(const Tensor<uint64_t> &scales, const Meta &meta,
                    std::vector<seal::Plaintext> &out,
                    size_t nthreads = 1) const;

  Code bn(const std::vector<seal::Ciphertext> &vec_share0,
          const std::vector<seal::Plaintext> &vec_share1,
          const std::vector<seal::Plaintext> &scales, const Meta &meta,
          std::vector<seal::Ciphertext> &out_share0,
          Tensor<uint64_t> &out_share1, size_t nthreads = 1) const;

  Code decryptToVector(const std::vector<seal::Ciphertext> &in_vec,
                       const Meta &meta, Tensor<uint64_t> &out_vec,
                       size_t nthreads = 1) const;

  Code encryptTensor(const Tensor<uint64_t> &in_tensor, const Meta &meta,
                     std::vector<seal::Serializable<seal::Ciphertext>> &out,
                     size_t nthreads = 1) const;

  Code encodeTensor(const Tensor<uint64_t> &in_tensor, const Meta &meta,
                    std::vector<seal::Plaintext> &out,
                    size_t nthreads = 1) const;

  Code bn_direct(const std::vector<seal::Ciphertext> &tensor_share0,
                 const std::vector<seal::Plaintext> &tensor_share1,
                 const Tensor<uint64_t> &scales, const Meta &meta,
                 std::vector<seal::Ciphertext> &out_share0,
                 Tensor<uint64_t> &out_share1, size_t nthreads = 1) const;

  Code decryptToTensor(const std::vector<seal::Ciphertext> &in_vec,
                       const Meta &meta, Tensor<uint64_t> &out_tensor,
                       size_t nthreads = 1) const;

  template <class IO, class CtVecType>
  Code sendEncryptVector(IO *io, const CtVecType &ct, const Meta &meta) const {
    TensorShape split_shape = getSplit(meta);
    const size_t nCRT = split_shape.dim_size(0);
    const size_t sub_vec_len = split_shape.dim_size(1);
    const size_t n_sub_vecs =
        CeilDiv<size_t>(meta.vec_shape.length(), sub_vec_len);
    const size_t n_ct = nCRT * n_sub_vecs;
    if (ct.size() != n_ct) {
      LOG(WARNING) << "sendEncryptVector number of ct mismatch";
      return Code::ERR_INVALID_ARG;
    }
    uint32_t n = n_ct;
    io->send_data(&n, sizeof(uint32_t));

    for (size_t i = 0; i < nCRT; ++i) {
      for (size_t j = 0; j < n_sub_vecs; ++j) {
        size_t cid = i * n_sub_vecs + j;
        std::stringstream os;
        uint64_t ct_size;
        ct.at(cid).save(os);
        ct_size = os.tellp();
        std::string ct_ser = os.str();
        io->send_data(&ct_size, sizeof(uint64_t));
        io->send_data(ct_ser.c_str(), ct_ser.size());
      }
    }
    return Code::OK;
  }

  template <class IO>
  Code recvEncryptVector(IO *io, std::vector<seal::Ciphertext> &ct,
                         const Meta &meta) const {
    TensorShape split_shape = getSplit(meta);
    const size_t nCRT = split_shape.dim_size(0);
    const size_t sub_vec_len = split_shape.dim_size(1);
    const size_t n_sub_vecs =
        CeilDiv<size_t>(meta.vec_shape.length(), sub_vec_len);
    const size_t n_ct = nCRT * n_sub_vecs;

    uint32_t n;
    io->recv_data(&n, sizeof(uint32_t));
    if (n != n_ct) {
      LOG(WARNING) << "sendEncryptVector number of ct mismatch";
      return Code::ERR_INVALID_ARG;
    }

    ct.resize(n_ct);
    for (size_t i = 0; i < nCRT; ++i) {
      for (size_t j = 0; j < n_sub_vecs; ++j) {
        size_t cid = i * n_sub_vecs + j;

        std::stringstream is;
        uint64_t ct_size;
        io->recv_data(&ct_size, sizeof(uint64_t));
        char *c_enc_result = new char[ct_size];
        io->recv_data(c_enc_result, ct_size);
        is.write(c_enc_result, ct_size);
        ct.at(cid).unsafe_load(*contexts_[i], is);
        if (!seal::is_valid_for(ct[cid], *contexts_[i])) {
          LOG(WARNING) << "bn recvEncryptVector invalid ciphertext";
        }
        delete[] c_enc_result;
      }
    }

    return Code::OK;
  }

 protected:
  TensorShape getSplit(const Meta &meta) const;

  Code addMaskPrimeField(std::vector<seal::Ciphertext> &ct,
                         Tensor<uint64_t> &mask, const Meta &meta,
                         ThreadPool &) const;
  Code addMaskRing(std::vector<seal::Ciphertext> &ct, Tensor<uint64_t> &mask,
                   const Meta &meta, ThreadPool &) const;
  Code addMask(std::vector<seal::Ciphertext> &ct, Tensor<uint64_t> &mask,
               const Meta &meta, ThreadPool &) const;

  Code initPtx(seal::Plaintext &pt,
               seal::parms_id_type pid = seal::parms_id_zero) const;
  Code vec2PolyBFV(const uint64_t *vec, size_t len, seal::Plaintext &pt,
                   bool is_ntt) const;

 private:
  uint64_t target_base_mod_{0};  // mod 2^k or mod p
  unsigned n_base_mod_bits_{0};  // ceil(log2(target_base_mod))
  size_t poly_degree_{0};
  seal::scheme_type scheme_{seal::scheme_type::none};
  std::shared_ptr<seal::SEALContext> crt_context_;

  // Cheetah BN
  std::shared_ptr<seal::SEALContext> direct_context_;
  std::optional<seal::SecretKey> direct_sk_;
  std::shared_ptr<seal::Evaluator> direct_evaluator_;
  std::shared_ptr<seal::Encryptor> direct_encryptor_;

  // CrytoFlow2-like BN
  std::vector<std::shared_ptr<seal::SEALContext>> contexts_;
  std::vector<std::optional<seal::SecretKey>> sks_;
  std::vector<std::shared_ptr<seal::BatchEncoder>> encoders_;
  std::vector<std::shared_ptr<seal::Evaluator>> evaluators_;
  std::vector<std::shared_ptr<seal::Encryptor>> encryptors_;
};

}  // namespace gemini

#endif  // GEMINI_CHEETAH_HOM_BN_SS_H_
