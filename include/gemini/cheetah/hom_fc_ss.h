//  Authors: Wen-jie Lu on 2021/9/14.
#ifndef GEMINI_CHEETAH_HOM_FC_SS_H_
#define GEMINI_CHEETAH_HOM_FC_SS_H_
#include <seal/secretkey.h>
#include <seal/serializable.h>

#include <optional>
#include <vector>

#include "gemini/cheetah/tensor.h"
#include "gemini/cheetah/tensor_shape.h"
#include "gemini/core/util/ThreadPool.h"

// Forward
namespace seal {
class SEALContext;
class PublicKey;

class Plaintext;
class Ciphertext;
class Evaluator;
class UniformRandomGenerator;
}  // namespace seal

namespace gemini {

class HomFCSS {
 public:
#ifdef HOM_CONV2D_SS_MAX_THREADS
  static constexpr size_t kMaxThreads = HOM_CONV2D_SS_MAX_THREADS;
#else
  static constexpr size_t kMaxThreads = 16;
#endif
  struct Meta {
    TensorShape input_shape;
    TensorShape weight_shape;
    bool is_shared_input;
  };

  explicit HomFCSS() = default;

  ~HomFCSS() = default;

  Code setUp(const seal::SEALContext &context,
             std::optional<seal::SecretKey> sk);

  [[nodiscard]] seal::scheme_type scheme() const;

  [[nodiscard]] size_t poly_degree() const;

  uint64_t plain_modulus() const;

  Code encryptInputVector(
      const Tensor<uint64_t> &vector, const Meta &meta,
      std::vector<seal::Serializable<seal::Ciphertext>> &encrypted_share,
      size_t nthreads = 1) const;

  Code encodeInputVector(const Tensor<uint64_t> &vector, const Meta &meta,
                         std::vector<seal::Plaintext> &encoded_share,
                         size_t nthreads = 1) const;

  Code encodeWeightMatrix(
      const Tensor<uint64_t> &weight_matrix, const Meta &meta,
      std::vector<std::vector<seal::Plaintext>> &encoded_share,
      size_t nthreads = 1) const;

  Code matVecMul(const std::vector<std::vector<seal::Plaintext>> &matrix,
                 const std::vector<seal::Ciphertext> &vec_share0,
                 const std::vector<seal::Plaintext> &vec_share1,
                 const Meta &meta,
                 std::vector<seal::Ciphertext> &out_vec_share0,
                 Tensor<uint64_t> &out_vec_share1, size_t nthreads = 1) const;

  Code decryptToVector(const std::vector<seal::Ciphertext> &enc_vector,
                       const Meta &meta, Tensor<uint64_t> &out,
                       size_t nthreads = 1) const;

  Code idealFunctionality(const Tensor<uint64_t> &weight_matrix,
                          const Tensor<uint64_t> &vector, const Meta &meta,
                          Tensor<uint64_t> &out) const;

 protected:
  enum class Role {
    encryptor,
    encoder,
    masking,
    evaluator,
    none,
  };

  Code initPtx(seal::Plaintext &pt,
               seal::parms_id_type pid = seal::parms_id_zero) const;

  Code vec2PolyBFV(const uint64_t *vec, size_t len, seal::Plaintext &pt,
                   bool is_ntt) const;

  Code vec2Poly(const uint64_t *vec, size_t len, seal::Plaintext &pt,
                const Role role, bool is_ntt = false) const;

  Code sampleRandomMask(const std::vector<size_t> &targets,
                        uint64_t *coeffs_buff, size_t buff_size,
                        seal::Plaintext &mask, seal::parms_id_type pid,
                        bool is_ntt) const;

  Code addRandomMask(std::vector<seal::Ciphertext> &enc_tensor,
                     Tensor<uint64_t> &mask_tensor, const Meta &meta,
                     gemini::ThreadPool &tp) const;

  Code removeUnusedCoeffs(std::vector<seal::Ciphertext> &ct, const Meta &meta,
                          double *density = nullptr) const;

 private:
  std::shared_ptr<seal::SEALContext> context_;
  std::shared_ptr<seal::Evaluator> evaluator_{nullptr};
  std::shared_ptr<seal::Encryptor> encryptor_{nullptr};

  std::optional<seal::SecretKey> sk_{std::nullopt};
};
}  // namespace gemini

#endif  // GEMINI_CHEETAH_HOM_FC_SS_H_
