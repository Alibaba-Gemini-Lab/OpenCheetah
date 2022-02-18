// Author: Wen-jie Lu on 2021/9/14.
#ifndef SCI_CHEETAH_CHEETAH_API_H_
#define SCI_CHEETAH_CHEETAH_API_H_

#include "gemini/cheetah/hom_bn_ss.h"
#include "gemini/cheetah/hom_conv2d_ss.h"
#include "gemini/cheetah/hom_fc_ss.h"

namespace sci {
class NetIO;
}

namespace gemini {

// The set of the linear protocols in the Cheetah's paper.
class CheetahLinear {
 public:
  using ConvMeta = HomConv2DSS::Meta;
  using FCMeta = HomFCSS::Meta;
  using BNMeta = HomBNSS::Meta;

  CheetahLinear(int party, sci::NetIO *io, uint64_t base_mod, size_t nthreads = 1);

  ~CheetahLinear() = default;

  // HomConv
  void conv2d(const Tensor<uint64_t> &in_tensor,
              const std::vector<Tensor<uint64_t>> &filters,
              const ConvMeta &meta, Tensor<uint64_t> &out_tensor) const;

  // HomFC
  void fc(const Tensor<uint64_t> &input_matrix,
          const Tensor<uint64_t> &weight_matrix, const FCMeta &meta,
          Tensor<uint64_t> &out_matrix) const;

  // The CrypTFlow2's like BN protocol.
  // Need element-wise multiplication.
  void bn(const Tensor<uint64_t> &input_vector,
          const Tensor<uint64_t> &scale_vector, const BNMeta &meta,
          Tensor<uint64_t> &out_vector) const;

  // HomBN
  void bn_direct(const Tensor<uint64_t> &input_tensor,
                 const Tensor<uint64_t> &scale_vector, const BNMeta &meta,
                 Tensor<uint64_t> &out_tensor) const;

  uint64_t io_counter() const;

  int party() const { return party_; }

  bool verify(const Tensor<uint64_t> &int_tensor,
              const std::vector<Tensor<uint64_t>> &filters,
              const ConvMeta &meta, const Tensor<uint64_t> &computed_tensor,
              int nbit_precision) const;

  template <typename T>
  void safe_erase(T *mem, size_t cnt) const {
    if (mem && cnt > 0) {
      seal::util::seal_memzero(mem, sizeof(T) * cnt);
    }
  }

 protected:
  int64_t get_signed(uint64_t v) const;
  uint64_t reduce(uint64_t v) const;

 private:
  void setUpForBN();

  int party_{-1};
  sci::NetIO *io_{nullptr};
  size_t nthreads_{1};

  uint64_t base_mod_{0};
  uint64_t mod_mask_{0};
  uint64_t positive_upper_{0};
  // If base_mod_ is not power-of-two
  std::optional<seal::Modulus> barrett_reducer_{std::nullopt};

  std::shared_ptr<seal::SEALContext> context_;
  std::shared_ptr<seal::SecretKey> sk_;  // Bob only
  std::shared_ptr<seal::PublicKey> pk_;  // Alice only

  std::vector<std::shared_ptr<seal::SEALContext>> bn_contexts_;
  std::vector<std::shared_ptr<seal::SecretKey>> bn_sks_;  // Bob only
  std::vector<std::shared_ptr<seal::PublicKey>> bn_pks_;  // Alice only

  HomFCSS fc_impl_;
  HomConv2DSS conv2d_impl_;
  HomBNSS bn_impl_;
};

}  // namespace gemini

#endif 
