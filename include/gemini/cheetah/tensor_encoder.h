//  Authors: Wen-jie Lu on 2021/9/15.
#ifndef GEMINI_HE_LINEAR_TENSOR_ENCODER_H
#define GEMINI_HE_LINEAR_TENSOR_ENCODER_H
#include <memory>

#include "gemini/core/types.h"
#include "gemini/cheetah/shape_inference.h"
#include "gemini/cheetah/tensor.h"

namespace gemini {
class TensorShape;
template <class>
class Conv2DSliceHelper;

class TensorEncoder {
 public:
  enum class Role {
    encryptor,
    encoder,
    masking,
    evaluator,
    none,
  };

  explicit TensorEncoder(const RunTime &rt);

  ~TensorEncoder() {}

  Code EncodeImageShare(Role role, const U64Tensor &img_tensor,
                        const TensorShape &filter_shape, const Padding &padding,
                        size_t stride, bool is_ntt,
                        std::vector<RLWEPt> &out) const;

  Code EncodeFilter(const U64Tensor &filter, const TensorShape &image_shape,
                    const Padding &padding, size_t stride, bool is_ntt,
                    std::vector<RLWEPt> &out) const;

 private:
  template <class TensorType, class Indexer>
  Code Encode(TensorShape image_shape, TensorShape filter_shape,
              const TensorType &img_or_filter, const Indexer &indexer,
              U64 *out_poly, size_t out_max_size) const;

  Code A2HBFV(const U64 *vec, size_t len, RLWEPt &pt, const Role role, bool is_ntt) const;

  Code A2H(const U64 *vec, size_t len, RLWEPt &pt, const Role role, bool is_ntt) const;

  Code InitPtx(RLWEPt &pt, seal::parms_id_type pid = seal::parms_id_zero) const;

  inline size_t poly_degree() const {
    return rt_.first_context_data()->parms().poly_modulus_degree();
  }

  inline seal::Modulus plain_modulus() const {
    return rt_.first_context_data()->parms().plain_modulus();
  }

  inline size_t num_moduli() const {
    return rt_.first_context_data()->parms().coeff_modulus().size();
  }

  inline int logq() const {
    return rt_.first_context_data()->total_coeff_modulus_bit_count();
  }

  inline seal::scheme_type scheme() const {
    return rt_.first_context_data()->parms().scheme();
  }

  const RunTime &rt_;
};

class ConvCoeffIndexCalculator {
 public:
  explicit ConvCoeffIndexCalculator(size_t poly_degree, TensorShape ishape,
                                    TensorShape fshape, Padding padding,
                                    size_t stride);
  int slice_size(size_t d) const;

  Code Get(std::array<int, 2> coords, TensorShape &out_shape,
           std::vector<size_t> &indices) const;

  size_t NumAllIndices() const;

 private:
  Code Get(std::array<int, 2> coords, TensorShape &out_shape,
           std::vector<size_t> *indices) const;

  size_t poly_degree_, stride_;
  TensorShape ishape_, fshape_;
  TensorShape mock_shape_;
  std::shared_ptr<Conv2DSliceHelper<U64Tensor>> helper_;
};

}  // namespace gemini

#endif  // PEGASUS_HE_LINEAR_TENSOR_ENCODER_H
