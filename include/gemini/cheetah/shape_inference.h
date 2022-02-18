//  Authors: Wen-jie Lu on 2021/9/14.
#ifndef GEMINI_SHAPE_INFERENCE_H
#define GEMINI_SHAPE_INFERENCE_H
#include <array>
#include <optional>

#include "gemini/cheetah/tensor_shape.h"
namespace gemini {

namespace shape_inference {

bool MakeSamePadShape(const TensorShape &tensor_shape,
                      const TensorShape &filter_shape,
                      TensorShape &output_shape);

std::optional<TensorShape> Conv2D(const TensorShape &tensor_shape,
                                  const TensorShape &filter_shape,
                                  Padding padding, size_t stride);

/// Inference on how to split the input tensor using `poly_degree` ///
/// coefficients.
bool Conv2D(const TensorShape &tensor_shape, const TensorShape &filter_shape,
            const size_t poly_degree, Padding padding, size_t stride,
            TensorShape &strided_tshape, std::array<int, 2> &paddings,
            std::array<int, 3> &slice_strides);


struct PaddedTensor {
  TensorShape shape;
  TensorShape padded_shape;
};

}  // namespace shape_inference

}  // namespace gemini
#endif
