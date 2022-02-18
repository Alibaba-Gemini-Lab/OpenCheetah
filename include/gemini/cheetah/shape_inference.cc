//  Authors: Wen-jie Lu on 2021/9/14.
#include "gemini/cheetah/shape_inference.h"

#include "gemini/core/logging.h"
#include "gemini/core/util/math.h"

namespace gemini {
namespace shape_inference {

bool MakeSamePadShape(const TensorShape &tshape, const TensorShape &fshape,
                      TensorShape &padded_shape) {
  if (tshape.dims() != fshape.dims()) {
    return false;
  }
  if (!tshape.IsValid() || !fshape.IsValid()) {
    return false;
  }

  padded_shape = tshape;
  for (int d = 1; d < tshape.dims(); ++d) {
    padded_shape.Update(d, tshape.dim_size(d) + fshape.dim_size(d) - 1);
  }
  padded_shape.Update(0, fshape.dim_size(0));

  return true;
}

std::optional<TensorShape> Conv2D(const TensorShape &tshape,
                                  const TensorShape &fshape,
                                  const Padding padding, const size_t stride) {
  TensorShape oshape;
  if (padding == Padding::SAME) {
    if (!shape_inference::MakeSamePadShape(tshape, fshape, oshape)) {
      return std::nullopt;
    }
  } else {
    oshape = tshape;
  }

  int s = static_cast<int>(stride);
  int h = (oshape.height() - fshape.height() + s) / s;
  int w = (oshape.width() - fshape.width() + s) / s;

  if (h > 0 && w > 0) {
    oshape.Update(0, 1);
    oshape.Update(1, h);
    oshape.Update(2, w);
    return oshape;
  } else {
    return std::nullopt;
  }
}

bool Conv2D(const TensorShape &tensor_shape, const TensorShape &filter_shape,
            const size_t poly_degree, Padding padding, const size_t stride,
            TensorShape &strided_tshape, std::array<int, 2> &paddings,
            std::array<int, 3> &slice_strides) {
  if (tensor_shape.dims() != 3 || filter_shape.dims() != 3) {
    LOG(WARNING) << "Conv2D: dim_size != 3";
    return false;
  }
  if (tensor_shape.channels() != filter_shape.channels()) {
    LOG(WARNING) << "Conv2D: channels mismatch";
    return false;
  }
  if (!filter_shape.IsValid()) {
    LOG(WARNING) << "Conv2D: invalid filter size";
    return false;
  }
  const int Cw = poly_degree / (filter_shape.height() * filter_shape.width());
  if (Cw <= 0) {
    LOG(WARNING) << "Conv2D: filter size out-of-bound";
    return false;
  }

  TensorShape tshape;
  if (padding == Padding::SAME) {
    if (!MakeSamePadShape(tensor_shape, filter_shape, tshape)) {
      LOG(WARNING) << "Conv2D: failed to pad same shape";
      return false;
    }
  } else {
    tshape = tensor_shape;
  }

  paddings[0] = tshape.height() - tensor_shape.height();
  paddings[1] = tshape.width() - tensor_shape.width();

  for (int d = 0; d < filter_shape.dims(); ++d) {
    if (tshape.dim_size(d) < filter_shape.dim_size(d)) {
      LOG(WARNING)
          << "Conv2D: tensor_shape.dim_size(d) < filter_shape.dim_size(d)";
      return false;
    }
  }

  const int s = static_cast<int>(stride);
  strided_tshape = tshape;
  for (int d : {1, 2}) {
    /// Optimization for h = 1 with stride > 1.
    if (filter_shape.dim_size(d) == 1) {
      strided_tshape.Update(d, (tshape.dim_size(d) + s - 1) / s);
    }
  }

  const int H = strided_tshape.height();
  const int W = strided_tshape.width();
  size_t HW = static_cast<size_t>(H) * W;
  if (HW <= poly_degree) {
    // H * W <= N
    slice_strides[0] = std::min<int>(poly_degree / HW, tshape.dim_size(0));
    slice_strides[0] = std::min<int>(slice_strides[0], Cw);
    slice_strides[1] = H;
    slice_strides[2] = W;
  } else {
    // H * W > N
    double ratio = static_cast<double>(H) / W;
    slice_strides[0] = 1;

    slice_strides[1] = std::min(static_cast<int>(std::sqrt(poly_degree)), H);
    slice_strides[2] = slice_strides[1];
    slice_strides[1] =
        std::min(static_cast<int>(std::sqrt(poly_degree * ratio)), H);
    slice_strides[2] = std::min<int>(poly_degree / slice_strides[1], W);

    if (std::abs(slice_strides[2] - slice_strides[1]) < 2) {
      int min_s = std::min(slice_strides[1], slice_strides[2]);
      slice_strides[1] = min_s;
      slice_strides[2] = min_s;
    }
  }
  return true;
}

}  // namespace shape_inference
}  // namespace gemini
