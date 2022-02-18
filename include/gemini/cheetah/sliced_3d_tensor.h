//  Authors: Wen-jie Lu on 2021/9/11.
#pragma once
#include <array>
#include <cassert>

#include "gemini/core/common.h"
#include "gemini/core/logging.h"
#include "gemini/core/util/math.h"
#include "gemini/cheetah/tensor_shape.h"

namespace gemini {

template <class Base>
class Strided3DTensor {
 public:
  using ScalarType = typename Base::ScalarType;

  ScalarType operator()(int c, int h, int w) const {
    if (c < 0 || c >= shape_.dim_size(0)) {
      LOG(WARNING) << "Strided3DTensor index out-of-bound";
    }
    if (h < 0 || h >= shape_.dim_size(1)) {
      LOG(WARNING) << "Strided3DTensor index out-of-bound";
    }
    if (w < 0 || w >= shape_.dim_size(2)) {
      LOG(WARNING) << "Strided3DTensor index out-of-bound";
    }

    return base_(c * stride3d_[0], h * stride3d_[1], w * stride3d_[2]);
  }

  explicit Strided3DTensor(const Base& base, std::array<int, 3> strides3d)
      : base_(base), stride3d_(strides3d) {
    shape_ = base.shape();
    for (int d : {0, 1, 2}) {
      shape_.Update(d, CeilDiv<int64_t>(shape_.dim_size(d), strides3d[d]));
    }
  }

  TensorShape shape() const { return shape_; }
  inline auto dims() const { return shape_.dims(); }
  inline auto channels() const { return shape_.channels(); }
  inline auto height() const { return shape_.height(); }
  inline auto width() const { return shape_.width(); }

 private:
  const Base& base_;
  std::array<int, 3> stride3d_;
  TensorShape shape_;
};

template <class Base>
class SlicedPaddedTensor {
 public:
  using ScalarType = typename Base::ScalarType;

  ScalarType operator()(int c, int h, int w) const {
    assert(c >= 0 && c < mock_shape_.dim_size(0));
    assert(h >= 0 && h < mock_shape_.dim_size(1));
    assert(w >= 0 && w < mock_shape_.dim_size(2));

    if (c >= shape_.dim_size(0)) return zero_;

    if (h < hpads_[0] ||
        static_cast<int64_t>(h + hpads_[1]) >= shape_.dim_size(1))
      return zero_;
    if (w < wpads_[0] ||
        static_cast<int64_t>(w + wpads_[1]) >= shape_.dim_size(2))
      return zero_;

    return base_->operator()(c + offsets_[0], (h - hpads_[0]) + offsets_[1],
                             w - wpads_[0] + offsets_[2]);
  }

  SlicedPaddedTensor(const SlicedPaddedTensor& oth)
      : zero_(oth.zero_),
        base_(oth.base_),
        offsets_(oth.offsets_),
        hpads_(oth.hpads_),
        wpads_(oth.wpads_),
        shape_(oth.shape_),
        mock_shape_(oth.mock_shape_) {}

  inline const TensorShape& shape() const { return mock_shape_; }
  inline auto dims() const { return shape().dims(); }
  inline auto channels() const { return shape().channels(); }
  inline auto height() const { return shape().height(); }
  inline auto width() const { return shape().width(); }

  explicit SlicedPaddedTensor() {}

  SlicedPaddedTensor(const Base* base, std::array<int, 3> offsets,
                     std::array<int, 2> hpads, std::array<int, 2> wpads,
                     const TensorShape& shape)
      : base_(base),
        offsets_(offsets),
        hpads_(hpads),
        wpads_(wpads),
        shape_(shape),
        mock_shape_(shape) {
    if (!base_) {
      LOG(FATAL) << "SlicedPaddedTensor: nullptr base";
    }
  }

  SlicedPaddedTensor(const Base* base, std::array<int, 3> offsets,
                     const TensorShape& shape)
      : SlicedPaddedTensor(base, offsets, {0, 0}, {0, 0}, shape) {}

  void Mock(TensorShape shape) {
    assert(shape.IsValid());
    mock_shape_ = shape;
  }

 private:
  ScalarType zero_{0};
  const Base* base_{nullptr};
  std::array<int, 3> offsets_{0, 0};
  std::array<int, 2> hpads_{0, 0}, wpads_{0, 0};
  TensorShape shape_;
  TensorShape mock_shape_;
};

template <class Base>
class Conv2DSliceHelper {
 public:
  explicit Conv2DSliceHelper(const Base* img, const TensorShape& ishape,
                             const TensorShape& fshape,
                             std::array<int, 3> slice_strides,
                             std::array<int, 2> paddings)
      : base_(img),
        shape_(ishape),
        padded_shape_({shape_.dim_size(0), shape_.dim_size(1) + paddings[0],
                       shape_.dim_size(2) + paddings[1]}),
        slice_strides_(slice_strides),
        paddings_(paddings) {
    if (!shape_.IsValid() || shape_.dims() != 3) {
      LOG(FATAL) << "Conv2DSliceHelper: invalid shape";
    }
    if (std::any_of(slice_strides_.begin(), slice_strides_.end(),
                    [](int s) { return s < 1; })) {
      LOG(FATAL) << "Conv2DSliceHelper: invalid slice strides";
    }
    if (std::any_of(paddings_.begin(), paddings_.end(),
                    [](int s) { return s < 0; })) {
      LOG(FATAL) << "Conv2DSliceHelper: invalid paddings";
    }
    if (slice_strides_[1] < fshape.dim_size(1) ||
        slice_strides_[2] < fshape.dim_size(1)) {
      LOG(FATAL) << "Conv2DSliceHelper: invalid paddings";
    }

    overlap_strides_[0] = slice_strides_[0];
    slices_[0] = CeilDiv<int>(shape_.dim_size(0), overlap_strides_[0]);

    for (int d : {1, 2}) {
      overlap_strides_[d] = slice_strides_[d];
      /// We need to split this dimension with overlaps
      if (padded_shape_.dim_size(d) > slice_strides_[d]) {
        overlap_strides_[d] -= (fshape.dim_size(d) - 1);
        slices_[d] =
            1 + CeilDiv<int>(padded_shape_.dim_size(d) - slice_strides_[d],
                             overlap_strides_[d]);
      } else {
        slices_[d] = CeilDiv<int>(padded_shape_.dim_size(d), slice_strides_[d]);
      }

      half_pads_[d] = CeilDiv<int>(paddings_[d - 1], 2);
    }
    half_pads_[0] = 0;
  }

  Conv2DSliceHelper(const TensorShape& ishape, const TensorShape& fshape,
                    std::array<int, 3> slice_strides,
                    std::array<int, 2> paddings)
      : Conv2DSliceHelper(nullptr, ishape, fshape, slice_strides, paddings) {}

  int slice_size(int d) const {
    if (d < 0 || d >= 3) return -1;
    return slices_[d];
  }

  int slice_start_at(int d, int s) const {
    if (d < 0 || d >= 3) return -1;
    if (s < 0 || s >= slices_[d]) return -1;
    return s * overlap_strides_[d];
  }

  int num_slices() const { return slices_[0] * slices_[1] * slices_[2]; }

  Code slice(std::array<int, 3> indices, TensorShape& sliced_shape) const {
    ENSURE_OR_RETURN(indices.size() == 3, Code::ERR_DIM_MISMATCH);

    sliced_shape = TensorShape({0, 0, 0});
    for (int d = 0; d < 3; ++d) {
      ENSURE_OR_RETURN(indices[d] >= 0 && indices[d] < slices_[d],
                       Code::ERR_INVALID_ARG);
      int start = indices[d] * overlap_strides_[d];
      int end =
          std::min<int>(start + slice_strides_[d], padded_shape_.dim_size(d));
      sliced_shape.Update(d, end - start);
    }
    return Code::OK;
  }

  Code slice(std::array<int, 3> indices,
             SlicedPaddedTensor<Base>& sp_tensor) const {
    ENSURE_OR_RETURN(indices.size() == 3, Code::ERR_DIM_MISMATCH);
    ENSURE_OR_RETURN(base_ != nullptr, Code::ERR_NULL_POINTER);

    TensorShape sliced_shape({0, 0, 0});
    auto code = this->slice(indices, sliced_shape);
    if (Code::OK != code) {
      LOG(WARNING) << CodeMessage(code);
      return Code::ERR_INTERNAL;
    }

    std::array<int, 3> clipped_coords;
    std::array<int, 2> pads[3];
    /// find the coords for the clipped cube.
    for (int d = 0; d < 3; ++d) {
      int p_coord = indices[d] * overlap_strides_[d] - half_pads_[d];
      int p_size = sliced_shape.dim_size(d);

      clipped_coords[d] = std::max<int>(0, p_coord);
      int clipped_size = std::min<int>(shape_.dim_size(d), p_coord + p_size);
      clipped_size -= clipped_coords[d];
      if (clipped_size <= 0) {
        LOG(WARNING) << "clipped_size <= 0";
        return Code::ERR_INTERNAL;
      }
      pads[d][0] = clipped_coords[d] - p_coord;
      pads[d][1] = (p_coord + p_size) - (clipped_coords[d] + clipped_size);
    }

    /// Now we do no pad the dimension(0)
    sp_tensor = SlicedPaddedTensor<Base>(base_, clipped_coords, pads[1],
                                         pads[2], sliced_shape);
    return Code::OK;
  }

 private:
  const Base* base_{nullptr};
  TensorShape shape_, padded_shape_;
  std::array<int, 3> slice_strides_;
  std::array<int, 3> overlap_strides_;
  std::array<int, 2> paddings_;
  std::array<int, 3> half_pads_;
  std::array<int, 3> slices_;
};

}  // namespace gemini
