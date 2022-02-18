//  Authors: Wen-jie Lu on 2021/9/11.
#ifndef GEMINI_HE_LINEAR_TENSOR_H
#define GEMINI_HE_LINEAR_TENSOR_H
#include <iostream>
#include <type_traits>

#include "gemini/core/common.h"
#include "gemini/core/logging.h"
#include "gemini/core/types.h"
#include "gemini/core/util/math.h"
#include "gemini/cheetah/shape_inference.h"
#include "gemini/cheetah/tensor_shape.h"
#include "gemini/cheetah/tensor_types.h"

namespace gemini {

template <typename TT = uint64_t>
class Tensor {
 public:
  using ScalarType = TT;
  using SignedScalarType =
      typename std::conditional<std::is_integral<ScalarType>::value, int64_t,
                                double>::type;

  static_assert(std::is_same<double, ScalarType>::value ||
                    std::is_same<uint64_t, ScalarType>::value,
                "require uint64_t or double");

  explicit Tensor(TensorShape shape) { Reshape(shape); }

  explicit Tensor() : shape_(TensorShape({0, 0, 0})) {}

  Tensor(const Tensor& oth)
      : shape_(oth.shape_),
        offsets_(oth.offsets_),
        raw_data_(oth.raw_data_),
        wrapped_raw_data_(oth.wrapped_raw_data_) {}

  Tensor(Tensor&& oth)
      : shape_(oth.shape_),
        offsets_(oth.offsets_),
        raw_data_(oth.raw_data_),
        wrapped_raw_data_(oth.wrapped_raw_data_) {
    oth.wrapped_raw_data_ = nullptr;
  }

  Tensor& operator=(const Tensor& oth) {
    shape_ = oth.shape_;
    raw_data_ = oth.raw_data_;
    wrapped_raw_data_ = oth.wrapped_raw_data_;
    offsets_ = oth.offsets_;
    return *this;
  }

  static Tensor<ScalarType> Wrap(ScalarType* raw_data, TensorShape shape) {
    Tensor<ScalarType> ret;
    ret.shape_ = shape;
    if (raw_data) ret.wrapped_raw_data_ = raw_data;
    const size_t ndims = shape.dims();
    ret.offsets_ = {0, 0};
    if (ndims == 3) {
      ret.offsets_[0] = shape.dim_size(1) * shape.dim_size(2);
      ret.offsets_[1] = shape.dim_size(2);
    } else if (ndims == 2) {
      ret.offsets_[0] = shape.dim_size(1);
    } else if (ndims != 1) {
      LOG(FATAL) << "invalid shape to wrap";
    }
    return ret;
  }

  void Reshape(TensorShape shape) {
    if (!shape.IsValid() || shape.dims() < 1) {
      LOG(FATAL) << "invalid tensor shape (or dims < 1) to Reshape";
    }
    if (wrapped_raw_data_) {
      LOG(FATAL) << "Reshape on wrapped tensor is not allowed";
    }

    shape_ = shape;
    raw_data_.resize(shape_.num_elements());
    wrapped_raw_data_ = nullptr;
    std::fill_n(raw_data_.begin(), raw_data_.size(), 0.);
    size_t ndims = shape.dims();
    offsets_ = {0, 0};
    if (ndims == 3) {
      offsets_[0] = shape.dim_size(1) * shape.dim_size(2);
      offsets_[1] = shape.dim_size(2);
    } else if (ndims == 2) {
      offsets_[0] = shape.dim_size(1);
    } else if (ndims != 1) {
      LOG(FATAL) << "invalid shape to Reshape";
    }
  }

  const TensorShape& shape() const { return shape_; }

  int dims() const { return shape().dims(); }

  bool IsValid() const { return shape().IsValid(); }

  bool IsZero() const {
    if (!IsValid()) return false;
    if (0 == NumElements()) return true;
    return !std::any_of(data(), data() + NumElements(),
                        [](ScalarType c) { return c != 0; });
  }

  inline int64_t dim_size(int d) const { return shape().dim_size(d); }
  inline int64_t channels() const { return shape().channels(); }
  inline int64_t height() const { return shape().height(); }
  inline int64_t width() const { return shape().width(); }
  inline int64_t rows() const { return shape().rows(); }
  inline int64_t cols() const { return shape().cols(); }
  inline int64_t length() const { return shape().length(); }

  int64_t NumElements() const { return shape().num_elements(); }

  bool IsSameSize(const Tensor& b) const {
    return shape().IsSameSize(b.shape());
  }

  SignedScalarType MaxDiff(const Tensor<ScalarType>& oth) const {
    if (!IsSameSize(oth)) {
      throw std::invalid_argument("Shape mismatch");
    }

    SignedScalarType max_err{0};
    SignedScalarType expect{0}, computed{0};

    int ncorrect = 0;
    int nerror = 0;

    auto ptr0 = data();
    auto ptr1 = oth.data();

    for (size_t i = 0; i < shape().num_elements(); ++i) {
      SignedScalarType _expect = static_cast<SignedScalarType>(*ptr0++);
      SignedScalarType _computed = static_cast<SignedScalarType>(*ptr1++);
      SignedScalarType e = std::abs(_expect - _computed);
      if (e < 1e-1) {
        ncorrect += 1;
      } else if (e > 10) {
        nerror += 1;
      }

      if (e > max_err) {
        max_err = e;
        expect = _expect;
        computed = _computed;
      }
    }
    return max_err;
  }

  const ScalarType* data() const {
    if (!wrapped_raw_data_ && raw_data_.empty()) {
      throw std::logic_error("Tensor::data on emtpy tensor");
    }
    return wrapped_raw_data_ ? wrapped_raw_data_ : raw_data_.data();
  }

  ScalarType* data() {
    if (!wrapped_raw_data_ && raw_data_.empty()) {
      throw std::logic_error("Tensor::data on emtpy tensor");
    }
    return wrapped_raw_data_ ? wrapped_raw_data_ : raw_data_.data();
  }

  typename TTypes<ScalarType, 1>::ConstTensor vector() const {
    if (dims() != 1) {
      LOG(FATAL) << "vector() demands 1D shape";
    }
    return typename TTypes<ScalarType, 1>::ConstTensor(data(), shape_.length());
  }

  typename TTypes<ScalarType, 1>::Tensor vector() {
    if (dims() != 1) {
      LOG(FATAL) << "vector () demands 1D shape";
    }
    return typename TTypes<ScalarType, 1>::Tensor(data(), shape_.length());
  }

  typename TTypes<ScalarType, 2>::ConstTensor matrix() const {
    if (dims() != 2) {
      LOG(FATAL) << "matrix() demands 2D shape";
    }
    return typename TTypes<ScalarType, 2>::ConstTensor(data(), shape_.rows(), shape_.cols());
  }

  typename TTypes<ScalarType, 2>::ConstTensor matrix() {
    if (dims() != 2) {
      LOG(FATAL) << "matrix() demands 2D shape";
    }
    return typename TTypes<ScalarType, 2>::Tensor(data(), shape_.rows(), shape_.cols());
  }

  typename TTypes<ScalarType, 3>::Tensor tensor() {
    if (dims() != 3) {
      LOG(FATAL) << "tensor() support only for 3D Tensor";
    }
    return typename TTypes<ScalarType, 3>::Tensor(
        data(), shape_.dim_size(0), shape_.dim_size(1), shape_.dim_size(2));
  }

  typename TTypes<ScalarType, 3>::ConstTensor tensor() const {
    if (dims() != 3) {
      LOG(FATAL) << "tensor() support only for 3D Tensor";
    }
    return typename TTypes<ScalarType, 3>::ConstTensor(
        data(), shape_.dim_size(0), shape_.dim_size(1), shape_.dim_size(2));
  }

  inline ScalarType operator()(long i) const {
    if (dims() != 1) {
      LOG(FATAL) << "1D indexing on non-1D tensor";
    }
    if (i < 0) return (ScalarType)0;
    if (i >= dim_size(0)) return (ScalarType)(0);
    return data()[i];
  }

  inline ScalarType& operator()(long i) {
    if (dims() != 1) {
      LOG(FATAL) << "1D indexing on non-1D tensor";
    }
    if (i < 0 || i >= dim_size(0)) {
      LOG(FATAL) << "Tensor index out-of-bound";
    }
    return data()[i];
  }

  inline ScalarType operator()(long r, long c) const {
    if (dims() != 2) {
      LOG(FATAL) << "2D indexing on non-2D tensor";
    }
    if (r < 0 || c < 0) return (ScalarType)0;
    if (r >= dim_size(0) || c >= dim_size(1)) return (ScalarType)(0);
    return data()[r * offsets_[0] + c];
  }

  inline ScalarType& operator()(long r, long c) {
    if (dims() != 2) {
      LOG(FATAL) << "2D indexing on non-2D tensor";
    }
    if (r < 0 || c < 0) throw std::invalid_argument("negative index");
    if (r >= dim_size(0) || c >= dim_size(1)) {
      throw std::invalid_argument("Tensor index out-of-bound");
    }
    return data()[r * offsets_[0] + c];
  }

  inline ScalarType operator()(long c, long h, long w) const {
    if (dims() != 3) {
      LOG(FATAL) << "3D indexing on non-3D tensor";
    }
    if (h < 0 || w < 0) return (ScalarType)0;
    if (h >= dim_size(1) || w >= dim_size(2)) return (ScalarType)(0);

    return data()[c * offsets_[0] + h * offsets_[1] + w];
  }

  inline ScalarType& operator()(long c, long h, long w) {
    if (dims() != 3) {
      LOG(FATAL) << "3D indexing on non-3D tensor";
    }
    if (h < 0 || w < 0) throw std::invalid_argument("negative index");
    if (h >= dim_size(1) || w >= dim_size(2)) {
      throw std::invalid_argument("Tensor index out-of-bound");
    }
    return data()[c * offsets_[0] + h * offsets_[1] + w];
  }

  Code Conv2D(const Tensor<ScalarType>& filter, int stride, Padding padding,
              Tensor<ScalarType>& out, const seal::Modulus& mod) const {
    ENSURE_OR_RETURN(IsValid() && filter.IsValid(), Code::ERR_INVALID_ARG);
    ENSURE_OR_RETURN(dims() == 3 && filter.dims() == 3, Code::ERR_INVALID_ARG);
    ENSURE_OR_RETURN(stride > 0, Code::ERR_INVALID_ARG);
    ENSURE_OR_RETURN(dim_size(0) == filter.dim_size(0), Code::ERR_DIM_MISMATCH);
    ENSURE_OR_RETURN(dim_size(1) >= filter.dim_size(1), Code::ERR_DIM_MISMATCH);
    ENSURE_OR_RETURN(dim_size(2) >= filter.dim_size(2), Code::ERR_DIM_MISMATCH);

    TensorShape oshape;
    const TensorShape fshape = filter.shape();

    if (padding == Padding::SAME) {
      shape_inference::MakeSamePadShape(shape(), fshape, oshape);
    } else {
      oshape = shape();
    }

    const long padh = CeilDiv<long>(oshape.height() - height(), 2);
    const long padw = CeilDiv<long>(oshape.width() - width(), 2);

    oshape.Update(0, 1);
    oshape.Update(1, (oshape.height() - fshape.height() + stride) / stride);
    oshape.Update(2, (oshape.width() - fshape.width() + stride) / stride);

    out.Reshape(oshape);

    for (long ih = -padh, oh = 0; oh < oshape.height(); ih += stride, ++oh) {
      for (long iw = -padw, ow = 0; ow < oshape.width(); iw += stride, ++ow) {
        ScalarType sum[2]{0};
        for (long ic = 0; ic < fshape.channels(); ++ic) {
          for (long fh = 0; fh < fshape.height(); ++fh) {
            for (long fw = 0; fw < fshape.width(); ++fw) {
              ScalarType m = seal::util::multiply_uint_mod(
                  filter(ic, fh, fw), (*this)(ic, ih + fh, iw + fw), mod);
              seal::util::add_uint(sum, 2, &m, 1, 0, 2, sum);
            }
          }
        }
        out(0, oh, ow) = seal::util::barrett_reduce_128(sum, mod);
      }
    }

    return Code::OK;
  }

  Code Conv2D(const Tensor<ScalarType>& filter, int stride, Padding padding,
              Tensor<ScalarType>& out) const {
    ENSURE_OR_RETURN(IsValid() && filter.IsValid(), Code::ERR_INVALID_ARG);
    ENSURE_OR_RETURN(dims() == 3 && filter.dims() == 3, Code::ERR_INVALID_ARG);
    ENSURE_OR_RETURN(stride > 0, Code::ERR_INVALID_ARG);
    ENSURE_OR_RETURN(dim_size(0) == filter.dim_size(0), Code::ERR_DIM_MISMATCH);
    ENSURE_OR_RETURN(dim_size(1) >= filter.dim_size(1), Code::ERR_DIM_MISMATCH);
    ENSURE_OR_RETURN(dim_size(2) >= filter.dim_size(2), Code::ERR_DIM_MISMATCH);

    TensorShape oshape;
    const TensorShape fshape = filter.shape();

    if (padding == Padding::SAME) {
      shape_inference::MakeSamePadShape(shape(), fshape, oshape);
    } else {
      oshape = shape();
    }

    const long padh = CeilDiv<long>(oshape.height() - height(), 2);
    const long padw = CeilDiv<long>(oshape.width() - width(), 2);

    oshape.Update(0, 1);
    oshape.Update(1, (oshape.height() - fshape.height() + stride) / stride);
    oshape.Update(2, (oshape.width() - fshape.width() + stride) / stride);

    out.Reshape(oshape);

    for (long ih = -padh, oh = 0; oh < oshape.height(); ih += stride, ++oh) {
      for (long iw = -padw, ow = 0; ow < oshape.width(); iw += stride, ++ow) {
        ScalarType sum{0};
        for (long ic = 0; ic < fshape.channels(); ++ic) {
          for (long fh = 0; fh < fshape.height(); ++fh) {
            for (long fw = 0; fw < fshape.width(); ++fw) {
              sum += filter(ic, fh, fw) * (*this)(ic, ih + fh, iw + fw);
            }
          }
        }
        out(0, oh, ow) = sum;
      }
    }

    return Code::OK;
  }

  void Uniform() {
    if (std::is_same<ScalarType, uint64_t>::value) {
      std::random_device rdv;
      std::uniform_int_distribution<uint64_t> uniform(
          0, static_cast<uint64_t>(-1));

      std::generate_n(wrapped_raw_data_ ? wrapped_raw_data_ : raw_data_.data(),
                      shape_.num_elements(), [&]() { return uniform(rdv); });
    }
  }

  void Randomize(ScalarType range = 1) {
    SignedScalarType upper = std::abs(static_cast<SignedScalarType>(range));
    std::random_device rdv;

    if (std::is_same<ScalarType, double>::value) {
      SignedScalarType lower = -upper;
      std::uniform_real_distribution<double> uniform(lower, upper);
      std::generate_n(wrapped_raw_data_ ? wrapped_raw_data_ : raw_data_.data(),
                      shape_.num_elements(), [&]() { return uniform(rdv); });
    } else if (std::is_same<ScalarType, uint64_t>::value) {
      std::uniform_int_distribution<uint64_t> uniform(0, upper);
      std::generate_n(wrapped_raw_data_ ? wrapped_raw_data_ : raw_data_.data(),
                      shape_.num_elements(), [&]() { return uniform(rdv); });
    }
  }

 private:
  TensorShape shape_;
  std::array<size_t, 2> offsets_{0, 0};
  std::vector<ScalarType> raw_data_;
  ScalarType* wrapped_raw_data_{nullptr};
};

using U64Tensor = Tensor<U64>;
using F64Tensor = Tensor<F64>;
}  // namespace gemini
#endif  // PEGASUS_HE_LINEAR_TENSOR_H
