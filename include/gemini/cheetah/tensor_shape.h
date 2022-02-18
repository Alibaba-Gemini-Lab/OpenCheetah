//  Authors: Wen-jie Lu on 2021/9/15.
#pragma once
#include <array>
#include <cstdint>
#include <initializer_list>
#include <iosfwd>
#include <vector>

namespace gemini {
enum class Padding { VALID, SAME };

template <typename U>
class Tensor;
class TensorShape {
 public:
  explicit TensorShape();
  TensorShape(std::initializer_list<int64_t> dim_sizes);
  TensorShape(std::array<int, 2> const& s2d) : TensorShape({s2d[0], s2d[1]}) {}
  TensorShape(std::array<int, 3> const& s3d)
      : TensorShape({s3d[0], s3d[1], s3d[2]}) {}
  int64_t num_elements() const { return num_elements_; }

  /// Return the number of dimensions in the tensor.
  /// Can be -1 meaning unknown rank
  int dims() const { return static_cast<int>(dims_.size()); };
  /// Returns the number of elements in dimension `d`.
  int64_t dim_size(int d) const;

  inline int64_t channels() const {
    if (dims() != 3) {
      throw std::invalid_argument("no channels() for non 3D tensor");
    }
    return dim_size(0);
  }

  inline int64_t height() const {
    if (dims() != 3) {
      throw std::invalid_argument("no height() for non 3D tensor");
    }
    return dim_size(1);
  }

  inline int64_t width() const {
    if (dims() != 3) {
      throw std::invalid_argument("no width() for non 3D tensor");
    }
    return dim_size(2);
  }

  inline int64_t rows() const {
    if (dims() != 2) {
      throw std::invalid_argument("no rows() for non 2D tensor");
    }
    return dim_size(0);
  }

  inline int64_t cols() const {
    if (dims() != 2) {
      throw std::invalid_argument("no cols() for non 2D tensor");
    }
    return dim_size(1);
  }

  inline int64_t length() const {
    if (dims() != 1) {
      throw std::invalid_argument("no length() for non 1D tensor");
    }
    return dim_size(0);
  }

  bool IsValid() const;

  /// Returns true if `*this` and `b` have the same sizes.
  bool IsSameSize(const TensorShape& b) const;
  bool operator==(const TensorShape& b) const { return IsSameSize(b); }
  bool operator!=(const TensorShape& b) const { return !IsSameSize(b); }
  void Update(int d, int64_t new_dim);
  friend std::ostream& operator<<(std::ostream& os, const TensorShape& shape);

 private:
  void OnUpdateDims();
  int64_t num_elements_ = -1;
  std::vector<int64_t> dims_;
};

}  // namespace gemini
