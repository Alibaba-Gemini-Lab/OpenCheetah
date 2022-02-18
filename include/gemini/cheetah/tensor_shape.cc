//  Authors: Wen-jie Lu on 2021/9/15.
#include "gemini/cheetah/tensor_shape.h"

#include <iostream>
namespace gemini {
TensorShape::TensorShape() : num_elements_(0) {}

TensorShape::TensorShape(std::initializer_list<int64_t> dim_sizes) {
  int dsize = static_cast<int>(dim_sizes.size());
  if (dsize > 0) {
    dims_.resize(dsize, 0);
    int d = 0;
    for (int64_t dim : dim_sizes) {
      Update(d++, dim);
    }
  }
}

void TensorShape::Update(int d, int64_t new_dim) {
  // dims_.resize(dim_size);
  // num_elements_ = -1;
  if (d < 0 || d >= dims() || new_dim < 0) {
    std::cout << "dims " << dims() << " but updatting dimension " << d << " to " << new_dim << "\n";
    throw std::invalid_argument("TensorShape: Update invalid arguments");
  }

  dims_[d] = new_dim;
  int64_t num_elements = 1;
  for (int64_t d : dims_) {
    num_elements *= d;
    if (num_elements < 0) {
      throw std::invalid_argument("TensorShape: Update product overflow");
    }
  }

  if (num_elements >= 0) num_elements_ = num_elements;
}

bool TensorShape::IsValid() const { return num_elements_ > 0; }

bool TensorShape::IsSameSize(const TensorShape& b) const {
  if (!IsValid() || !b.IsValid()) {
    return false;
  }
  if (b.dims() != dims()) return false;
  for (int d = 0; d < dims(); d++) {
    if (dim_size(d) != b.dim_size(d)) return false;
  }
  return true;
}

int64_t TensorShape::dim_size(int d) const {
  if (d < 0 || d >= dims()) {
    return -1;
  }
  return dims_[d];
}

std::ostream& operator<<(std::ostream& ss, const TensorShape& shape) {
  ss << "[";
  for (int d = 0; d < shape.dims(); ++d) {
    ss << shape.dim_size(d);
    if (d + 1 < shape.dims()) {
      ss << ", ";
    }
  }
  ss << "]";
  return ss;
}

}  // namespace gemini
