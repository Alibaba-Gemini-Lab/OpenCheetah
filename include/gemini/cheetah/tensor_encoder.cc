//  Authors: Wen-jie Lu on 2021/9/15.
#include "gemini/cheetah/tensor_encoder.h"

#include <seal/plaintext.h>
#include <seal/util/polyarithsmallmod.h>

#include "gemini/core/common.h"
#include "gemini/core/logging.h"
#include "gemini/core/types.h"
#include "gemini/cheetah/sliced_3d_tensor.h"
namespace gemini {

struct ConvIndexer {
 public:
  explicit ConvIndexer(long N, TensorShape ishape, TensorShape fshape) {
    if (!ishape.IsValid() || !fshape.IsValid() || ishape.dims() != 3 ||
        fshape.dims() != 3 || ishape.dim_size(0) != fshape.dim_size(0)) {
      LOG(FATAL) << "invalid shapes" << ishape << " " << fshape;
    }

    if (N < ishape.num_elements()) {
      LOG(FATAL) << "tensor shape out-of-bound";
    }
  }

  inline long PadPow2(long lower, long upper) const {
    assert(lower <= upper);
    // return lower <= 2^k <= upper
    if (upper >= (lower << 1)) {
      return 1L << static_cast<int>(std::floor(std::log2(upper)));
    } else {
      return upper;
    }
  }
};

struct ImageIndexer : public ConvIndexer {
 public:
  ImageIndexer(long poly_degree, TensorShape ishape, TensorShape fshape,
               Padding padding = Padding::VALID)
      : ConvIndexer(poly_degree, ishape, fshape) {
    shape_ = ishape;

    if (padding == Padding::SAME) {
      shape_inference::MakeSamePadShape(ishape, fshape, shape_);
    }

    offset_ = shape_.height() * shape_.width();
  }

  inline long operator()(long chl, long row, long col) const {
    if (chl < 0 || row < 0 || col < 0) {
      throw std::runtime_error("invalid negative index");
    }

    if (chl >= shape_.dim_size(0) || row >= shape_.dim_size(1) ||
        col >= shape_.dim_size(2)) {
      throw std::runtime_error("TensorIndexer index out-of-bound");
    }

    return chl * offset_ + row * shape_.dim_size(2) + col;
  }

 private:
  TensorShape shape_;
  long offset_;
};

struct FilterIndexer : public ConvIndexer {
 public:
  FilterIndexer(long poly_degree, TensorShape ishape, TensorShape fshape,
                Padding padding = Padding::VALID)
      : ConvIndexer(poly_degree, ishape, fshape), shape_(fshape) {
    if (padding == Padding::SAME) {
      shape_inference::MakeSamePadShape(ishape, fshape, ishape);
    }

    long h = fshape.dim_size(2);
    row_nskip_ = ishape.width();

    offset_ = ishape.height() * ishape.width();
    // O = HW*(C-1) + W*(h-1) + (h-1)
    begin_ =
        offset_ * (fshape.channels() - 1) + ishape.width() * (h - 1) + h - 1;
  }

  inline long operator()(long chl, long row, long col) const {
    if (chl < 0 || row < 0 || col < 0) {
      LOG(FATAL) << "Negative index";
    }

    if (chl >= shape_.dim_size(0) || row >= shape_.dim_size(1) ||
        col >= shape_.dim_size(2)) {
      LOG(FATAL) << "index out-of-bound";
    }

    // O - c*H*W - l*W - l'
    return begin_ - chl * offset_ - row * row_nskip_ - col;
  }

  inline long index_begin() const { return begin_; }

 private:
  TensorShape shape_;
  long row_nskip_, offset_, begin_;
};

TensorEncoder::TensorEncoder(const RunTime &rt) : rt_(rt) {}

template <class TT>
void print_tensor(TT const &tensor) {
  TensorShape shape = tensor.shape();
  for (size_t c = 0; c < shape.channels(); c++) {
    std::cout << "[";
    for (size_t h = 0; h < shape.height(); ++h) {
      for (size_t w = 0; w < shape.width(); ++w) {
        std::cout << tensor(c, h, w) << " ";
      }
    }
    std::cout << "]\n";
  }
}

Code TensorEncoder::EncodeFilter(const U64Tensor &filter,
                                 const TensorShape &img_shape,
                                 const Padding &padding, const size_t stride,
                                 const bool is_ntt_form,
                                 std::vector<RLWEPt> &out) const {
  TensorShape fshape = filter.shape();
  ENSURE_OR_RETURN(img_shape.dims() == 3 && fshape.dims() == 3, Code::ERR_DIM_MISMATCH);
  if (filter.width() * filter.height() > poly_degree()) {
     LOG(FATAL) << "filter.num_elements > poly_degree";
  }

  TensorShape strided_tshape;
  std::array<int, 2> paddings;
  std::array<int, 3> slice_strides;
  if (!shape_inference::Conv2D(img_shape, fshape, poly_degree(), padding,
                               stride, strided_tshape, paddings,
                               slice_strides)) {
    LOG(WARNING) << "EncodeFilter: shape_inference failed";
    return Code::ERR_INVALID_ARG;
  }

  TensorShape sliced_ishape({0, 0, 0});
  for (int d : {1, 2}) {
    sliced_ishape.Update(d, slice_strides[d]);
  }

  slice_strides[1] = fshape.dim_size(1);
  slice_strides[2] = fshape.dim_size(2);
  // Juhou: use same shapes for the slicing.
  Conv2DSliceHelper<U64Tensor> helper(&filter, fshape, fshape, slice_strides,
                                      /*padding*/ {0, 0});
  out.resize(helper.num_slices());

  auto pool = seal::MemoryManager::GetPool(seal::mm_force_thread_local);
  auto tmp_buf = seal::util::allocate_poly(poly_degree(), 1, pool);

  for (int c = 0; c < helper.num_slices(); ++c) {
    SlicedPaddedTensor<U64Tensor> sliced_filter;
    CHECK_ERR(helper.slice({c, 0, 0}, sliced_filter), "slice");

    TensorShape sliced_fshape = sliced_filter.shape();
    sliced_ishape.Update(0, slice_strides[0]);
    sliced_fshape.Update(0, slice_strides[0]);
    sliced_filter.Mock(sliced_fshape);

    FilterIndexer indexer(poly_degree(), sliced_ishape, sliced_fshape);
    CHECK_ERR(Encode(sliced_ishape, sliced_fshape, sliced_filter, indexer,
                     tmp_buf.get(), poly_degree()), "Encode");
    CHECK_ERR(A2H(tmp_buf.get(), poly_degree(), out.at(c), Role::evaluator, is_ntt_form), "A2H");
  }

  return Code::OK;
}

Code TensorEncoder::EncodeImageShare(Role role, const U64Tensor &img_tensor,
                                     const TensorShape &filter_shape,
                                     const Padding &padding,
                                     const size_t stride,
                                     const bool is_ntt_form,
                                     std::vector<RLWEPt> &out) const {
  ENSURE_OR_RETURN(img_tensor.dims() == 3 && filter_shape.dims() == 3,
                   Code::ERR_DIM_MISMATCH);

  TensorShape strided_tshape;
  std::array<int, 2> paddings{0};
  std::array<int, 3> slice_strides{0};
  if (!shape_inference::Conv2D(img_tensor.shape(), filter_shape, poly_degree(),
                               padding, stride, strided_tshape, paddings,
                               slice_strides)) {
    LOG(WARNING) << "EncodeImageShare: shape_inference failed";
    return Code::ERR_INVALID_ARG;
  }

  std::array<int, 3> strides3d{1, 1, 1};
  for (int d : {1, 2}) {
    if (strided_tshape.dim_size(d) < img_tensor.dim_size(d)) {
      strides3d[d] = static_cast<int>(stride);
    }
  }

  Strided3DTensor strided_img(img_tensor, strides3d);

  Conv2DSliceHelper helper(&strided_img, strided_img.shape(), filter_shape,
                           slice_strides, paddings);

  TensorShape mock_shape({0, 0, 0});
  for (int d = 0; d < 3; ++d) {
    mock_shape.Update(d, static_cast<int64_t>(slice_strides[d]));
  }

  const size_t n_slices = helper.num_slices();
  const size_t one_channel = helper.slice_size(1) * helper.slice_size(2);
  out.resize(n_slices);

  auto pool = seal::MemoryManager::GetPool(seal::mm_prof_opt::mm_force_thread_local);
  auto tmp_buf = seal::util::allocate_poly(poly_degree(), 1, pool);

  for (size_t i = 0; i < n_slices; ++i) {
    int c = static_cast<int>(i / one_channel);
    int h = static_cast<int>((i % one_channel) / helper.slice_size(2));
    int w = static_cast<int>(i % helper.slice_size(2));

    SlicedPaddedTensor<Strided3DTensor<U64Tensor>> sliced_padded_img;
    CHECK_ERR(helper.slice({c, h, w}, sliced_padded_img), "slice");
    sliced_padded_img.Mock(mock_shape);

    TensorShape si_shape = sliced_padded_img.shape();
    TensorShape sf_shape = filter_shape;
    sf_shape.Update(0, si_shape.channels());

    ImageIndexer indexer(poly_degree(), si_shape, sf_shape);

    CHECK_ERR(Encode(si_shape, sf_shape, sliced_padded_img, indexer,
                     tmp_buf.get(), poly_degree()),
              "Encode");

    CHECK_ERR(A2H(tmp_buf.get(), poly_degree(), out[i], role, is_ntt_form),
              "A2H");
  }
  return Code::OK;
}

template <class TensorType, class Indexer>
Code TensorEncoder::Encode(TensorShape ishape, TensorShape fshape,
                           const TensorType &tensor, const Indexer &indexer,
                           U64 *out_poly, size_t out_max_sze) const {
  ENSURE_OR_RETURN(out_poly != nullptr, Code::ERR_NULL_POINTER);
  ENSURE_OR_RETURN(fshape.IsValid() && ishape.IsValid(), Code::ERR_INVALID_ARG);
  ENSURE_OR_RETURN(fshape.channels() == ishape.channels(),
                   Code::ERR_DIM_MISMATCH);
  ENSURE_OR_RETURN(
      tensor.shape().IsSameSize(ishape) || tensor.shape().IsSameSize(fshape),
      Code::ERR_DIM_MISMATCH);

  const size_t N = poly_degree();
  const size_t n_elt = tensor.shape().num_elements();
  ENSURE_OR_RETURN(n_elt <= N && out_max_sze >= N, Code::ERR_OUT_BOUND);

  std::fill_n(out_poly, N, 0);
  TensorShape shape = tensor.shape();
  for (int c = 0; c < shape.channels(); ++c) {
    for (int h = 0; h < shape.height(); ++h) {
      for (int w = 0; w < shape.width(); ++w) {
        int coeff_index = indexer(c, h, w);
        if (coeff_index < 0 || coeff_index >= N) {
          LOG(FATAL) << "invalid index " << c << "," << h << "," << w;
        }
        out_poly[coeff_index] = tensor(c, h, w);
      }
    }
  }
  return Code::OK;
}

Code TensorEncoder::InitPtx(RLWEPt &pt, seal::parms_id_type pid) const {

  if (scheme() != seal::scheme_type::ckks) {
    // BFV or BGV
    pt.parms_id() = seal::parms_id_zero;  // foo SEAL when using BFV
    pt.resize(poly_degree());
    ENSURE_OR_RETURN(pt.data() != nullptr, Code::ERR_SEAL_MEMORY);
    return Code::OK;
  }

  if (pid == seal::parms_id_zero) {
    pid = rt_.first_parms_id();
  }

  auto cntxt_data = rt_.get_context_data(pid);
  ENSURE_OR_RETURN(cntxt_data != nullptr, Code::ERR_INTERNAL);
  const size_t num_moduli = cntxt_data->parms().coeff_modulus().size();
  const size_t num_elt = seal::util::mul_safe(num_moduli, poly_degree());
  pt.parms_id() = seal::parms_id_zero;  // foo SEAL when using BFV
  pt.resize(num_elt);
  pt.parms_id() = pid;
  ENSURE_OR_RETURN(pt.data() != nullptr, Code::ERR_SEAL_MEMORY);
  return Code::OK;
}

Code TensorEncoder::A2HBFV(const U64 *vec, size_t len, RLWEPt &pt,
                           const Role role, bool is_to_ntt) const {
  if (scheme() != seal::scheme_type::bfv) {
    LOG(FATAL) << "A2HBFV: invalid scheme";
  }

  if (is_to_ntt) {
    LOG(WARNING) << "A2H: demand is_to_ntt = false for scheme bfv";
  }

  CHECK_ERR(InitPtx(pt), "A2H: InitPtx");
  ENSURE_OR_RETURN(vec != nullptr, Code::ERR_NULL_POINTER);
  ENSURE_OR_RETURN(len > 0 && len <= poly_degree(), Code::ERR_OUT_BOUND);

  seal::util::modulo_poly_coeffs(vec, len, plain_modulus(), pt.data());
  std::fill_n(pt.data() + len, pt.coeff_count() - len, 0);

  return Code::OK;
}

Code TensorEncoder::A2H(const U64 *vec, size_t len, RLWEPt &pt, const Role role,
                        bool is_to_ntt) const {
  switch (scheme()) {
    case seal::scheme_type::bfv:
      return A2HBFV(vec, len, pt, role, is_to_ntt);
    default:
      LOG(WARNING) << "A2H: shceme is not supported yet\n";
  }
  return Code::ERR_INTERNAL;
}

ConvCoeffIndexCalculator::ConvCoeffIndexCalculator(size_t poly_degree,
                                                   TensorShape ishape,
                                                   TensorShape fshape,
                                                   Padding padding,
                                                   size_t stride)
    : poly_degree_(poly_degree),
      stride_(stride),
      ishape_(ishape),
      fshape_(fshape) {
  if (!ishape.IsValid() || !fshape.IsValid()) LOG(FATAL) << "invalid shape";

  TensorShape strided_ishape;
  std::array<int, 2> paddings;
  std::array<int, 3> slice_strides;
  if (!shape_inference::Conv2D(ishape, fshape, poly_degree, padding, stride,
                               strided_ishape, paddings, slice_strides)) {
    LOG(FATAL) << "shape inference failed";
  }

  bool is_input_compressed =
      strided_ishape.num_elements() < ishape.num_elements();

  helper_.reset(new Conv2DSliceHelper<U64Tensor>(
      is_input_compressed ? strided_ishape : ishape, fshape, slice_strides,
      paddings));

  mock_shape_ = TensorShape({0, 0, 0});
  for (int d = 0; d < 3; ++d) {
    mock_shape_.Update(d, slice_strides[d]);
  }
}

int ConvCoeffIndexCalculator::slice_size(size_t d) const {
  return helper_->slice_size(d);
}

Code ConvCoeffIndexCalculator::Get(std::array<int, 2> coords,
                                   TensorShape &out_shape,
                                   std::vector<size_t> &indices) const {
  return Get(coords, out_shape, &indices);
}

Code ConvCoeffIndexCalculator::Get(std::array<int, 2> coords,
                                   TensorShape &out_shape,
                                   std::vector<size_t> *indices) const {
  ENSURE_OR_RETURN(coords[0] >= 0 && coords[0] < slice_size(1),
                   Code::ERR_INVALID_ARG);
  ENSURE_OR_RETURN(coords[1] >= 0 && coords[1] < slice_size(2),
                   Code::ERR_INVALID_ARG);

  TensorShape sliced_shape;
  CHECK_ERR(helper_->slice({0, coords[0], coords[1]}, sliced_shape), "slice");

  TensorShape mock_ishape = mock_shape_;
  TensorShape mock_fshape = fshape_;
  mock_ishape.Update(0, sliced_shape.dim_size(0));
  mock_fshape.Update(0, sliced_shape.dim_size(0));
  FilterIndexer filter_indexer(poly_degree_, mock_ishape, mock_fshape);
  ImageIndexer img_indexer(poly_degree_, mock_ishape, mock_fshape);

  long offsets[2]{0, 0};
  int out_dims[2];
  out_shape = TensorShape({1, 0, 0});
  for (int d : {1, 2}) {
    int tmp = helper_->slice_start_at(d, coords[d - 1]) % stride_;
    offsets[d - 1] = (stride_ - tmp) % stride_;

    tmp = sliced_shape.dim_size(d) - fshape_.dim_size(d);
    tmp += (stride_ - offsets[d - 1]);
    tmp /= stride_;

    out_dims[d - 1] = tmp;

    out_shape.Update(d, out_dims[d - 1]);
  }

  if (indices) {
    indices->resize(out_dims[0] * out_dims[1]);

    auto dst_ptr = indices->begin();
    long O = filter_indexer.index_begin();

    for (int h = 0; h < out_dims[0]; ++h) {
      for (int w = 0; w < out_dims[1]; ++w) {
        *dst_ptr++ = O + img_indexer(0, h * stride_ + offsets[0],
                                     w * stride_ + offsets[1]);
      }
    }
  }

  return Code::OK;
}

size_t ConvCoeffIndexCalculator::NumAllIndices() const {
  TensorShape oshape;
  size_t accum_sze = 0;
  for (int h = 0; h < slice_size(1); ++h) {
    for (int w = 0; w < slice_size(2); ++w) {
      Get({h, w}, oshape, nullptr);
      accum_sze += oshape.num_elements();
    }
  }
  return accum_sze;
}

}  // namespace gemini
