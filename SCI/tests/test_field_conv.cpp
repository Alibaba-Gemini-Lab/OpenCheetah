/*
Authors: Deevashwer Rathee
Copyright:
Copyright (c) 2020 Microsoft Research
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

// #include "LinearOT/linear-ot.h"
#include "gemini/mvp/tensor.h"
#include "gemini/mvp/tensor_shape.h"
#include "library_fixed.h"

using namespace std;
using namespace seal;
using namespace sci;

int party = 0;
int bitlength = 40;
int num_threads = 4;
int port = 8000;
string address = "127.0.0.1";
int image_h = 56;
int inp_chans = 64;
int filter_h = 3;
int out_chans = 64;
int pad_l = 0;
int pad_r = 0;
int stride = 2;
int filter_precision = 12;

extern uint64_t moduloMask;
extern uint64_t moduloMidPt;

static inline int64_t getSignedVal(uint64_t x) {
  assert(x < prime_mod);
  int64_t sx = x;
  if (x >= moduloMidPt)
    sx = x - prime_mod;
  return sx;
}

static inline uint64_t getRingElt(int64_t x) { return ((uint64_t)x) & moduloMask; }

void TestImageNetFirstLayer(gemini::HomConv2DSSField &he_convss) {
  printf("Alice %d, Server %d\n", ALICE, SERVER);

  gemini::TensorShape ishape({3, 224, 224});
  gemini::TensorShape fshape({3, 7, 7});

  gemini::Tensor<uint64_t> image(ishape);
  std::vector<gemini::Tensor<uint64_t>> filters(64);

  if (he_convss.party() == BOB) {
    std::ifstream fin("/Users/juhou/MVP/cmake-build-release/resnet50_input_scale_12.inp");

    for (long h = 0; h < ishape.height(); ++h) {
      for (long w = 0; w < ishape.width(); ++w) {
        for (long c = 0; c < ishape.channels(); ++c) {
          fin >> image(c, h, w);
        }
      }
    }

    for (auto &f : filters) f.Reshape(fshape);
  } else {
    std::ifstream fin("/Users/juhou/Documents/codes/FHE/MVP/cmake-build-release/resnet50_model_scale_12.inp");

    for (long h = 0; h < fshape.height(); ++h) {
      for (long w = 0; w < fshape.width(); ++w) {
        for (long c = 0; c < fshape.channels(); ++c) {
          for (long m = 0; m < filters.size(); ++m) {
            filters[m].Reshape(fshape);
            fin >> filters[m](c, h, w);
          }
        }
      }
    }
  }

  gemini::HomConv2DSSField::Meta meta;
  meta.ishape = ishape;
  meta.fshape = fshape;
  meta.n_filters = 64;
  meta.stride = 2;
  meta.padding = gemini::Padding::SAME;
  meta.is_shared_input = false;

  gemini::Tensor<uint64_t> out_tensor;
  he_convss.run(image, filters, meta, out_tensor);
  he_convss.verify(image, filters, meta, out_tensor, filter_precision);
}

void Conv(gemini::HomConv2DSSField &he_conv, int32_t H, int32_t CI, int32_t FH,
          int32_t CO, int32_t zPadHLeft, int32_t zPadHRight, int32_t strideH) {
  int newH = 1 + (H + zPadHLeft + zPadHRight - FH) / strideH;
  int N = 1;
  int W = H;
  int FW = FH;
  int zPadWLeft = zPadHLeft;
  int zPadWRight = zPadHRight;
  int strideW = strideH;
  int newW = newH;

  gemini::HomConv2DSSField::Meta meta;
  meta.ishape = gemini::TensorShape({CI, H, W});
  meta.fshape = gemini::TensorShape({CI, FH, FW});
  meta.n_filters = CO;
  meta.stride = strideH;
  meta.padding = zPadHRight + zPadHLeft == 0 ? gemini::Padding::VALID
                                             : gemini::Padding::SAME;
  meta.is_shared_input = true;

  gemini::Tensor<uint64_t> input_tensor;

  std::vector<gemini::Tensor<uint64_t>> filters(meta.n_filters);
  input_tensor.Reshape(meta.ishape);

  gemini::Tensor<uint64_t> out_tensor;

  if (he_conv.party() == sci::ALICE) {
    input_tensor.Randomize(128ULL << filter_precision);
    input_tensor.tensor() = input_tensor.tensor().unaryExpr( [](uint64_t v) { return getRingElt(v); });

    for (size_t i = 0; i < meta.n_filters; ++i) {
      filters[i].Reshape(meta.fshape);
      gemini::Tensor<double> f64_input(meta.fshape);

      f64_input.Randomize(0.3);

      filters[i].tensor() = f64_input.tensor().unaryExpr([](double u) {
        auto sign = std::signbit(u);
        uint64_t su = std::floor(std::abs(u * (1 << filter_precision)));
        return getRingElt(sign ? -su : su);
      });

    }

  } else {
    input_tensor.Randomize(128ULL << filter_precision);
    input_tensor.tensor() = input_tensor.tensor().unaryExpr( [](uint64_t v) { return getRingElt(-v); });

    // Dummy filter
    for (auto &f : filters) {
      f.Reshape(meta.fshape);
    }
  }

  uint64_t comm_start = he_conv.io_counter();
  INIT_TIMER;
  START_TIMER;
  he_conv.run(input_tensor, filters, meta, out_tensor);
  STOP_TIMER("Total Time for Conv");
  uint64_t comm_end = he_conv.io_counter();
  cout << "Total Comm (MB): " << (comm_end - comm_start) / (1.0 * (1ULL << 20))
       << endl;
  he_conv.verify(input_tensor, filters, meta, out_tensor, filter_precision);
}

int main(int argc, char **argv) {
  ArgMapping amap;
  amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
  amap.arg("nt", num_threads, "Number of Threads");
  amap.arg("l", bitlength, "Bitlength");
  amap.arg("ip", address, "IP Address of server (ALICE)");
  amap.arg("p", port, "Port Number");
  amap.arg("h", image_h, "Image Height/Width");
  amap.arg("f", filter_h, "Filter Height/Width");
  amap.arg("i", inp_chans, "Input Channels");
  amap.arg("o", out_chans, "Ouput Channels");
  amap.arg("s", stride, "stride");
  amap.arg("pl", pad_l, "Left Padding");
  amap.arg("pr", pad_r, "Right Padding");
  amap.arg("fp", filter_precision, "Filter Precision");
  amap.parse(argc, argv);

#ifdef SCI_OT
  printf("base mod %llx\n", prime_mod);
#else
  prime_mod = sci::default_prime_mod.at(bitlength);
#endif

  cout << "=================================================================="
       << endl;
  printf(
      "Role: %s, Image: %dx%dx%d, Filter: %dx%dx%d\nStride: "
      "%dx%d, Padding %dx%d, Threads %d, precision %d\n",
      (party == BOB ? "BOB" : "ALICE"), image_h, image_h,
      inp_chans, filter_h, filter_h, out_chans, stride, stride, pad_l, pad_r,
      num_threads, filter_precision);
  cout << "=================================================================="
       << endl;

  NetIO *io = new NetIO(party == 1 ? nullptr : address.c_str(), port);

  // ConvField he_conv(party, io);
  gemini::HomConv2DSSField he_conv_ss(party, io);
//  Conv(he_conv_ss, image_h, inp_chans, filter_h, out_chans, pad_l, pad_r, stride);
  TestImageNetFirstLayer(he_conv_ss);
  io->flush();
  return 0;
}
