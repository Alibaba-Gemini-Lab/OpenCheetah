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
#include "LinearOT/hom-linear.h"
#include "LinearHE/elemwise-prod-field.h"

using namespace std;
using namespace seal;
using namespace sci;

int party = 0;
int bitlength = 32;
int num_threads = 1;
int port = 8000;
string address = "127.0.0.1";
int vec_size = 56 * 56;
int filter_precision = 15;

void ElemWiseProd(gemini::HomLinear &he_prod, int32_t len) {
  printf("EleWiseProd %zd, prime_mod %zd\n", len, prime_mod);

  PRG128 prg;
  gemini::TensorShape input_shape({len, 8, 8});
  std::cout << input_shape << "\n";
  gemini::Tensor<uint64_t> input(input_shape);
  gemini::Tensor<uint64_t> scale(gemini::TensorShape{input_shape.channels()});
  gemini::Tensor<uint64_t> output(input_shape);

  if (party == ALICE) {
    scale.Randomize(2 << filter_precision);
  }
  input.Randomize(prime_mod);

  gemini::HomLinear::BNMeta meta;
  meta.ishape = input_shape;
  meta.target_base_mod = prime_mod;
  meta.is_shared_input = true;
  meta.target_base_mod = prime_mod;

   he_prod.bn_direct(input, scale, meta, output);
   std::cout << output.tensor().unaryExpr([](uint64_t x) -> double {
     uint64_t upper = 1UL << (bitlength - 1);
     if (x >= upper) {
       return static_cast<int64_t>(x - (1UL << bitlength)) / std::pow(4., filter_precision);
     } else {
       return static_cast<int64_t>(x) / std::pow(4., filter_precision);
     }
   }) << "\n\n";
  // he_prod.elemwise_product(len, inArr, multArr, outArr, true, true);
}

int main(int argc, char **argv) {
  ArgMapping amap;
  amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
  amap.arg("p", port, "Port Number");
  amap.arg("s", vec_size, "Size of Vectors");
  amap.arg("fp", filter_precision, "Filter Precision");
  amap.arg("l", bitlength, "Bitlength of inputs");
  amap.parse(argc, argv);

  prime_mod = sci::default_prime_mod.at(bitlength);
  cout << "===================================================================="
          "===="
       << endl;
  cout << "Role: " << party << " - Bitlength: " << bitlength
       << " - Mod: " << prime_mod << " - Vector Size: " << vec_size
       << " - # Threads: " << num_threads << endl;
  cout << "===================================================================="
          "===="
       << endl;

  NetIO *io = new NetIO(party == 1 ? nullptr : address.c_str(), port);

  gemini::HomLinear hom_linear(party, io, prime_mod);
  ElemWiseProd(hom_linear, vec_size);
  return 0;
}
