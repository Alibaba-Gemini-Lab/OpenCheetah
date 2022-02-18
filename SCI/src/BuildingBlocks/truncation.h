/*
Authors: Mayank Rathee, Deevashwer Rathee
Copyright:
Copyright (c) 2021 Microsoft Research
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

#ifndef TRUNCATION_H__
#define TRUNCATION_H__

#include "BuildingBlocks/aux-protocols.h"
#include "Millionaire/equality.h"
#include "Millionaire/millionaire_with_equality.h"

class Truncation {
public:
  sci::NetIO *io = nullptr;
  sci::OTPack<sci::NetIO> *otpack;
  TripleGenerator<sci::NetIO> *triple_gen = nullptr;
  MillionaireProtocol<sci::NetIO> *mill = nullptr;
  MillionaireWithEquality<sci::NetIO> *mill_eq = nullptr;
  Equality<sci::NetIO> *eq = nullptr;
  AuxProtocols *aux = nullptr;
  bool del_aux = false;
  bool del_milleq = false;
  int party;

  // Constructor
  Truncation(int party, sci::NetIO *io, sci::OTPack<sci::NetIO> *otpack,
             AuxProtocols *auxp = nullptr,
             MillionaireWithEquality<sci::NetIO> *mill_eq_in = nullptr);

  // Destructor
  ~Truncation();

  // Truncate (right-shift) by shift in the same ring (round towards -inf)
  void truncate(
      // Size of vector
      int32_t dim,
      // input vector
      uint64_t *inA,
      // output vector
      uint64_t *outB,
      // right shift amount
      int32_t shift,
      // Input and output bitwidth
      int32_t bw,
      // signed truncation?
      bool signed_arithmetic = true,
      // msb of input vector elements
      uint8_t *msb_x = nullptr);

    // Truncate (right-shift) by shift in the same ring (round towards -inf)
  void truncate_msb(
      // Size of vector
      int32_t dim,
      // input vector
      uint64_t *inA,
      // output vector
      uint64_t *outB,
      // right shift amount
      int32_t shift,
      // Input and output bitwidth
      int32_t bw,
      // signed truncation?
      bool signed_arithmetic,
      // msb of input vector elements
      uint8_t *msb_x);

    // Truncate (right-shift) by shift in the same ring (round towards -inf)
  // All elements have msb equal to 0. 
  void truncate_msb0(
      // Size of vector
      int32_t dim,
      // input vector
      uint64_t *inA,
      // output vector
      uint64_t *outB,
      // right shift amount
      int32_t shift,
      // Input and output bitwidth
      int32_t bw,
      // signed truncation?
      bool signed_arithmetic = true);


  // Divide by 2^shift in the same ring (round towards 0)
  void div_pow2(
      // Size of vector
      int32_t dim,
      // input vector
      uint64_t *inA,
      // output vector
      uint64_t *outB,
      // right shift amount
      int32_t shift,
      // Input and output bitwidth
      int32_t bw,
      // signed truncation?
      bool signed_arithmetic = true,
      // msb of input vector elements
      uint8_t *msb_x = nullptr);

  // Truncate (right-shift) by shift in the same ring
  void truncate_red_then_ext(
      // Size of vector
      int32_t dim,
      // input vector
      uint64_t *inA,
      // output vector
      uint64_t *outB,
      // right shift amount
      int32_t shift,
      // Input and output bitwidth
      int32_t bw,
      // signed truncation?
      bool signed_arithmetic = true,
      // msb of input vector elements
      uint8_t *msb_x = nullptr);

  // Truncate (right-shift) by shift and go to a smaller ring
  void truncate_and_reduce(
      // Size of vector
      int32_t dim,
      // input vector
      uint64_t *inA,
      // output vector
      uint64_t *outB,
      // right shift amount
      int32_t shift,
      // Input bitwidth
      int32_t bw);
};

#endif // TRUNCATION_H__
