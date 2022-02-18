/*
Authors: Mayank Rathee
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

#include "library_fixed.h"
#include "library_fixed_uniform.h"
#include <iostream>

using namespace sci;
using namespace std;

int dim = 1 << 20;
int bw = 41;
int shift = 12;

int party = 0;
int bitlength = 41;
int num_threads = 1;
int port = 8000;
string address = "127.0.0.1";

PRG128 prg;

signedIntType getAnyRingSignedVal(intType x);
signedIntType div_floor(signedIntType a, signedIntType b);
inline intType getFieldMsb(intType x) { return (x > (prime_mod / 2)); }

template <typename intType>
void funcFieldDiv(int curParty, sci::NetIO *curio,
                  sci::OTPack<sci::NetIO> *curotpack,
                  sci::IKNP<sci::NetIO> *curiknp,
                  sci::KKOT<sci::NetIO> *curkkot,
                  ReLUProtocol<sci::NetIO, intType> *curReluImpl,
                  sci::PRG128 *curPrgInstance, int size, intType *inp,
                  intType *outp, intType divisor, uint8_t *msbShare) {
  assert(inp != outp &&
  "Assumption is there is a separate array for input and output");
  assert(size % 8 == 0 && "pad size to multiple of 8 and pass");
  assert((divisor > 0) && (divisor < prime_mod) &&
  "working with positive divisor");
  assert(prime_mod > 6 * divisor);
  const intType ringRem = prime_mod % divisor;
  const intType ringQuot = prime_mod / divisor;
  bool doMSBComputation = (msbShare == nullptr);
  if (doMSBComputation)
    msbShare = new uint8_t[size];

  for (int i = 0; i < size; i++) {
    assert(inp[i] < prime_mod && "input is not a valid share modulo prime_mod");
    signedIntType shareTerm1 =
        div_floor((signedIntType)getAnyRingSignedVal(inp[i]), divisor);
    intType localShareMSB = getFieldMsb(inp[i]);
    signedIntType shareTerm2 = div_floor(
        (signedIntType)((inp[i] % divisor) - localShareMSB * ringRem), divisor);
    signedIntType temp = shareTerm1 - shareTerm2;
    if (curParty == sci::BOB)
      temp += 1;
    outp[i] = sci::neg_mod(temp, (int64_t)prime_mod);
  }

  if (doMSBComputation) {
    curReluImpl->relu(nullptr, inp, size, msbShare, true);
  }

  static const int fieldBits = std::ceil(std::log2(prime_mod));
  const uint64_t bitsForA = std::ceil(std::log2(6 * divisor)); // delta in Algorithm 9.
  const uint64_t totalBitlen = fieldBits + bitsForA;
  bool OT1oo4FitsIn64Bits = (totalBitlen <= 64);

  intType *localShareCorr = new intType[size];
  int totalRandomBytesForSmallRing = sci::ceil_val(bitsForA * size, 8);
  uint8_t *localShareCorrSmallRingPacked =
      new uint8_t[totalRandomBytesForSmallRing];
  uint64_t *localShareCorrSmallRing = new uint64_t[size];
  intType *otMsgCorrField = new intType[4 * size];
  intType *otMsgCorrSmallRing = new intType[4 * size];

  if (curParty == sci::ALICE) {
    curPrgInstance->random_mod_p<intType>(localShareCorr, size, prime_mod);
    curPrgInstance->random_data(localShareCorrSmallRingPacked, totalRandomBytesForSmallRing);
    for (int i = 0; i < size; i++) {
      localShareCorrSmallRing[i] = sci::readFromPackedArr(
          localShareCorrSmallRingPacked, totalRandomBytesForSmallRing,
          i * bitsForA, bitsForA);
      uint8_t localShareMSB = getFieldMsb(inp[i]);
      for (int j = 0; j < 4; j++) {
        uint8_t b0 = j & 1;
        uint8_t b1 =
            (j >> 1) &
            1; // b1,b0 is the bit representation of j = [msb(a)]_1, msb(a_1)
            uint8_t temp = ((msbShare[i] + b1 + localShareMSB) & 1) &
                ((msbShare[i] + b1 + b0) & 1);
            signedIntType curMsg = -localShareCorr[i];
            intType curMsgSmallRing = -localShareCorrSmallRing[i];
            if (temp & (localShareMSB == 0)) {
              // msb(a_0)=0, msb(a_1)=0, msb(a)=1
              curMsg -= 1;
              curMsgSmallRing -= 1;
            } else if (temp & (localShareMSB == 1)) {
              // msb(a_0)=1, msb(a_1)=1, msb(a)=0
              curMsg += 1;
              curMsgSmallRing += 1;
            }
            intType curMsgField = sci::neg_mod(curMsg, (int64_t)prime_mod);
            curMsgSmallRing = curMsgSmallRing & sci::all1Mask(bitsForA);
            otMsgCorrField[i * 4 + j] = curMsgField;
            otMsgCorrSmallRing[i * 4 + j] = curMsgSmallRing;
      }
    }
    if (OT1oo4FitsIn64Bits) {
      uint64_t **otMessages1oo4 = new uint64_t *[size];
      for (int i = 0; i < size; i++) {
        otMessages1oo4[i] = new uint64_t[4];
        for (int j = 0; j < 4; j++)
          otMessages1oo4[i][j] =
              (((uint64_t)otMsgCorrSmallRing[i * 4 + j]) << fieldBits) +
              ((uint64_t)otMsgCorrField[i * 4 + j]);
      }
      curotpack->kkot[1]->send_impl(otMessages1oo4, size, totalBitlen);
      for (int i = 0; i < size; i++) {
        delete[] otMessages1oo4[i];
      }
      delete[] otMessages1oo4;
    } else {
      sci::block128 **otMessages1oo4 = new sci::block128 *[size];
      for (int i = 0; i < size; i++) {
        otMessages1oo4[i] = new sci::block128[4];
        for (int j = 0; j < 4; j++)
          otMessages1oo4[i][j] = _mm_set_epi64x(otMsgCorrSmallRing[i * 4 + j],
                                                otMsgCorrField[i * 4 + j]);
      }
      curkkot->send_impl(otMessages1oo4, size, 4);
      for (int i = 0; i < size; i++) {
        delete[] otMessages1oo4[i];
      }
      delete[] otMessages1oo4;
    }
    for (int i = 0; i < size; i++) {
#ifdef __SIZEOF_INT128__
      intType temp =
      (((__int128)localShareCorr[i]) * ((__int128)ringQuot)) % prime_mod;
#else
      intType temp = sci::moduloMult(localShareCorr[i], ringQuot, prime_mod);
#endif
      outp[i] = sci::neg_mod(outp[i] + temp, (int64_t)prime_mod);
    }
  } else {
    uint8_t *choiceBits = new uint8_t[size];
    for (int i = 0; i < size; i++) {
      uint8_t localShareMSB = getFieldMsb(inp[i]);
      choiceBits[i] = (msbShare[i] << 1) + localShareMSB;
    }

    if (OT1oo4FitsIn64Bits) {
      uint64_t *otRecvMsg1oo4 = new uint64_t[size];
      curotpack->kkot[1]->recv_impl(otRecvMsg1oo4, choiceBits, size,
                                    totalBitlen);
      for (int i = 0; i < size; i++) {
        otMsgCorrField[i] = otRecvMsg1oo4[i] & sci::all1Mask(fieldBits);
        otMsgCorrSmallRing[i] = otRecvMsg1oo4[i] >> fieldBits;
      }
      delete[] otRecvMsg1oo4;
    } else {
      sci::block128 *otRecvMsg1oo4 = new sci::block128[size];
      curkkot->recv_impl(otRecvMsg1oo4, choiceBits, size, 4);
      for (int i = 0; i < size; i++) {
        uint64_t temp = _mm_extract_epi64(otRecvMsg1oo4[i], 0);
        otMsgCorrField[i] = temp;
        temp = _mm_extract_epi64(otRecvMsg1oo4[i], 1);
        otMsgCorrSmallRing[i] = temp;
      }
      delete[] otRecvMsg1oo4;
    }

    for (int i = 0; i < size; i++) {
      localShareCorr[i] = otMsgCorrField[i];
      localShareCorrSmallRing[i] = otMsgCorrSmallRing[i];
#ifdef __SIZEOF_INT128__
      intType temp =
          (((__int128)localShareCorr[i]) * ((__int128)ringQuot)) % prime_mod;
#else
      intType temp = sci::moduloMult(localShareCorr[i], ringQuot, prime_mod);
#endif
      outp[i] = sci::neg_mod(outp[i] + temp, (int64_t)prime_mod);
    }

    delete[] choiceBits;
  }

  int totalComp = 3 * size;
  int compPerElt = 3;
  if (2 * ringRem < divisor) {
    // A+d<0 becomes moot
    totalComp = 2 * size;
    compPerElt = 2;
  }

  intType *localShareA_all3 = new intType[3 * size];
  uint8_t *localShareA_all3_drelu = new uint8_t[3 * size];
  uint64_t bitsAmask = sci::all1Mask(bitsForA);
  uint64_t bitsAMinusOneMask = sci::all1Mask((bitsForA - 1));
  uint64_t *radixCompValues = new uint64_t[3 * size];
  uint8_t *carryBit = new uint8_t[3 * size];
  for (int i = 0; i < size; i++) {
    intType localShareA =
        (inp[i] % divisor) -
        (getFieldMsb(inp[i]) - localShareCorrSmallRing[i]) * ringRem;
    for (int j = 0; j < compPerElt; j++) {
      localShareA_all3[compPerElt * i + j] = localShareA;
    }

    if (curParty == sci::ALICE) {
      if (compPerElt == 3) {
        localShareA_all3[3 * i] =
            (localShareA_all3[3 * i] - divisor) & bitsAmask;
        localShareA_all3[3 * i + 2] =
            (localShareA_all3[3 * i + 2] + divisor) & bitsAmask;
      } else {
        localShareA_all3[2 * i] =
            (localShareA_all3[2 * i] - divisor) & bitsAmask;
      }
    }
    for (int j = 0; j < compPerElt; j++) {
      radixCompValues[compPerElt * i + j] =
          (localShareA_all3[compPerElt * i + j] & bitsAMinusOneMask);
      localShareA_all3_drelu[compPerElt * i + j] =
          (localShareA_all3[compPerElt * i + j] >> (bitsForA - 1));
      if (curParty == sci::BOB) {
        radixCompValues[compPerElt * i + j] =
            bitsAMinusOneMask - radixCompValues[compPerElt * i + j];
      }
    }
  }

  MillionaireProtocol millionaire(curParty, curio, curotpack);
  millionaire.compare(carryBit, radixCompValues, totalComp, bitsForA - 1);
  for (int i = 0; i < totalComp; i++) {
    localShareA_all3_drelu[i] = (localShareA_all3_drelu[i] + carryBit[i]) & 1;
  }

  if (curParty == sci::ALICE) {
    uint64_t **otMsg = new uint64_t *[totalComp];
    intType *localShareDRelu = new intType[totalComp];
    curPrgInstance->random_mod_p<intType>(localShareDRelu, totalComp,
                                          prime_mod);
    for (int i = 0; i < totalComp; i++) {
      otMsg[i] = new uint64_t[2];
      otMsg[i][0] =
          sci::neg_mod(-localShareDRelu[i] + (localShareA_all3_drelu[i]),
                       (int64_t)prime_mod);
      otMsg[i][1] = sci::neg_mod(-localShareDRelu[i] +
          ((localShareA_all3_drelu[i] + 1) & 1),
          (int64_t)prime_mod);
    }
    curiknp->send_impl(otMsg, totalComp, fieldBits);
    for (int i = 0; i < size; i++) {
      intType curCTermShare = 0;
      for (int j = 0; j < compPerElt; j++) {
        curCTermShare =
            sci::neg_mod(curCTermShare + localShareDRelu[compPerElt * i + j],
                         (int64_t)prime_mod);
      }
      outp[i] = sci::neg_mod(outp[i] - curCTermShare, (int64_t)prime_mod);
      delete[] otMsg[i];
    }
    delete[] otMsg;
    delete[] localShareDRelu;
  } else {
    uint64_t *otDataRecvd = new uint64_t[totalComp];
    curiknp->recv_impl(otDataRecvd, localShareA_all3_drelu, totalComp,
                       fieldBits);
    for (int i = 0; i < size; i++) {
      intType curCTermShare = 0;
      for (int j = 0; j < compPerElt; j++) {
        uint64_t curDReluAns = otDataRecvd[compPerElt * i + j];
        curCTermShare =
            sci::neg_mod(curCTermShare + curDReluAns, (int64_t)prime_mod);
      }
      outp[i] = sci::neg_mod(outp[i] - curCTermShare, (int64_t)prime_mod);
    }
    delete[] otDataRecvd;
  }

  if (doMSBComputation)
    delete[] msbShare;
  delete[] localShareCorr;
  delete[] localShareCorrSmallRingPacked;
  delete[] localShareCorrSmallRing;
  delete[] otMsgCorrField;
  delete[] otMsgCorrSmallRing;
  delete[] localShareA_all3;
  delete[] localShareA_all3_drelu;
  delete[] radixCompValues;
  delete[] carryBit;
}

int main(int argc, char **argv) {
  ArgMapping amap;
  amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
  amap.arg("p", port, "Port Number");
  amap.arg("N", dim, "Number of ReLU operations");
  amap.arg("l", bw, "Bitlength of inputs");
  amap.arg("s", shift, "Bitlength of shift");
  amap.arg("ip", address, "IP Address of server (ALICE)");
  bw = bitlength;

  amap.parse(argc, argv);
  StartComputation();

  uint64_t *inputArr = new uint64_t[dim];
  uint64_t *outputArr = new uint64_t[dim];
  prg.random_mod_p<uint64_t>(inputArr, dim, prime_mod);

  if (party == SERVER) {
    // std::transform(inputArr, inputArr + dim, inputArr, [](uint64_t u) { return u >> (bitlength - 2 * shift); });
    prg.random_mod_p(inputArr, dim, prime_mod);
    io->send_data(inputArr, sizeof(uint64_t) * dim);

//    std::uniform_real_distribution<double> uniform(-1024., 1024.);
//    std::random_device rdv;

    for (int i = 0; i < dim; ++i) {
      double r = 128.; //uniform(rdv);
      uint64_t sr = static_cast<uint64_t>(std::floor(std::abs(r) * std::pow(4., shift)));
      sr %= prime_mod;
      if (std::signbit(r) && sr > 0) {
        sr = prime_mod - sr;
      }

      inputArr[i] = prime_mod + sr - inputArr[i];
      if (inputArr[i] >= prime_mod) {
        inputArr[i] -= prime_mod;
      }
    }
  } else {
    io->recv_data(inputArr, sizeof(uint64_t) * dim);
  }

  std::copy_n(inputArr, dim, outputArr);

  int64_t c0 = io->counter;
  ScaleDown(dim, outputArr, shift);
  int64_t c1 = io->counter;
  printf("Truncate %dbit in %d prime filed sent %ld bits\n", shift, (int) std::ceil(std::log2(prime_mod)), (c1 - c0) * 8 / dim);

  uint64_t *input = new uint64_t[dim];
  uint64_t *output = new uint64_t[dim];

  if (party == SERVER) {
    io->send_data(inputArr, sizeof(uint64_t) * dim);
    io->send_data(outputArr, sizeof(uint64_t) * dim);
  } else {
    io->recv_data(input, sizeof(uint64_t) * dim);
    io->recv_data(output, sizeof(uint64_t) * dim);

    for (int i = 0; i < dim; ++i) {
      input[i] = (input[i] + inputArr[i]) % prime_mod;
      output[i] = (output[i] + outputArr[i]) % prime_mod;
    }

    int one_bit_err = 0;
    int large_err = 0;
    int correct = 0;
    for (int i = 0; i < dim; ++i) {
      int64_t in = getAnyRingSignedVal(input[i]) >> shift;
      int64_t out = getAnyRingSignedVal(output[i]);
      int64_t diff = std::abs(in - out);
      if (diff > 1) {
        large_err += 1;
        if (large_err < 4) {
          std::cout << in / std::pow(2., shift) << " => " << out / std::pow(2., shift) << "\n";
        }
      } else {

        if (correct < 4) {
          std::cout << in  << " -> " << out  << "\n";
        }
        one_bit_err += (int) diff;
        correct += (int)(1 - diff);
      }
    }

    printf("local truncate 2^%d (%d), correct %d one-bit-error %d, large-error %d %f\n", shift, dim, correct, one_bit_err, large_err, large_err * 1. / dim);
  }

//  EndComputation();

  return 0;
}
