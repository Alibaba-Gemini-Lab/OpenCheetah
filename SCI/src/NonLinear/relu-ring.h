/*
Authors: Mayank Rathee
Modified by Zhicong Huang and Wen-jie Lu
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

#ifndef RELU_RING_H__
#define RELU_RING_H__

#include "BuildingBlocks/aux-protocols.h"
#include "Millionaire/millionaire.h"
#include "NonLinear/relu-interface.h"

extern int32_t bitlength;
extern int32_t kScale;

#define RING 0
#define OFF_PLACE

template <typename IO, typename type>
class ReLURingProtocol : public ReLUProtocol<IO, type> {
public:
  IO *io = nullptr;
  sci::OTPack<IO> *otpack;
  TripleGenerator<IO> *triple_gen = nullptr;
  MillionaireProtocol<IO> *millionaire = nullptr;
  AuxProtocols *aux = nullptr;
  int party;
  int algeb_str;
  int l, b;
  int num_cmps;
  uint8_t two_small = 1 << 1;
  uint8_t zero_small = 0;
  uint64_t mask_take_32 = -1;
  uint64_t msb_one;
  uint64_t cut_mask;
  uint64_t relu_comparison_rhs;
  type mask_l;
  type relu_comparison_rhs_type;
  type cut_mask_type;
  type msb_one_type;

  // Constructor
  ReLURingProtocol(int party, int algeb_str, IO *io, int l, int b,
                   sci::OTPack<IO> *otpack) {
    this->party = party;
    this->algeb_str = algeb_str;
    this->io = io;
    this->l = l;
    this->b = b;
    this->otpack = otpack;
    this->millionaire = new MillionaireProtocol<IO>(party, io, otpack);
    this->triple_gen = this->millionaire->triple_gen;
    this->aux = new AuxProtocols(party, io, otpack);
    configure();
  }

  // Destructor
  virtual ~ReLURingProtocol() { delete millionaire; }

  void configure() {
    if (this->l != 32 && this->l != 64) {
      mask_l = (type)((1ULL << l) - 1);
    } else if (this->l == 32) {
      mask_l = -1;
    } else { // l = 64
      mask_l = -1ULL;
    }
    if (sizeof(type) == sizeof(uint64_t)) {
      msb_one = (1ULL << (this->l - 1));
      relu_comparison_rhs_type = msb_one - 1ULL;
      relu_comparison_rhs = relu_comparison_rhs_type;
      cut_mask_type = relu_comparison_rhs_type;
      cut_mask = cut_mask_type;
    } else {
      msb_one_type = (1 << (this->l - 1));
      relu_comparison_rhs_type = msb_one_type - 1;
      relu_comparison_rhs = relu_comparison_rhs_type + 0ULL;
      cut_mask_type = relu_comparison_rhs_type;
      cut_mask = cut_mask_type + 0ULL;
    }
  }

  // Ideal Functionality
  void drelu_ring_ideal_func(uint8_t *result, type *sh1, type *sh2,
                             int num_relu) {
    uint8_t *msb1 = new uint8_t[num_relu];
    uint8_t *msb2 = new uint8_t[num_relu];
    type *plain_value = new type[num_relu];
    for (int i = 0; i < num_relu; i++) {
      plain_value[i] = sh1[i] + sh2[i];
    }
    uint8_t *actual_drelu = new uint8_t[num_relu];

    uint64_t index_fetch = (sizeof(type) == sizeof(uint64_t)) ? 7 : 3;
    for (int i = 0; i < num_relu; i++) {
      msb1[i] = (*((uint8_t *)(&(sh1[i])) + index_fetch)) >> 7;
      msb2[i] = (*((uint8_t *)(&(sh2[i])) + index_fetch)) >> 7;
      actual_drelu[i] = (*((uint8_t *)(&(plain_value[i])) + index_fetch)) >> 7;
    }

    type *sh1_cut = new type[num_relu];
    type *sh2_cut = new type[num_relu];
    uint8_t *wrap = new uint8_t[num_relu];
    uint8_t *wrap_orig = new uint8_t[num_relu];
    uint8_t *relu_comparison_avoid_warning = new uint8_t[sizeof(type)];
    memcpy(relu_comparison_avoid_warning, &relu_comparison_rhs, sizeof(type));
    for (int i = 0; i < num_relu; i++) {
      sh1_cut[i] = sh1[i] & cut_mask;
      sh2_cut[i] = sh2[i] & cut_mask;
      wrap_orig[i] =
          ((sh1_cut[i] + sh2_cut[i]) > *(type *)relu_comparison_avoid_warning);
      wrap[i] = wrap_orig[i];
      wrap[i] ^= msb1[i];
      wrap[i] ^= msb2[i];
    }
    memcpy(result, wrap, num_relu);
    for (int i = 0; i < num_relu; i++) {
      assert((wrap[i] == actual_drelu[i]) &&
             "The computed DReLU did not match the actual DReLU");
    }
  }

  void relu(type *result, type *share, int num_relu,
            uint8_t *drelu_res = nullptr, bool skip_ot = false,
            bool do_trunc = false, bool approx = false) {
    uint8_t *msb_local_share = new uint8_t[num_relu];
    uint64_t *array64;
    type *array_type;
    array64 = new uint64_t[num_relu];

    if (this->algeb_str == RING) {
      this->num_cmps = num_relu;
    } else {
      abort();
    }
    uint8_t *wrap = new uint8_t[num_cmps];

    if (approx) {
      // clang-format off
      // Ref: Kiwan Maeng and G. Edward Suh.
      // "Approximating ReLU on a Reduced Ring for Efficient MPC-based Private Inference"
      // clang-format on
      // NOTE(lwj): we don't drop too much for double width fixed-point.
      int lo = do_trunc ? kScale * 3 / 2 : kScale;
      int this_l = bitlength - lo;
      // NOTE(lwj): we can also drop some high-order bits
      this_l -= ((this_l - 1) % this->b);

      type _mask = (1 << this_l) - 1;
      type _upper = 1 << (this_l - 1);
      // x0, x1 \in [0, 2^l)
      // x'0, x'1 \in [0, 2^k)
      for (int i = 0; i < num_relu; i++) {
        array64[i] = static_cast<uint64_t>((share[i] >> lo) & _mask);

        msb_local_share[i] =
            static_cast<uint8_t>((array64[i] >> (this_l - 1)) & 1);

        if (party == sci::BOB) {
          array64[i] = (_upper - array64[i]) & _mask;
        }
      }

      if (do_trunc) {
        printf("Mill dotrunc %d on %d bits using b = %d\n", do_trunc,
               this_l - 1, this->b);
      }

      this->millionaire->compare(wrap, array64, num_cmps, this_l - 1, true,
                                 false, this->b);

      for (int i = 0; i < num_relu; i++) {
        msb_local_share[i] = (msb_local_share[i] + wrap[i]) % two_small;
      }

      if (drelu_res != nullptr) {
        for (int i = 0; i < num_relu; i++) {
          drelu_res[i] = msb_local_share[i];
        }
      }

      if (party == sci::ALICE) {
        for (int i = 0; i < num_relu; i++) {
          msb_local_share[i] = msb_local_share[i] ^ 1;
        }
      }

      aux->multiplexer(msb_local_share, share, result, num_relu, this->l,
                       this->l);

      delete[] msb_local_share;
      delete[] wrap;

      io->flush();
      return;
    }
    ///------------///

    array_type = new type[num_relu];
    for (int i = 0; i < num_relu; i++) {
      msb_local_share[i] = (uint8_t)(share[i] >> (l - 1));
      array_type[i] = share[i] & cut_mask_type;
    }

    type temp;

    switch (this->party) {
    case sci::ALICE: {
      for (int i = 0; i < num_relu; i++) {
        array64[i] = array_type[i] + 0ULL;
      }
      break;
    }
    case sci::BOB: {
      for (int i = 0; i < num_relu; i++) {
        temp = this->relu_comparison_rhs_type -
               array_type[i]; // This value is never negative.
        array64[i] = 0ULL + temp;
      }
      break;
    }
    }

    // DRelu(x) = 1 ^ msb(x)
    //          = 1 ^ msb(x0) ^ msb(x1) ^ 1(x0 + x1 > 2^{l - 1})
    this->millionaire->compare(wrap, array64, num_cmps, l - 1, true, false, b);
    for (int i = 0; i < num_relu; i++) {
      msb_local_share[i] = (msb_local_share[i] + wrap[i]) % two_small;
    }

    if (drelu_res != nullptr) {
      for (int i = 0; i < num_relu; i++) {
        drelu_res[i] = msb_local_share[i];
      }
    }

    if (skip_ot) {
      delete[] msb_local_share;
      delete[] array64;
      delete[] array_type;
      return;
    }

#if !USE_CHEETAH
    // Now perform x.msb(x)
    uint64_t **ot_messages = new uint64_t *[num_relu];
    for (int i = 0; i < num_relu; i++) {
      ot_messages[i] = new uint64_t[2];
    }
    uint64_t *additive_masks = new uint64_t[num_relu];
    uint64_t *received_shares = new uint64_t[num_relu];
    this->triple_gen->prg->random_data(additive_masks, num_relu * sizeof(type));
    switch (this->party) {
    case sci::ALICE: {
      for (int i = 0; i < num_relu; i++) {
        set_relu_end_ot_messages(ot_messages[i], share + i, msb_local_share + i,
                                 ((type *)additive_masks) + i);
      }
      otpack->iknp_straight->send(ot_messages, num_relu, this->l);
      otpack->iknp_reversed->recv(received_shares, msb_local_share, num_relu,
                                  this->l);
      break;
    }
    case sci::BOB: {
      for (int i = 0; i < num_relu; i++) {
        set_relu_end_ot_messages(ot_messages[i], share + i, msb_local_share + i,
                                 ((type *)additive_masks) + i);
      }
      otpack->iknp_straight->recv(received_shares, msb_local_share, num_relu,
                                  this->l);
      otpack->iknp_reversed->send(ot_messages, num_relu, this->l);
      break;
    }
    }
    for (int i = 0; i < num_relu; i++) {
      result[i] = ((type *)additive_masks)[i] +
                  ((type *)received_shares)[(8 / sizeof(type)) * i];
      result[i] &= mask_l;
    }
    delete[] msb_local_share;
    delete[] array64;
    delete[] array_type;
    delete[] additive_masks;
    delete[] received_shares;
    for (int i = 0; i < num_relu; i++) {
      delete[] ot_messages[i];
    }
    delete[] ot_messages;
    delete[] wrap;
#else
    if (party == sci::ALICE) {
      for (int i = 0; i < num_relu; i++)
        msb_local_share[i] = msb_local_share[i] ^ 1;
    }
    aux->multiplexer(msb_local_share, share, result, num_relu, this->l,
                     this->l);
    delete[] msb_local_share;
    delete[] array64;
    delete[] array_type;
    delete[] wrap;
#endif
    io->flush();
  }

  void set_relu_end_ot_messages(uint64_t *ot_messages, type *value_share,
                                uint8_t *xor_share, type *additive_mask) {
    type temp0, temp1;
    temp0 = (value_share[0] - additive_mask[0]);
    temp1 = (0 - additive_mask[0]);
    if (*xor_share == zero_small) {
      ot_messages[0] = 0ULL + temp0;
      ot_messages[1] = 0ULL + temp1;
    } else {
      ot_messages[0] = 0ULL + temp1;
      ot_messages[1] = 0ULL + temp0;
    }
  }
};

#endif // RELU_RING_H__
