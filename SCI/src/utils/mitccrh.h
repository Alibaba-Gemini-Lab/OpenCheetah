#ifndef SCI_MITCCRH_H__
#define SCI_MITCCRH_H__
#include <emp-tool/utils/aes_opt.h>
#include <stdio.h>

#include <stdexcept>
#include <string>

namespace cheetah {

/*
 * With numKeys keys, use each key to encrypt numEncs blocks.
 */
#ifdef __x86_64__
template <int numKeys>
static inline void ParaEncExp(block* blks, AES_KEY* keys) {
  block* first = blks;
  for (size_t i = 0; i < numKeys; ++i) {
    block K = keys[i].rd_key[0];
    int numEncs = 1 << i;
    for (size_t j = 0; j < numEncs; ++j) {
      *blks = *blks ^ K;
      ++blks;
    }
  }

  for (unsigned int r = 1; r < 10; ++r) {
    blks = first;
    for (size_t i = 0; i < numKeys; ++i) {
      block K = keys[i].rd_key[r];
      int numEncs = 1 << i;
      for (size_t j = 0; j < numEncs; ++j) {
        *blks = _mm_aesenc_si128(*blks, K);
        ++blks;
      }
    }
  }

  blks = first;
  for (size_t i = 0; i < numKeys; ++i) {
    block K = keys[i].rd_key[10];
    int numEncs = 1 << i;
    for (size_t j = 0; j < numEncs; ++j) {
      *blks = _mm_aesenclast_si128(*blks, K);
      ++blks;
    }
  }
}
#elif __aarch64__
template <int numKeys>
static inline void ParaEncExp(block* _blks, AES_KEY* keys) {
  uint8x16_t* first = (uint8x16_t*)(_blks);

  for (unsigned int r = 0; r < 9; ++r) {
    auto blks = first;
    for (size_t i = 0; i < numKeys; ++i) {
      uint8x16_t K = vreinterpretq_u8_m128i(keys[i].rd_key[r]);
      int numEncs = 1 << i;
      for (size_t j = 0; j < numEncs; ++j, ++blks) *blks = vaeseq_u8(*blks, K);
    }
    blks = first;
    for (size_t i = 0; i < numKeys; ++i) {
      int numEncs = 1 << i;
      for (size_t j = 0; j < numEncs; ++j, ++blks) *blks = vaesmcq_u8(*blks);
    }
  }

  auto blks = first;
  for (size_t i = 0; i < numKeys; ++i) {
    uint8x16_t K = vreinterpretq_u8_m128i(keys[i].rd_key[9]);
    int numEncs = 1 << i;
    for (size_t j = 0; j < numEncs; ++j, ++blks)
      *blks = vaeseq_u8(*blks, K) ^ K;
  }
}
#endif

/*
 * [REF] Implementation of "Better Concrete Security for Half-Gates Garbling (in
 * the Multi-Instance Setting)" https://eprint.iacr.org/2019/1168.pdf
 */

template <int BatchSize = 8>
class MITCCRH {
 public:
  AES_KEY scheduled_key[BatchSize];
  block keys[BatchSize];

  void renew_ks(block* new_keys, int n) {
    for (int i = 0; i < n; ++i) keys[i] = new_keys[i];
    switch (n) {
      case 1:
        AES_opt_key_schedule<1>(keys, scheduled_key);
        break;
      case 2:
        AES_opt_key_schedule<2>(keys, scheduled_key);
        break;
      case 3:
        AES_opt_key_schedule<3>(keys, scheduled_key);
        break;
      case 4:
        AES_opt_key_schedule<4>(keys, scheduled_key);
        break;
      case 8:
        AES_opt_key_schedule<8>(keys, scheduled_key);
        break;
      default:
        throw std::invalid_argument(string("MITCCRH not implemented: ") +
                                    std::to_string(n));
    }
  }

  void hash_exp(block* out, const block* in, int n) {
    int n_blks = (1 << n) - 1;
    for (int i = 0; i < n_blks; ++i) out[i] = in[i];

    switch (n) {
      case 1:
        ParaEncExp<1>(out, scheduled_key);
        break;
      case 2:
        ParaEncExp<2>(out, scheduled_key);
        break;
      case 3:
        ParaEncExp<3>(out, scheduled_key);
        break;
      case 4:
        ParaEncExp<4>(out, scheduled_key);
        break;
      case 8:
        ParaEncExp<8>(out, scheduled_key);
        break;
      default:
        throw std::invalid_argument(string("MITCCRH not implemented: ") +
                                    std::to_string(n));
    }

    // for(int i = 0; i < n_blks; ++i)
    // out[i] = in[i] ^ out[i];
  }

  void hash_single(block* out, const block* in, int n) {
    int n_blks = n;
    for (int i = 0; i < n_blks; ++i) out[i] = in[i];

    switch (n) {
      case 1:
        ParaEnc<1, 1>(out, scheduled_key);
        break;
      case 2:
        ParaEnc<2, 1>(out, scheduled_key);
        break;
      case 3:
        ParaEnc<3, 1>(out, scheduled_key);
        break;
      case 4:
        ParaEnc<4, 1>(out, scheduled_key);
        break;
      case 8:
        ParaEnc<8, 1>(out, scheduled_key);
        break;
      default:
        throw std::invalid_argument(string("MITCCRH not implemented: ") +
                                    std::to_string(n));
    }

    // for(int i = 0; i < n_blks; ++i)
    // out[i] = in[i] ^ out[i];
  }
};
}  // namespace cheetah
#endif  // MITCCRH_H__

