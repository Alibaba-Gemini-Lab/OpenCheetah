// Author: Zhicong Huang
#ifndef CHEETAH_SILENT_OT_H
#define CHEETAH_SILENT_OT_H

#include <emp-ot/cot.h>
#include <emp-ot/ferret/ferret_cot.h>
#include <math.h>

#include <algorithm>
#include <atomic>
#include <stdexcept>

#include "OT/ot-utils.h"
#include "OT/ot.h"
#include "utils/mitccrh.h"
#include "utils/performance.h"

namespace cheetah {

template <typename IO> class SilentOT : public sci::OT<SilentOT<IO>> {
  std::atomic<int64_t> count_rcot_;

public:
  FerretCOT<IO> *ferret;
  cheetah::MITCCRH<8> mitccrh;

  SilentOT(int party, int threads, IO **ios, bool malicious = false,
           bool run_setup = true, std::string pre_file = "",
           bool warm_up = true) {
    ferret =
        new FerretCOT<IO>(party, threads, ios, malicious, run_setup, pre_file);
    if (warm_up) {
      block tmp;
      ferret->rcot(&tmp, 1);
    }
    count_rcot_ = 0;
  }

  ~SilentOT() { delete ferret; }

  void send_impl(const block *data0, const block *data1, int64_t length) {
    send_ot_cm_cc(data0, data1, length);
  }

  void recv_impl(block *data, const bool *b, int64_t length) {
    recv_ot_cm_cc(data, b, length);
  }

  template <typename T> void send_impl(T **data, int length, int l) {
    send_ot_cm_cc(data, length, l);
  }

  template <typename T>
  void recv_impl(T *data, const uint8_t *b, int length, int l) {
    recv_ot_cm_cc(data, b, length, l);
  }

  template <typename T> void send_impl(T **data, int length, int N, int l) {
    send_ot_cm_cc(data, length, N, l);
  }

  template <typename T>
  void recv_impl(T *data, const uint8_t *b, int length, int N, int l) {
    recv_ot_cm_cc(data, b, length, N, l);
  }

  void send_cot(uint64_t *data0, const uint64_t *corr, int length, int l) {
    send_ot_cam_cc(data0, corr, length, l);
  }

  void recv_cot(uint64_t *data, const bool *b, int length, int l) {
    recv_ot_cam_cc(data, b, length, l);
  }

  // chosen additive message, chosen choice
  // Sender chooses one message 'corr'. A correlation is defined by the addition
  // function: f(x) = x + corr Sender receives a random message 'x' as output
  // ('data0').
  void send_ot_cam_cc(uint64_t *data0, const uint64_t *corr, int64_t length,
                      int l) {
    uint64_t modulo_mask = (1ULL << l) - 1;
    if (l == 64)
      modulo_mask = -1;
    block *rcm_data = new block[length];
    send_ot_rcm_cc(rcm_data, length);

    block s;
    ferret->prg.random_block(&s, 1);
    ferret->io->send_block(&s, 1);
    ferret->mitccrh.setS(s);
    ferret->io->flush();

    block pad[2 * ot_bsize];
    uint32_t y_size = (uint32_t)ceil((ot_bsize * l) / (float(64)));
    uint32_t corrected_y_size, corrected_bsize;
    uint64_t y[y_size];
    uint64_t corr_data[ot_bsize];

    for (int64_t i = 0; i < length; i += ot_bsize) {
      for (int64_t j = i; j < std::min(i + ot_bsize, length); ++j) {
        pad[2 * (j - i)] = rcm_data[j];
        pad[2 * (j - i) + 1] = rcm_data[j] ^ ferret->Delta;
      }

      ferret->mitccrh.template hash<ot_bsize, 2>(pad);

      for (int j = i; j < i + ot_bsize and j < length; ++j) {
        data0[j] = _mm_extract_epi64(pad[2 * (j - i)], 0) & modulo_mask;
        corr_data[j - i] =
            (corr[j] + data0[j] + _mm_extract_epi64(pad[2 * (j - i) + 1], 0)) &
            modulo_mask;
      }
      corrected_y_size = (uint32_t)ceil((std::min(ot_bsize, length - i) * l) /
                                        ((float)sizeof(uint64_t) * 8));
      corrected_bsize = std::min(ot_bsize, length - i);

      sci::pack_cot_messages(y, corr_data, corrected_y_size, corrected_bsize,
                             l);
      ferret->io->send_data(y, sizeof(uint64_t) * (corrected_y_size));
    }

    delete[] rcm_data;
  }

  // chosen additive message, chosen choice
  // Receiver chooses a choice bit 'b', and
  // receives 'x' if b = 0, and 'x + corr' if b = 1
  void recv_ot_cam_cc(uint64_t *data, const bool *b, int64_t length, int l) {
    uint64_t modulo_mask = (1ULL << l) - 1;
    if (l == 64)
      modulo_mask = -1;

    block *rcm_data = new block[length];
    recv_ot_rcm_cc(rcm_data, b, length);
    block s;
    ferret->io->recv_block(&s, 1);
    ferret->mitccrh.setS(s);
    // ferret->io->flush();

    block pad[ot_bsize];

    uint32_t recvd_size = (uint32_t)ceil((ot_bsize * l) / (float(64)));
    uint32_t corrected_recvd_size, corrected_bsize;
    uint64_t corr_data[ot_bsize];
    uint64_t recvd[recvd_size];

    for (int64_t i = 0; i < length; i += ot_bsize) {
      corrected_recvd_size =
          (uint32_t)ceil((std::min(ot_bsize, length - i) * l) / (float(64)));
      corrected_bsize = std::min(ot_bsize, length - i);

      memcpy(pad, rcm_data + i, std::min(ot_bsize, length - i) * sizeof(block));
      ferret->mitccrh.template hash<ot_bsize, 1>(pad);

      ferret->io->recv_data(recvd, sizeof(uint64_t) * corrected_recvd_size);

      sci::unpack_cot_messages(corr_data, recvd, corrected_bsize, l);

      for (int j = i; j < i + ot_bsize and j < length; ++j) {
        if (b[j])
          data[j] = (corr_data[j - i] - _mm_extract_epi64(pad[j - i], 0)) &
                    modulo_mask;
        else
          data[j] = _mm_extract_epi64(pad[j - i], 0) & modulo_mask;
      }
    }

    delete[] rcm_data;
  }

  // chosen message, chosen choice
  void send_ot_cm_cc(const block *data0, const block *data1, int64_t length) {
    block *data = new block[length];
    send_ot_rcm_cc(data, length);

    block s;
    ferret->prg.random_block(&s, 1);
    ferret->io->send_block(&s, 1);
    ferret->mitccrh.setS(s);
    ferret->io->flush();

    block pad[2 * ot_bsize];
    for (int64_t i = 0; i < length; i += ot_bsize) {
      for (int64_t j = i; j < std::min(i + ot_bsize, length); ++j) {
        pad[2 * (j - i)] = data[j];
        pad[2 * (j - i) + 1] = data[j] ^ ferret->Delta;
      }
      // here, ferret depends on the template parameter "IO", making mitccrh
      // also dependent, hence we have to explicitly tell the compiler that
      // "hash" is a template function. See:
      // https://stackoverflow.com/questions/7397934/calling-template-function-within-template-class
      ferret->mitccrh.template hash<ot_bsize, 2>(pad);
      for (int64_t j = i; j < std::min(i + ot_bsize, length); ++j) {
        pad[2 * (j - i)] = pad[2 * (j - i)] ^ data0[j];
        pad[2 * (j - i) + 1] = pad[2 * (j - i) + 1] ^ data1[j];
      }
      ferret->io->send_data(pad,
                            2 * sizeof(block) * std::min(ot_bsize, length - i));
    }
    delete[] data;
  }

  // chosen message, chosen choice
  void recv_ot_cm_cc(block *data, const bool *r, int64_t length) {
    recv_ot_rcm_cc(data, r, length);

    block s;
    ferret->io->recv_block(&s, 1);
    ferret->mitccrh.setS(s);
    // ferret->io->flush();

    block res[2 * ot_bsize];
    block pad[ot_bsize];
    for (int64_t i = 0; i < length; i += ot_bsize) {
      memcpy(pad, data + i, std::min(ot_bsize, length - i) * sizeof(block));
      ferret->mitccrh.template hash<ot_bsize, 1>(pad);
      ferret->io->recv_data(res,
                            2 * sizeof(block) * std::min(ot_bsize, length - i));
      for (int64_t j = 0; j < ot_bsize and j < length - i; ++j) {
        data[i + j] = res[2 * j + r[i + j]] ^ pad[j];
      }
    }
  }

  // chosen message, chosen choice.
  // Here, the 2nd dim of data is always 2. We use T** instead of T*[2] or two
  // arguments of T*, in order to be general and compatible with the API of
  // 1-out-of-N OT.
  template <typename T> void send_ot_cm_cc(T **data, int64_t length, int l) {
    block *rcm_data = new block[length];
    send_ot_rcm_cc(rcm_data, length);

    block s;
    ferret->prg.random_block(&s, 1);
    ferret->io->send_block(&s, 1);
    ferret->mitccrh.setS(s);
    ferret->io->flush();

    block pad[2 * ot_bsize];
    uint32_t y_size =
        (uint32_t)ceil((2 * ot_bsize * l) / ((float)sizeof(T) * 8));
    uint32_t corrected_y_size, corrected_bsize;
    T y[y_size];

    for (int64_t i = 0; i < length; i += ot_bsize) {
      for (int64_t j = i; j < std::min(i + ot_bsize, length); ++j) {
        pad[2 * (j - i)] = rcm_data[j];
        pad[2 * (j - i) + 1] = rcm_data[j] ^ ferret->Delta;
      }
      // here, ferret depends on the template parameter "IO", making mitccrh
      // also dependent, hence we have to explicitly tell the compiler that
      // "hash" is a template function. See:
      // https://stackoverflow.com/questions/7397934/calling-template-function-within-template-class
      ferret->mitccrh.template hash<ot_bsize, 2>(pad);

      corrected_y_size = (uint32_t)ceil(
          (2 * std::min(ot_bsize, length - i) * l) / ((float)sizeof(T) * 8));
      corrected_bsize = std::min(ot_bsize, length - i);

      sci::pack_ot_messages<T>((T *)y, data + i, pad, corrected_y_size,
                               corrected_bsize, l, 2);

      ferret->io->send_data(y, sizeof(T) * (corrected_y_size));
    }
    delete[] rcm_data;
  }

  // chosen message, chosen choice
  // Here, r[i]'s value is always 0 or 1. We use uint8_t instead of bool, in
  // order to be general and compatible with the API of 1-out-of-N OT.
  template <typename T>
  void recv_ot_cm_cc(T *data, const uint8_t *r, int64_t length, int l) {
    block *rcm_data = new block[length];
    recv_ot_rcm_cc(rcm_data, (const bool *)r, length);

    block s;
    ferret->io->recv_block(&s, 1);
    ferret->mitccrh.setS(s);
    // ferret->io->flush();

    block pad[ot_bsize];

    uint32_t recvd_size =
        (uint32_t)ceil((2 * ot_bsize * l) / ((float)sizeof(T) * 8));
    uint32_t corrected_recvd_size, corrected_bsize;
    T recvd[recvd_size];

    for (int64_t i = 0; i < length; i += ot_bsize) {
      corrected_recvd_size = (uint32_t)ceil(
          (2 * std::min(ot_bsize, length - i) * l) / ((float)sizeof(T) * 8));
      corrected_bsize = std::min(ot_bsize, length - i);

      ferret->io->recv_data(recvd, sizeof(T) * (corrected_recvd_size));

      memcpy(pad, rcm_data + i, std::min(ot_bsize, length - i) * sizeof(block));
      ferret->mitccrh.template hash<ot_bsize, 1>(pad);

      sci::unpack_ot_messages<T>(data + i, r + i, (T *)recvd, pad,
                                 corrected_bsize, l, 2);
    }
    delete[] rcm_data;
  }

  // random correlated message, chosen choice
  void send_ot_rcm_cc(block *data0, int64_t length) {
    ferret->send_cot(data0, length);
    count_rcot_.fetch_add(length);
  }

  // random correlated message, chosen choice
  void recv_ot_rcm_cc(block *data, const bool *b, int64_t length) {
    ferret->recv_cot(data, b, length);
  }

  // random message, chosen choice
  void send_ot_rm_cc(block *data0, block *data1, int64_t length) {
    send_ot_rcm_cc(data0, length);
    block s;
    ferret->prg.random_block(&s, 1);
    ferret->io->send_block(&s, 1);
    ferret->mitccrh.setS(s);
    ferret->io->flush();

    block pad[ot_bsize * 2];
    for (int64_t i = 0; i < length; i += ot_bsize) {
      for (int64_t j = i; j < std::min(i + ot_bsize, length); ++j) {
        pad[2 * (j - i)] = data0[j];
        pad[2 * (j - i) + 1] = data0[j] ^ ferret->Delta;
      }
      ferret->mitccrh.template hash<ot_bsize, 2>(pad);
      for (int64_t j = i; j < std::min(i + ot_bsize, length); ++j) {
        data0[j] = pad[2 * (j - i)];
        data1[j] = pad[2 * (j - i) + 1];
      }
    }
  }

  // random message, chosen choice
  void recv_ot_rm_cc(block *data, const bool *r, int64_t length) {
    recv_ot_rcm_cc(data, r, length);
    block s;
    ferret->io->recv_block(&s, 1);
    ferret->mitccrh.setS(s);
    // ferret->io->flush();
    block pad[ot_bsize];
    for (int64_t i = 0; i < length; i += ot_bsize) {
      std::memcpy(pad, data + i,
                  std::min(ot_bsize, length - i) * sizeof(block));
      ferret->mitccrh.template hash<ot_bsize, 1>(pad);
      std::memcpy(data + i, pad,
                  std::min(ot_bsize, length - i) * sizeof(block));
    }
  }

  // random message, random choice
  void send_ot_rm_rc(block *data0, block *data1, int64_t length) {
    ferret->rcot(data0, length);

    block s;
    ferret->prg.random_block(&s, 1);
    ferret->io->send_block(&s, 1);
    ferret->mitccrh.setS(s);
    ferret->io->flush();

    block pad[ot_bsize * 2];
    for (int64_t i = 0; i < length; i += ot_bsize) {
      for (int64_t j = i; j < std::min(i + ot_bsize, length); ++j) {
        pad[2 * (j - i)] = data0[j];
        pad[2 * (j - i) + 1] = data0[j] ^ ferret->Delta;
      }
      ferret->mitccrh.template hash<ot_bsize, 2>(pad);
      for (int64_t j = i; j < std::min(i + ot_bsize, length); ++j) {
        data0[j] = pad[2 * (j - i)];
        data1[j] = pad[2 * (j - i) + 1];
      }
    }
  }

  // random message, random choice
  void recv_ot_rm_rc(block *data, bool *r, int64_t length) {
    ferret->rcot(data, length);
    for (int64_t i = 0; i < length; i++) {
      r[i] = getLSB(data[i]);
    }

    block s;
    ferret->io->recv_block(&s, 1);
    ferret->mitccrh.setS(s);
    // ferret->io->flush();
    block pad[ot_bsize];
    for (int64_t i = 0; i < length; i += ot_bsize) {
      std::memcpy(pad, data + i,
                  std::min(ot_bsize, length - i) * sizeof(block));
      ferret->mitccrh.template hash<ot_bsize, 1>(pad);
      std::memcpy(data + i, pad,
                  std::min(ot_bsize, length - i) * sizeof(block));
    }
  }

  // random message, random choice
  template <typename T>
  void send_ot_rm_rc(T *data0, T *data1, int64_t length, int l) {
    block *rm_data0 = new block[length];
    block *rm_data1 = new block[length];
    send_ot_rm_rc(rm_data0, rm_data1, length);

    T mask = (T)((1ULL << l) - 1ULL);

    for (int64_t i = 0; i < length; i++) {
      data0[i] = ((T)_mm_extract_epi64(rm_data0[i], 0)) & mask;
      data1[i] = ((T)_mm_extract_epi64(rm_data1[i], 0)) & mask;
    }

    delete[] rm_data0;
    delete[] rm_data1;
  }

  // random message, random choice
  template <typename T>
  void recv_ot_rm_rc(T *data, bool *r, int64_t length, int l) {
    block *rm_data = new block[length];
    recv_ot_rm_rc(rm_data, r, length);

    T mask = (T)((1ULL << l) - 1ULL);

    for (int64_t i = 0; i < length; i++) {
      data[i] = ((T)_mm_extract_epi64(rm_data[i], 0)) & mask;
    }

    delete[] rm_data;
  }

  // chosen message, chosen choice.
  // One-oo-N OT, where each message has l bits. Here, the 2nd dim of data is N.
  template <typename T>
  void send_ot_cm_cc(T **data, int64_t length, int N, int l) {
    int logN = (int)ceil(log2(N));

    block *rm_data0 = new block[length * logN];
    block *rm_data1 = new block[length * logN];
    send_ot_rm_cc(rm_data0, rm_data1, length * logN);

    block pad[ot_bsize * N];
    uint32_t y_size =
        (uint32_t)ceil((ot_bsize * N * l) / ((float)sizeof(T) * 8));
    uint32_t corrected_y_size, corrected_bsize;
    T y[y_size];

    block *hash_in0 = new block[N - 1];
    block *hash_in1 = new block[N - 1];
    block *hash_out = new block[2 * N - 2];
    int idx = 0;
    for (int x = 0; x < logN; x++) {
      for (int y = 0; y < (1 << x); y++) {
        hash_in0[idx] = makeBlock(y, 0);
        hash_in1[idx] = makeBlock((1 << x) + y, 0);
        idx++;
      }
    }

    for (int64_t i = 0; i < length; i += ot_bsize) {
      std::memset(pad, 0, sizeof(block) * N * ot_bsize);
      for (int64_t j = i; j < std::min(i + ot_bsize, length); ++j) {
        mitccrh.renew_ks(rm_data0 + j * logN, logN);
        mitccrh.hash_exp(hash_out, hash_in0, logN);
        mitccrh.renew_ks(rm_data1 + j * logN, logN);
        mitccrh.hash_exp(hash_out + N - 1, hash_in1, logN);

        for (int64_t k = 0; k < N; k++) {
          idx = 0;
          for (int64_t s = 0; s < logN; s++) {
            int mask = (1 << s) - 1;
            int pref = k & mask;
            if ((k & (1 << s)) == 0)
              pad[(j - i) * N + k] ^= hash_out[idx + pref];
            else
              pad[(j - i) * N + k] ^= hash_out[idx + N - 1 + pref];
            idx += 1 << s;
          }
        }
      }

      corrected_y_size = (uint32_t)ceil(
          (std::min(ot_bsize, length - i) * N * l) / ((float)sizeof(T) * 8));
      corrected_bsize = std::min(ot_bsize, length - i);

      sci::pack_ot_messages<T>((T *)y, data + i, pad, corrected_y_size,
                               corrected_bsize, l, N);

      ferret->io->send_data(y, sizeof(T) * (corrected_y_size));
    }

    delete[] hash_in0;
    delete[] hash_in1;
    delete[] hash_out;
    delete[] rm_data0;
    delete[] rm_data1;
  }

  // chosen message, chosen choice
  // One-oo-N OT, where each message has l bits. Here, r[i]'s value is in [0,
  // N).
  template <typename T>
  void recv_ot_cm_cc(T *data, const uint8_t *r, int64_t length, int N, int l) {
    int logN = (int)ceil(log2(N));

    block *rm_data = new block[length * logN];
    bool *b_choices = new bool[length * logN];
    for (int64_t i = 0; i < length; i++) {
      for (int64_t j = 0; j < logN; j++) {
        b_choices[i * logN + j] = (bool)((r[i] & (1 << j)) >> j);
      }
    }
    recv_ot_rm_cc(rm_data, b_choices, length * logN);

    block pad[ot_bsize];

    uint32_t recvd_size =
        (uint32_t)ceil((ot_bsize * N * l) / ((float)sizeof(T) * 8));
    uint32_t corrected_recvd_size, corrected_bsize;
    T recvd[recvd_size];

    block *hash_out = new block[logN];
    block *hash_in = new block[logN];

    for (int64_t i = 0; i < length; i += ot_bsize) {
      corrected_recvd_size = (uint32_t)ceil(
          (std::min(ot_bsize, length - i) * N * l) / ((float)sizeof(T) * 8));
      corrected_bsize = std::min(ot_bsize, length - i);

      ferret->io->recv_data(recvd, sizeof(T) * (corrected_recvd_size));

      std::memset(pad, 0, sizeof(block) * ot_bsize);
      for (int64_t j = i; j < std::min(i + ot_bsize, length); ++j) {
        for (int64_t s = 0; s < logN; s++)
          hash_in[s] = makeBlock(r[j] & ((1 << (s + 1)) - 1), 0);
        mitccrh.renew_ks(rm_data + j * logN, logN);
        mitccrh.hash_single(hash_out, hash_in, logN);

        for (int64_t s = 0; s < logN; s++) {
          pad[j - i] ^= hash_out[s];
        }
      }

      sci::unpack_ot_messages<T>(data + i, r + i, (T *)recvd, pad,
                                 corrected_bsize, l, N);
    }
    delete[] hash_in;
    delete[] hash_out;
    delete[] rm_data;
    delete[] b_choices;
  }

  void send_batched_got(uint64_t *data, int num_ot, int l,
                        int msgs_per_ot = 1) {
    throw std::logic_error("Not implemented");
  }

  void recv_batched_got(uint64_t *data, const uint8_t *r, int num_ot, int l,
                        int msgs_per_ot = 1) {
    throw std::logic_error("Not implemented");
  }

  void send_batched_cot(uint64_t *data0, uint64_t *corr,
                        std::vector<int> msg_len, int num_ot,
                        int msgs_per_ot = 1) {
    throw std::logic_error("Not implemented");
  }

  void recv_batched_cot(uint64_t *data, bool *b, std::vector<int> msg_len,
                        int num_ot, int msgs_per_ot = 1) {
    throw std::logic_error("Not implemented");
  }

  int64_t get_rcot_count() const { return count_rcot_.load(); }
};

template <typename IO> class SilentOTN : public sci::OT<SilentOTN<IO>> {
public:
  SilentOT<IO> *silent_ot;
  int N;

  SilentOTN(SilentOT<IO> *silent_ot, int N) {
    this->silent_ot = silent_ot;
    this->N = N;
  }

  template <typename T> void send_impl(T **data, int length, int l) {
    silent_ot->send_impl(data, length, N, l);
  }

  template <typename T>
  void recv_impl(T *data, const uint8_t *b, int length, int l) {
    silent_ot->recv_impl(data, b, length, N, l);
  }
};

} // namespace cheetah

#endif // CHEETAH_SILENT_OT_H
