/*
Original Work Copyright (c) 2018 Xiao Wang (wangxiao@gmail.com)
Modified Work Copyright (c) 2020 Microsoft Research

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

Enquiries about further applications and development opportunities are welcome.

Modified by Deevashwer Rathee
*/

#ifndef IO_CHANNEL_H__
#define IO_CHANNEL_H__
#include "utils/block.h"
#include "utils/group.h"
#include <memory> // std::align

/** @addtogroup IO
  @{
 */

namespace sci {
template <typename T> class IOChannel {
public:
    uint64_t counter = 0;
  void send_data(const void *data, int nbyte) {
      counter += nbyte;
    derived().send_data_internal(data, nbyte);
  }
  void recv_data(void *data, int nbyte) { derived().recv_data_internal(data, nbyte); }

  void send_block(const block128 *data, int nblock) {
    send_data(data, nblock * sizeof(block128));
  }

  void send_block(const block256 *data, int nblock) {
    send_data(data, nblock * sizeof(block256));
  }

  void recv_block(block128 *data, int nblock) {
    recv_data(data, nblock * sizeof(block128));
  }

  void send_pt(emp::Point *A, int num_pts = 1) {
    for (int i = 0; i < num_pts; ++i) {
      size_t len = A[i].size();
      A[i].group->resize_scratch(len);
      unsigned char *tmp = A[i].group->scratch;
      send_data(&len, 4);
      A[i].to_bin(tmp, len);
      send_data(tmp, len);
    }
  }

  void recv_pt(emp::Group *g, emp::Point *A, int num_pts = 1) {
    size_t len = 0;
    for (int i = 0; i < num_pts; ++i) {
      recv_data(&len, 4);
      g->resize_scratch(len);
      unsigned char *tmp = g->scratch;
      recv_data(tmp, len);
      A[i].from_bin(g, tmp, len);
    }
  }

  	void send_bool(bool * data, int length) {
		void * ptr = (void *)data;
		size_t space = length;
		const void * aligned = std::align(alignof(uint64_t), sizeof(uint64_t), ptr, space);
		if(aligned == nullptr)
			send_data(data, length);
		else{
			int diff = length - space;
			send_data(data, diff);
			send_bool_aligned((const bool*)aligned, length - diff);
		}
	}

	void recv_bool(bool * data, int length) {
		void * ptr = (void *)data;
		size_t space = length;
		void * aligned = std::align(alignof(uint64_t), sizeof(uint64_t), ptr, space);
		if(aligned == nullptr)
			recv_data(data, length);
		else{
			int diff = length - space;
			recv_data(data, diff);
			recv_bool_aligned((bool*)aligned, length - diff);
		}
	}


	void send_bool_aligned(const bool * data, int length) {
		unsigned long long * data64 = (unsigned long long * )data;
		int i = 0;
		for(; i < length/8; ++i) {
			unsigned long long mask = 0x0101010101010101ULL;
			unsigned long long tmp = 0;
#if defined(__BMI2__)
			tmp = _pext_u64(data64[i], mask);
#else
			// https://github.com/Forceflow/libmorton/issues/6
			for (unsigned long long bb = 1; mask != 0; bb += bb) {
				if (data64[i] & mask & -mask) { tmp |= bb; }
				mask &= (mask - 1);
			}
#endif
			send_data(&tmp, 1);
		}
		if (8*i != length)
			send_data(data + 8*i, length - 8*i);
	}
	void recv_bool_aligned(bool * data, int length) {
		unsigned long long * data64 = (unsigned long long *) data;
		int i = 0;
		for(; i < length/8; ++i) {
			unsigned long long mask = 0x0101010101010101ULL;
			unsigned long long tmp = 0;
			recv_data(&tmp, 1);
#if defined(__BMI2__)
			data64[i] = _pdep_u64(tmp, mask);
#else
			data64[i] = 0;
			for (unsigned long long bb = 1; mask != 0; bb += bb) {
				if (tmp & bb) {data64[i] |= mask & (-mask); }
				mask &= (mask - 1);
			}
#endif
		}
		if (8*i != length)
			recv_data(data + 8*i, length - 8*i);
	}


private:
  T &derived() { return *static_cast<T *>(this); }
};
/**@}*/
} // namespace sci
#endif // IO_CHANNEL_H__
