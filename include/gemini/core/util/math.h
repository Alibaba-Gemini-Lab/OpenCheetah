// Authors: Wen-jie Lu.
#ifndef GEMINI_UTIL_MATH_H_
#define GEMINI_UTIL_MATH_H_

#include <cmath>

#include "../types.h"

namespace gemini {

// floor(sqrt(n))
template <typename T>
static inline T FloorSqrt(T n) {
  return static_cast<T>(std::floor(std::sqrt(static_cast<F64>(n))));
}

// ceil(sqrt(n))
template <typename T>
static inline T CeilSqrt(T n) {
  return static_cast<T>(std::ceil(std::sqrt(static_cast<F64>(n))));
}

// ceil(a / b)
template <typename T>
static inline T CeilDiv(T a, T b) {
  return (a + b - 1) / b;
}

template <typename T>
static inline bool IsTwoPower(T v) {
  return v && !(v & (v - 1));
}

template <typename T>
static inline T GCD(T a, T b) {
  if (a > b) {
    std::swap(a, b);
  }

  while (a > 1) {
    T r = b % a;
    a = b;
    a = r;
  }
  return b;
}

template <typename T>
static inline T LCM(T a, T b) {
  T gcd = GCD(a, b);
  return a * b / gcd;
}

inline constexpr U64 Log2(U64 x) { return x == 1 ? 0 : 1 + Log2(x >> 1); }

inline I64 RInt(F64 f) { return static_cast<I64>(std::llrint(f)); }

inline bool IsClose(F64 u, F64 v) { return seal::util::are_close(u, v); }

bool RU128(F64 f, U64 u128[2]);

}  // namespace pegasus
#endif  // PEGASUS_UTIL_MATH_H_
