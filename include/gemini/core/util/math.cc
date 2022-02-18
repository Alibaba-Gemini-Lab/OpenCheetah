// Authors: Wen-jie Lu.
#include "gemini/core/util/math.h"
namespace gemini {
bool RU128(F64 f, U64 u128[2]) {
  constexpr F64 two_pow_64 = 4. * static_cast<F64>(1L << 62);
  constexpr F64 two_pow_128 = two_pow_64 * two_pow_64;
  f = std::fabs(f);
  if (f >= two_pow_128) {
    return false;
  }

  if (f >= two_pow_64) {
    u128[0] = static_cast<U64>(std::fmod(f, two_pow_64));
    u128[1] = static_cast<U64>(f / two_pow_64);
  } else {
    u128[0] = RInt(f);
    u128[1] = 0;
  }

  return true;
}
}  // namespace gemini
