// Authors: Wen-jie Lu.
#ifndef GEMINI_CORE_UTIL_SEAL_H
#define GEMINI_CORE_UTIL_SEAL_H

#include "gemini/core/types.h"

namespace gemini {
Code apply_galois_inplace(seal::Ciphertext &encrypted, uint32_t galois_elt,
                          const seal::GaloisKeys &galois_keys,
                          const seal::SEALContext &context);

}  // namespace gemini
#endif  // GEMINI_CORE_UTIL_SEAL_H
