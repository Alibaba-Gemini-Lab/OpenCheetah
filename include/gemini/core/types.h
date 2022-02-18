// Authors: Wen-jie Lu.
#ifndef GEMINI_CORE_TYPES_H
#define GEMINI_CORE_TYPES_H

#include <seal/ciphertext.h>

#include <cstdint>
#include <optional>
#include <string>

namespace seal {
class SEALContext;
class Ciphertext;
class Plaintext;
class GaloisKeys;
class Encryptor;
class Decryptor;
class Evaluator;
class KSwitchKeys;
};  // namespace seal

using F64 = double;
using U64 = uint64_t;
using I64 = int64_t;

using RunTime = seal::SEALContext;
using RLWECt = seal::Ciphertext;
using RLWEPt = seal::Plaintext;

enum class Code {
  OK,
  ERR_CONFIG,
  ERR_NULL_POINTER,
  ERR_DIM_MISMATCH,
  ERR_SEAL_MEMORY,
  ERR_KEY_MISSING,
  ERR_OUT_BOUND,
  ERR_INVALID_ARG,
  ERR_INTERNAL,
};

std::string CodeMessage(Code code);

#endif  // GEMINI_CORE_H
