// Authors: Wen-jie Lu.
#include "gemini/core/types.h"
std::string CodeMessage(Code code) {
  switch (code) {
    case Code::OK:
      return "ok";
    case Code::ERR_CONFIG:
      return "Error: configuration";
    case Code::ERR_NULL_POINTER:
      return "Error: null pointer";
    case Code::ERR_DIM_MISMATCH:
      return "Error: dimension mismatch";
    case Code::ERR_SEAL_MEMORY:
      return "Error: memory allocation";
    case Code::ERR_OUT_BOUND:
      return "Error: out-of-bound";
    case Code::ERR_INVALID_ARG:
      return "Error: invalid arguments";
    case Code::ERR_INTERNAL:
      return "Error: internal error";
    default:
      return "Unknown code";
  }
}
