//  Authors: Wen-jie Lu.
#ifndef GEMINI_CORE_COMMON_H
#define GEMINI_CORE_COMMON_H
namespace gemini {
#define LOG_ERROR(msg)             \
  do {                             \
    std::cerr << msg << std::endl; \
  } while (0)

#define CHECK_ERR(state, msg)                                       \
  do {                                                              \
    auto code = state;                                              \
    if (code != Code::OK) {                                         \
      LOG_ERROR("[" << msg << "]@" << __FILE__ << "$" << __LINE__); \
      return code;                                                  \
    }                                                               \
  } while (0)

#define ENSURE_OR_RETURN(cond, code) \
  do {                               \
    if (!(cond)) return code;        \
  } while (0)

#define CATCH_SEAL_ERROR(state)                                      \
  do {                                                               \
    try {                                                            \
      state;                                                         \
    } catch (const std::logic_error &e) {                            \
      std::cerr << "SEAL_ERROR " << __FILE__ << __LINE__ << e.what() \
                << std::endl;                                        \
    }                                                                \
  } while (0)
}  // namespace gemini
#endif  // GEMINI_MVP_COMMON_H
