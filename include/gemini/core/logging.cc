// Authors: Wen-jie Lu.
// Basically taken from TensorFlow
#include "gemini/core/logging.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <string>

namespace gemini {
namespace internal {

int ParseInteger(const char* str, size_t size) {
  // Ideally we would use env_var / safe_strto64, but it is
  // hard to use here without pulling in a lot of dependencies,
  // so we use std:istringstream instead
  std::string integer_str(str, size);
  std::istringstream ss(integer_str);
  int level = 0;
  ss >> level;
  return level;
}

// Parse log level (int64) from environment variable (char*)
int64_t LogLevelStrToInt(const char* env_var_val) {
  if (env_var_val == nullptr) {
    return 0;
  }
  return ParseInteger(env_var_val, strlen(env_var_val));
}

int64_t MinLogLevelFromEnv() {
  const char* env_var_val = getenv("GEMINI_CPP_MIN_LOG_LEVEL");
  return LogLevelStrToInt(env_var_val);
}

LogMessage::LogMessage(const char* fname, int line, int severity)
    : fname_(fname), line_(line), severity_(severity) {}

LogMessage& LogMessage::AtLocation(const char* fname, int line) {
  fname_ = fname;
  line_ = line;
  return *this;
}

LogMessage::~LogMessage() {
  // Read the min log level once during the first call to logging.
  static int64_t min_log_level = MinLogLevelFromEnv();
  if (severity_ >= min_log_level) {
    GenerateLogMessage();
  }
}

void LogMessage::GenerateLogMessage() {
  fprintf(stderr, "[%s:%d] %s\n", fname_, line_, str().c_str());
}

LogMessageFatal::LogMessageFatal(const char* file, int line)
    : LogMessage(file, line, FATAL) {}

LogMessageFatal::~LogMessageFatal() {
  // abort() ensures we don't return (we promised we would not via
  // ATTRIBUTE_NORETURN).
  GenerateLogMessage();
  abort();
}

}  // namespace internal
}  // namespace pegasus
