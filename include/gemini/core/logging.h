// Authors: Wen-jie Lu.
// Basically taken from TensorFlow
#ifndef GEMINI_CORE_LOGGING_H_
#define GEMINI_CORE_LOGGING_H_

#include <atomic>
#include <limits>
#include <memory>
#include <sstream>

namespace gemini {
const int INFO = 0;
const int WARNING = 1;
const int ERROR = 2;
const int FATAL = 3;
const int NUM_SEVERITIES = 4;

namespace internal {

class LogMessage : public std::basic_ostringstream<char> {
 public:
  LogMessage(const char* fname, int line, int severity);
  ~LogMessage() override;

  // Change the location of the log message.
  LogMessage& AtLocation(const char* fname, int line);

 protected:
  void GenerateLogMessage();

 private:
  const char* fname_;
  int line_;
  int severity_;
};

// LogMessageFatal ensures the process will exit in failure after
// logging this message.
class LogMessageFatal : public LogMessage {
 public:
  LogMessageFatal(const char* file, int line);
  ~LogMessageFatal() override;
};

}  // namespace internal

#define _GEMINI_LOG_INFO \
  ::gemini::internal::LogMessage(__FILE__, __LINE__, ::gemini::INFO)
#define _GEMINI_LOG_WARNING \
  ::gemini::internal::LogMessage(__FILE__, __LINE__, ::gemini::WARNING)
#define _GEMINI_LOG_ERROR \
  ::gemini::internal::LogMessage(__FILE__, __LINE__, ::gemini::ERROR)
#define _GEMINI_LOG_FATAL ::gemini::internal::LogMessageFatal(__FILE__, __LINE__)

#define _GEMINI_LOG_QFATAL _GEMINI_LOG_FATAL

#define LOG(severity) _GEMINI_LOG_##severity
}  // namespace gemini

#endif  // PEGASUS_CORE_LOGGING_H_
