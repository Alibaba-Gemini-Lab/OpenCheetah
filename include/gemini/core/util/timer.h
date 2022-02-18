// Authors: Wen-jie Lu.
#ifndef CORE_UTIL_TIMER_H
#define CORE_UTIL_TIMER_H
#include <chrono>
#include <string>

template <int units = 1>
class AutoTimer {
 public:
  using Time_t = std::chrono::nanoseconds;
  using Clock = std::chrono::high_resolution_clock;
  explicit AutoTimer(double *ret) : ret_(ret) { stamp_ = Clock::now(); }

  AutoTimer(double *ret, std::string const &tag_)
      : verbose(true), tag(tag_), ret_(ret) {
    stamp_ = Clock::now();
  }

  void reset() { stamp_ = Clock::now(); }

  void stop() {
    if (ret_) *ret_ += (Clock::now() - stamp_).count() / 1.0e9 * units;
    if (verbose && ret_) std::cout << tag << " " << (*ret_) << "\n";
  }

  ~AutoTimer() { stop(); }

 protected:
  bool verbose = false;
  std::string tag;
  double *ret_ = nullptr;
  Clock::time_point stamp_;
};

using MSecTimer = AutoTimer<1000>;
using USecTimer = AutoTimer<1000000>;

#endif  // CORE_UTIL_TIMER_H 
