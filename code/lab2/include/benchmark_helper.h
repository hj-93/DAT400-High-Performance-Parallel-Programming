#include <chrono>

class AutoProfiler {
 public:
  AutoProfiler()
    : m_start(std::chrono::high_resolution_clock::now())
  {
  }

  ~AutoProfiler()
  {
    auto end = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - m_start);
    totalExecTime = totalExecTime + dur;
  }

  static std::chrono::microseconds totalExecTime;

 private:
  std::chrono::time_point<std::chrono::high_resolution_clock> m_start;
};

