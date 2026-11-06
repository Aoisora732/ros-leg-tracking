#include "leg_tracking/core.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>
int main() {
  std::vector<leg::Segment> segments;
  for (int n = 8; n <= 64; n += 4) {
    leg::Segment s;
    for (int j = 0; j < n; ++j) {
      double a = .03 * j;
      s.emplace_back(1 + .06 * std::cos(a), .06 * std::sin(a));
    }
    segments.push_back(s);
  }
  volatile double sink = 0;
  auto run = [&](bool duplicate) {
    auto start = std::chrono::steady_clock::now();
    double checksum = 0;
    for (int repeat = 0; repeat < 200; ++repeat)
      for (const auto &s : segments) {
        auto first = leg::describe(s);
        auto second = duplicate ? leg::describe(s) : first;
        checksum += first.features[4] + second.features[4];
      }
    sink += checksum;
    return std::make_pair(std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - start)
                              .count(),
                          checksum);
  };
  run(false);
  run(true);
  std::vector<double> before, after;
  for (int i = 0; i < 21; ++i) {
    auto a = run(i % 2 == 0), b = run(i % 2 != 0);
    if (std::abs(a.second - b.second) > 1e-10)
      return 1;
    (i % 2 == 0 ? before : after).push_back(a.first);
    (i % 2 == 0 ? after : before).push_back(b.first);
  }
  std::sort(before.begin(), before.end());
  std::sort(after.begin(), after.end());
  std::cout << "{\"scope\":\"equivalent feature computation twice vs cached "
               "once; not original "
               "binary\",\"segments_per_round\":3000,\"rounds\":21,\"duplicate_"
               "median_ms\":"
            << before[10] << ",\"cached_median_ms\":" << after[10]
            << ",\"reduction_percent\":" << 100 * (1 - after[10] / before[10])
            << ",\"checksum\":" << sink << "}\n";
}
