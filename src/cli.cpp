#include "leg_tracking/core.hpp"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
using namespace leg;
// Text interfaces intentionally contain no ROS dependency; see
// docs/DATA_FORMAT.md.
int main(int argc, char **argv) {
  try {
    if (argc < 3)
      throw std::invalid_argument(
          "leg_cli features|train|classify|track|scan input [output/model]");
    std::string mode = argv[1];
    std::ifstream in(argv[2]);
    if (!in)
      throw std::runtime_error("input missing");
    std::cout << std::setprecision(12);
    if (mode == "features") {
      std::string id;
      int label;
      size_t n;
      while (in >> id >> label >> n) {
        Segment s;
        for (size_t i = 0; i < n; ++i) {
          double x, y;
          if (!(in >> x >> y))
            throw std::runtime_error("truncated segment");
          s.emplace_back(x, y);
        }
        auto d = describe(s);
        if (!d.valid)
          continue;
        std::cout << id << ' ' << label;
        for (double f : d.features)
          std::cout << ' ' << f;
        std::cout << '\n';
      }
    } else if (mode == "train" || mode == "classify") {
      if (argc < 4)
        throw std::invalid_argument("model path missing");
      std::vector<Features> xs;
      std::vector<int> ys;
      std::vector<std::string> ids;
      std::string id;
      int y;
      while (in >> id >> y) {
        Features x;
        for (double &v : x)
          if (!(in >> v))
            throw std::runtime_error("truncated features");
        ids.push_back(id);
        ys.push_back(y);
        xs.push_back(x);
      }
      GaussianNB model;
      if (mode == "train") {
        model.fit(xs, ys);
        if (argc >= 5) {
          std::ifstream validation(argv[4]);
          if (!validation)
            throw std::runtime_error("validation input missing");
          std::vector<Features> vx;
          std::vector<int> vy;
          while (validation >> id >> y) {
            Features f;
            for (double &value : f)
              if (!(validation >> value))
                throw std::runtime_error("truncated validation row");
            vx.push_back(f);
            vy.push_back(y);
          }
          model.calibrate(vx, vy);
        }
        model.save(argv[3]);
        std::cout << "trained " << xs.size() << " samples; threshold "
                  << model.threshold() << "\n";
      } else {
        model.load(argv[3]);
        std::cout << "id,label,score,prediction\n";
        for (size_t i = 0; i < xs.size(); ++i) {
          double v = model.score(xs[i]);
          std::cout << ids[i] << ',' << ys[i] << ',' << v << ',' << (v > 0)
                    << '\n';
        }
      }
    } else if (mode == "track") {
      Tracker tracker;
      Controller controller;
      double t, tx, ty;
      size_t n;
      double previous = 0;
      std::cout
          << "t,true_x,true_y,estimate_x,estimate_y,status,observed,v,w,us\n";
      while (in >> t >> tx >> ty >> n) {
        std::vector<Detection> ds;
        for (size_t i = 0; i < n; ++i) {
          double x, y;
          if (!(in >> x >> y))
            throw std::runtime_error("truncated detections");
          ds.push_back({Point(x, y), 1});
        }
        auto begin = std::chrono::steady_clock::now();
        auto out = tracker.step(t, ds);
        auto cmd = controller.step(
            out.position, safe_to_move(out, t, true, false), t - previous);
        previous = t;
        double us = std::chrono::duration<double, std::micro>(
                        std::chrono::steady_clock::now() - begin)
                        .count();
        std::cout << t << ',' << tx << ',' << ty << ',' << out.position.x()
                  << ',' << out.position.y() << ',' << int(out.status) << ','
                  << out.observed << ',' << cmd.linear << ',' << cmd.angular
                  << ',' << us << '\n';
      }
    } else if (mode == "scan") {
      if (argc < 4)
        throw std::invalid_argument("model path missing");
      GaussianNB model;
      model.load(argv[3]);
      Tracker tracker;
      Scan s;
      size_t n;
      std::cout << "t,detections,x,y,status,us\n";
      while (in >> s.stamp >> s.angle_min >> s.angle_increment >> s.range_min >>
             s.range_max >> n) {
        s.ranges.resize(n);
        for (auto &r : s.ranges)
          if (!(in >> r))
            throw std::runtime_error("truncated scan");
        auto start = std::chrono::steady_clock::now();
        auto ds = detect(s, model);
        auto t = tracker.step(s.stamp, ds);
        auto us = std::chrono::duration<double, std::micro>(
                      std::chrono::steady_clock::now() - start)
                      .count();
        std::cout << s.stamp << ',' << ds.size() << ',' << t.position.x() << ','
                  << t.position.y() << ',' << int(t.status) << ',' << us
                  << '\n';
      }
    } else
      throw std::invalid_argument("unknown command");
    if (in.bad() || (!in.eof() && in.fail()))
      throw std::runtime_error("invalid input");
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
