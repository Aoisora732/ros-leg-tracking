#pragma once
#include <Eigen/Dense>
#include <array>
#include <string>
#include <vector>

namespace leg {
using Point = Eigen::Vector2d;
using Features = std::array<double, 9>;
struct Scan {
  double stamp = 0, angle_min = 0, angle_increment = 0, range_min = .05,
         range_max = 20;
  std::vector<double> ranges;
};
struct Sample {
  Point point;
  bool valid = false;
};
using Segment = std::vector<Point>;
std::vector<Sample> preprocess(const Scan &scan);
std::vector<Segment> segment(const std::vector<Sample> &points,
                             double gap = .09, size_t minimum = 3);
struct Description {
  Features features{};
  Point center = Point::Zero();
  bool valid = false;
};
Description describe(const Segment &points);
Point transform(const Point &point, double x, double y, double yaw);

class GaussianNB {
public:
  void fit(const std::vector<Features> &x, const std::vector<int> &y);
  double score(const Features &x) const; // log likelihood ratio minus threshold
  void calibrate(const std::vector<Features> &x, const std::vector<int> &y);
  double threshold() const { return threshold_; }
  void save(const std::string &path) const;
  void load(const std::string &path);

private:
  std::array<Features, 2> means_{}, vars_{};
  std::array<double, 2> priors_{};
  bool ready_ = false;
  double threshold_ = 0;
};
struct Detection {
  Point position;
  double score;
};
std::vector<Detection> detect(const Scan &scan, const GaussianNB &model);

class Kalman {
public:
  using State = Eigen::Matrix<double, 4, 1>;
  using Covariance = Eigen::Matrix<double, 4, 4>;
  void initialize(const Point &p);
  void predict(double dt);
  void update(const Point &z);
  double distance(const Point &z) const;
  Point position() const { return Point(x_(0), x_(2)); }
  const State &state() const { return x_; }
  const Covariance &covariance() const { return p_; }

private:
  State x_ = State::Zero();
  Covariance p_ = Covariance::Identity();
};
int associate(const Kalman &k, const std::vector<Detection> &detections,
              double gate = 9.21);
enum class Status { Empty, Tentative, Tracking, Coasting, Lost };
struct Track {
  Point position = Point::Zero();
  Status status = Status::Empty;
  double stamp = 0;
  bool observed = false;
};
class Tracker {
public:
  Track step(double stamp, const std::vector<Detection> &detections);
  void reset();
  const Kalman &filter() const { return k_; }

private:
  Kalman k_;
  Track track_;
  bool initialized_ = false;
  int hits_ = 0;
  double last_seen_ = 0;
};
struct Command {
  double linear = 0, angular = 0;
};
class Controller {
public:
  Command step(const Point &target, bool allowed, double dt);
  void reset() { last_ = {}; }

private:
  Command last_;
};
bool safe_to_move(const Track &track, double now, bool transform_ok,
                  bool obstacle, double max_age = .3);
bool obstacle_near(const Scan &scan, double stop_distance = .3);
} // namespace leg
