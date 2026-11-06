#include "leg_tracking/core.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>

namespace leg {
std::vector<Sample> preprocess(const Scan &s) {
  if (!std::isfinite(s.stamp) || !std::isfinite(s.angle_min) ||
      !std::isfinite(s.angle_increment) || s.angle_increment == 0 ||
      !std::isfinite(s.range_min) || !std::isfinite(s.range_max) ||
      s.range_min < 0 || s.range_max <= s.range_min)
    throw std::invalid_argument("invalid scan metadata");
  std::vector<Sample> out;
  out.reserve(s.ranges.size());
  for (size_t i = 0; i < s.ranges.size(); ++i) {
    double r = s.ranges[i], a = s.angle_min + i * s.angle_increment;
    bool valid = std::isfinite(r) && r >= s.range_min && r <= s.range_max &&
                 std::isfinite(a);
    out.push_back(
        {valid ? Point(r * std::cos(a), r * std::sin(a)) : Point::Zero(),
         valid});
  }
  return out;
}
std::vector<Segment> segment(const std::vector<Sample> &points, double gap,
                             size_t minimum) {
  if (!std::isfinite(gap) || gap <= 0 || minimum < 3)
    throw std::invalid_argument("invalid segmentation parameters");
  std::vector<Segment> out;
  Segment current;
  auto flush = [&]() {
    if (current.size() >= minimum)
      out.push_back(std::move(current));
    current.clear();
  };
  for (const auto &p : points) {
    if (!p.valid || !p.point.allFinite()) {
      flush();
      continue;
    }
    if (!current.empty() &&
        (current.back() - p.point).squaredNorm() > gap * gap)
      flush();
    current.push_back(p.point);
  }
  flush();
  return out;
}
Description describe(const Segment &points) {
  Description d;
  const size_t n = points.size();
  if (n < 3)
    return d;
  Point mean = Point::Zero();
  double minr = 1e100, maxr = 0;
  for (const auto &p : points) {
    if (!p.allFinite())
      return d;
    mean += p;
    minr = std::min(minr, p.norm());
    maxr = std::max(maxr, p.norm());
  }
  mean /= double(n);
  d.center = mean;
  Eigen::MatrixXd a(n, 3);
  Eigen::VectorXd b(n);
  double spread = 0;
  for (size_t i = 0; i < n; ++i) {
    Point p = points[i] - mean;
    spread += p.squaredNorm();
    a.row(i) << 2 * p.x(), 2 * p.y(), 1;
    b(i) = p.squaredNorm();
  }
  auto qr = a.colPivHouseholderQr();
  double radius = 0, residual = 0;
  // Degenerate line segments remain valid non-leg candidates; bounded radius
  // sentinel.
  if (qr.rank() == 3) {
    Eigen::Vector3d c = qr.solve(b);
    double r2 = c.head<2>().squaredNorm() + c.z();
    if (r2 > 0 && std::isfinite(r2)) {
      radius = std::sqrt(r2);
      for (const auto &p : points) {
        double e = (p - mean - c.head<2>()).norm() - radius;
        residual += e * e;
      }
    }
  }
  if (radius <= 0 || radius > 100) {
    radius = 100;
    residual = spread;
  }
  double length = 0, squared = 0, curvature = 0, turn = 0;
  size_t triples = 0;
  for (size_t i = 1; i < n; ++i) {
    double v = (points[i] - points[i - 1]).norm();
    length += v;
    squared += v * v;
  }
  for (size_t i = 1; i + 1 < n; ++i) {
    Point u = points[i] - points[i - 1], v = points[i + 1] - points[i];
    double l = u.norm(), m = v.norm(),
           chord = (points[i + 1] - points[i - 1]).norm();
    if (l < 1e-9 || m < 1e-9 || chord < 1e-9)
      continue;
    double cross = u.x() * v.y() - u.y() * v.x();
    curvature += 2 * std::abs(cross) / (l * m * chord);
    turn += std::atan2(std::abs(cross), u.dot(v));
    ++triples;
  }
  d.features = {
      {double(n), std::sqrt(spread / n), maxr - minr, residual, radius, length,
       std::max(0., squared / (n - 1) - std::pow(length / (n - 1), 2)),
       triples ? curvature / triples : 0, triples ? turn / triples : 0}};
  d.valid = std::all_of(d.features.begin(), d.features.end(),
                        [](double v) { return std::isfinite(v); });
  return d;
}
Point transform(const Point &p, double x, double y, double yaw) {
  if (!p.allFinite() || !std::isfinite(x) || !std::isfinite(y) ||
      !std::isfinite(yaw))
    throw std::invalid_argument("invalid pose");
  return Point(std::cos(yaw) * p.x() - std::sin(yaw) * p.y() + x,
               std::sin(yaw) * p.x() + std::cos(yaw) * p.y() + y);
}
void GaussianNB::fit(const std::vector<Features> &xs,
                     const std::vector<int> &ys) {
  if (xs.empty() || xs.size() != ys.size())
    throw std::invalid_argument("training dimensions");
  means_ = {};
  vars_ = {};
  std::array<size_t, 2> count{{0, 0}};
  for (size_t i = 0; i < xs.size(); ++i) {
    int c = ys[i];
    if (c < 0 || c > 1)
      throw std::invalid_argument("label");
    ++count[c];
    for (size_t j = 0; j < 9; ++j) {
      if (!std::isfinite(xs[i][j]))
        throw std::invalid_argument("feature");
      if (xs[i][j] < 0)
        throw std::invalid_argument("negative feature");
      double value = std::log1p(xs[i][j]);
      double delta = value - means_[c][j];
      means_[c][j] += delta / count[c];
      vars_[c][j] += delta * (value - means_[c][j]);
    }
  }
  for (int c = 0; c < 2; ++c) {
    if (count[c] < 2)
      throw std::invalid_argument("two samples per class required");
    priors_[c] = double(count[c]) / xs.size();
    for (size_t j = 0; j < 9; ++j)
      vars_[c][j] = std::max(vars_[c][j] / count[c], 1e-9);
  }
  threshold_ = 0;
  ready_ = true;
}
double GaussianNB::score(const Features &f) const {
  if (!ready_)
    throw std::logic_error("model not loaded");
  double result[2];
  for (int c = 0; c < 2; ++c) {
    result[c] = std::log(priors_[c]);
    for (size_t j = 0; j < 9; ++j) {
      if (!std::isfinite(f[j]))
        throw std::invalid_argument("nonfinite feature");
      if (f[j] < 0)
        throw std::invalid_argument("negative feature");
      double e = std::log1p(f[j]) - means_[c][j];
      result[c] -= .5 * (std::log(2 * 3.141592653589793 * vars_[c][j]) +
                         e * e / vars_[c][j]);
    }
  }
  return result[1] - result[0] - threshold_;
}
void GaussianNB::calibrate(const std::vector<Features> &x,
                           const std::vector<int> &y) {
  if (x.empty() || x.size() != y.size())
    throw std::invalid_argument("calibration dimensions");
  std::vector<double> scores;
  size_t positives = 0;
  for (size_t i = 0; i < x.size(); ++i) {
    if (y[i] != 0 && y[i] != 1)
      throw std::invalid_argument("calibration label");
    positives += y[i];
    scores.push_back(score(x[i]) + threshold_);
  }
  if (positives == 0 || positives == x.size())
    throw std::invalid_argument("calibration needs both classes");
  auto candidates = scores;
  std::sort(candidates.begin(), candidates.end());
  candidates.insert(candidates.begin(), candidates.front() - 1);
  double best = -1, chosen = 0;
  for (double threshold : candidates) {
    size_t tp = 0, fp = 0, fn = 0;
    for (size_t i = 0; i < y.size(); ++i) {
      bool predicted = scores[i] > threshold;
      tp += predicted && y[i];
      fp += predicted && !y[i];
      fn += !predicted && y[i];
    }
    double f1 = 2. * tp / std::max(size_t(1), 2 * tp + fp + fn);
    if (f1 > best) {
      best = f1;
      chosen = threshold;
    }
  }
  threshold_ = chosen;
}
void GaussianNB::save(const std::string &path) const {
  if (!ready_)
    throw std::logic_error("untrained model");
  std::ofstream f(path);
  if (!f)
    throw std::runtime_error("cannot write model");
  f << "LEG_NB_LOG1P_V3\n" << std::setprecision(17) << threshold_ << '\n';
  for (int c = 0; c < 2; ++c) {
    f << priors_[c] << '\n';
    for (auto v : means_[c])
      f << v << ' ';
    f << '\n';
    for (auto v : vars_[c])
      f << v << ' ';
    f << '\n';
  }
}
void GaussianNB::load(const std::string &path) {
  GaussianNB tmp;
  std::ifstream f(path);
  std::string magic;
  f >> magic;
  if (magic != "LEG_NB_LOG1P_V3")
    throw std::runtime_error("invalid model version");
  f >> tmp.threshold_;
  if (!f || !std::isfinite(tmp.threshold_))
    throw std::runtime_error("invalid threshold");
  for (int c = 0; c < 2; ++c) {
    f >> tmp.priors_[c];
    for (auto &v : tmp.means_[c])
      f >> v;
    for (auto &v : tmp.vars_[c])
      f >> v;
    if (!f || !std::isfinite(tmp.priors_[c]) || tmp.priors_[c] <= 0 ||
        tmp.priors_[c] >= 1)
      throw std::runtime_error("invalid prior");
    for (size_t j = 0; j < 9; ++j)
      if (!std::isfinite(tmp.means_[c][j]) || !std::isfinite(tmp.vars_[c][j]) ||
          tmp.vars_[c][j] <= 0)
        throw std::runtime_error("invalid model parameter");
  }
  if (std::abs(tmp.priors_[0] + tmp.priors_[1] - 1) > 1e-6)
    throw std::runtime_error("prior sum");
  tmp.ready_ = true;
  *this = tmp;
}
std::vector<Detection> detect(const Scan &scan, const GaussianNB &model) {
  std::vector<Detection> out;
  for (const auto &s : segment(preprocess(scan))) {
    auto d = describe(s);
    if (d.valid) {
      double score = model.score(d.features);
      if (score > 0)
        out.push_back({d.center, score});
    }
  }
  return out;
}
} // namespace leg
