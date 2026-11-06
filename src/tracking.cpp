#include "leg_tracking/core.hpp"
#include <cmath>
#include <stdexcept>
namespace leg {
namespace {
Eigen::Matrix<double, 2, 4> observation() {
  Eigen::Matrix<double, 2, 4> h;
  h << 1, 0, 0, 0, 0, 0, 1, 0;
  return h;
}
} // namespace
void Kalman::initialize(const Point &p) {
  if (!p.allFinite())
    throw std::invalid_argument("position");
  x_ << p.x(), 0, p.y(), 0;
  p_.setIdentity();
  p_(0, 0) = p_(2, 2) = .01;
}
void Kalman::predict(double dt) {
  if (!std::isfinite(dt) || dt <= 0 || dt > 2)
    throw std::invalid_argument("dt outside (0,2]");
  Covariance a = Covariance::Identity();
  a(0, 1) = a(2, 3) = dt;
  Eigen::Matrix<double, 4, 2> g;
  g << dt * dt / 2, 0, dt, 0, 0, dt * dt / 2, 0, dt;
  x_ = a * x_;
  p_ = (a * p_ * a.transpose() + .8 * .8 * g * g.transpose()).eval();
}
double Kalman::distance(const Point &z) const {
  if (!z.allFinite())
    return INFINITY;
  auto h = observation();
  Eigen::Matrix2d s =
      h * p_ * h.transpose() + Eigen::Matrix2d::Identity() * .01;
  Point e = z - h * x_;
  return e.dot(s.ldlt().solve(e));
}
void Kalman::update(const Point &z) {
  if (!z.allFinite())
    throw std::invalid_argument("measurement");
  auto h = observation();
  Eigen::Matrix2d r = Eigen::Matrix2d::Identity() * .01;
  Eigen::Matrix2d s = h * p_ * h.transpose() + r;
  Eigen::Matrix<double, 4, 2> k =
      s.ldlt().solve((p_ * h.transpose()).transpose()).transpose();
  x_ += k * (z - h * x_);
  Covariance i = Covariance::Identity() - k * h;
  p_ = (i * p_ * i.transpose() + k * r * k.transpose()).eval();
  p_ = ((p_ + p_.transpose()) * .5).eval();
}
int associate(const Kalman &k, const std::vector<Detection> &ds, double gate) {
  int best = -1;
  double minimum = gate;
  for (size_t i = 0; i < ds.size(); ++i) {
    double d = k.distance(ds[i].position);
    if (d < minimum) {
      minimum = d;
      best = int(i);
    }
  }
  return best;
}
void Tracker::reset() {
  initialized_ = false;
  hits_ = 0;
  track_ = {};
  last_seen_ = 0;
}
Track Tracker::step(double stamp, const std::vector<Detection> &ds) {
  if (!std::isfinite(stamp))
    throw std::invalid_argument("timestamp");
  if (initialized_ && stamp <= track_.stamp) {
    auto unchanged = track_;
    unchanged.observed = false;
    return unchanged;
  }
  if (initialized_ && (stamp - last_seen_ > .6 || stamp - track_.stamp > 2)) {
    reset();
    track_.status = Status::Lost;
  }
  if (!initialized_) {
    int best = -1;
    double nearest = 3.0;
    for (size_t i = 0; i < ds.size(); ++i)
      if (ds[i].position.allFinite() && ds[i].position.norm() < nearest) {
        nearest = ds[i].position.norm();
        best = int(i);
      }
    track_.stamp = stamp;
    track_.observed = false;
    if (best < 0)
      return track_;
    k_.initialize(ds[best].position);
    initialized_ = true;
    hits_ = 1;
    last_seen_ = stamp;
    track_.position = k_.position();
    track_.status = Status::Tentative;
    track_.observed = true;
    return track_;
  }
  k_.predict(stamp - track_.stamp);
  int matched = associate(k_, ds);
  track_.observed = matched >= 0;
  if (matched >= 0) {
    k_.update(ds[matched].position);
    last_seen_ = stamp;
    ++hits_;
    track_.status = hits_ >= 3 ? Status::Tracking : Status::Tentative;
  } else {
    if (hits_ < 3)
      hits_ = 0;
    track_.status = hits_ >= 3 ? Status::Coasting : Status::Tentative;
  }
  track_.stamp = stamp;
  track_.position = k_.position();
  return track_;
}
} // namespace leg
