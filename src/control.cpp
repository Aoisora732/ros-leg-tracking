#include "leg_tracking/core.hpp"
#include <algorithm>
#include <cmath>
namespace leg {
namespace {
double clamp(double x, double a, double b) {
  return std::max(a, std::min(b, x));
}
} // namespace
Command Controller::step(const Point &p, bool allowed, double dt) {
  if (!allowed || !p.allFinite() || !std::isfinite(dt) || dt <= 0 || dt > .5) {
    reset();
    return last_;
  }
  double distance = p.norm(), angle = std::atan2(p.y(), p.x());
  if (distance < .3) {
    reset();
    return last_;
  }
  double e = distance - .8;
  double v = std::abs(e) < .05 ? 0 : clamp(.5 * e, 0, .25);
  if (std::abs(angle) > .6)
    v = 0;
  double w = std::abs(angle) < .03 ? 0 : clamp(1.2 * angle, -.8, .8);
  last_.linear += clamp(v - last_.linear, -.5 * dt, .5 * dt);
  last_.angular += clamp(w - last_.angular, -1.5 * dt, 1.5 * dt);
  return last_;
}
bool safe_to_move(const Track &t, double now, bool tf, bool obstacle,
                  double max_age) {
  return tf && !obstacle && t.status == Status::Tracking && t.observed &&
         t.position.allFinite() && std::isfinite(now) &&
         std::isfinite(t.stamp) && now >= t.stamp && now - t.stamp <= max_age;
}
bool obstacle_near(const Scan &scan, double threshold) {
  // Conservative all-direction clearance; a full collision planner is outside
  // scope.
  for (double r : scan.ranges)
    if (std::isfinite(r) && r >= scan.range_min && r <= scan.range_max &&
        r < threshold)
      return true;
  return false;
}
} // namespace leg
