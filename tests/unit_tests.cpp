#include "leg_tracking/core.hpp"
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
using namespace leg;
const double pi = 3.141592653589793;
TEST(Preprocess, Geometry) {
  Scan s;
  s.angle_min = 0;
  s.angle_increment = pi / 2;
  s.ranges = {1, 2};
  auto p = preprocess(s);
  EXPECT_NEAR(p[0].point.x(), 1, 1e-12);
  EXPECT_NEAR(p[1].point.y(), 2, 1e-12);
}
TEST(Preprocess, InvalidRanges) {
  Scan s;
  s.angle_increment = .1;
  s.ranges = {NAN, INFINITY, 0, 21, .05, 20};
  auto p = preprocess(s);
  for (int i = 0; i < 4; ++i)
    EXPECT_FALSE(p[i].valid);
  EXPECT_TRUE(p[4].valid);
  EXPECT_TRUE(p[5].valid);
}
TEST(Preprocess, VariableCount) {
  for (int n : {0, 360, 720, 1080}) {
    Scan s;
    s.angle_increment = .01;
    s.ranges.assign(n, 1);
    EXPECT_EQ(preprocess(s).size(), size_t(n));
  }
}
TEST(Preprocess, BadMetadata) {
  Scan s;
  EXPECT_THROW(preprocess(s), std::invalid_argument);
  s.angle_increment = .1;
  s.range_max = -1;
  EXPECT_THROW(preprocess(s), std::invalid_argument);
}
TEST(Segmentation, FlushAndGaps) {
  std::vector<Sample> ps;
  for (int j = 0; j < 2; ++j) {
    for (int i = 0; i < 3; ++i)
      ps.push_back({Point(j + i * .01, 0), true});
    if (j == 0)
      ps.push_back({Point::Zero(), false});
  }
  auto s = segment(ps);
  ASSERT_EQ(s.size(), 2u);
  EXPECT_EQ(s[1].size(), 3u);
}
TEST(Segmentation, EmptyAndShort) {
  EXPECT_TRUE(segment({}).empty());
  EXPECT_TRUE(segment({{Point(1, 0), true}}).empty());
  EXPECT_THROW(segment({}, 0), std::invalid_argument);
}
TEST(Segmentation, DistanceBreak) {
  std::vector<Sample> p;
  for (int j = 0; j < 2; ++j)
    for (int i = 0; i < 3; ++i)
      p.push_back({Point(j + i * .01, 0), true});
  EXPECT_EQ(segment(p).size(), 2u);
}
TEST(Features, Circle) {
  Segment s;
  for (int i = 0; i < 20; ++i)
    s.emplace_back(1 + .06 * std::cos(i * .1), 2 + .06 * std::sin(i * .1));
  auto d = describe(s);
  ASSERT_TRUE(d.valid);
  EXPECT_NEAR(d.features[4], .06, 1e-9);
  EXPECT_NEAR(d.features[3], 0, 1e-12);
  EXPECT_NEAR(d.features[7], 1 / .06, 1e-7);
}
TEST(Features, CenterNotFeatures) {
  auto d = describe({Point(2, 3), Point(2.01, 3), Point(2, 3.01)});
  EXPECT_NEAR(d.center.x(), 2 + .01 / 3, 1e-12);
  EXPECT_GT((d.center - Point(d.features[5], d.features[6])).norm(), 3);
}
TEST(Features, Degenerate) {
  for (const auto &s :
       std::vector<Segment>{{Point(0, 0), Point(0, 0), Point(0, 0)},
                            {Point(0, 0), Point(1, 0), Point(2, 0)}}) {
    auto d = describe(s);
    EXPECT_TRUE(d.valid);
    for (double f : d.features)
      EXPECT_TRUE(std::isfinite(f));
  }
}
TEST(Features, RejectInvalid) {
  EXPECT_FALSE(describe({}).valid);
  EXPECT_FALSE(describe({Point(NAN, 0), Point(0, 0), Point(1, 1)}).valid);
}
TEST(Transform, RotationTranslation) {
  auto p = transform(Point(1, 0), 2, 3, pi / 2);
  EXPECT_NEAR(p.x(), 2, 1e-12);
  EXPECT_NEAR(p.y(), 4, 1e-12);
  EXPECT_THROW(transform(Point(NAN, 0), 0, 0, 0), std::invalid_argument);
}
GaussianNB model() {
  GaussianNB m;
  Features a{}, b{};
  b.fill(10);
  m.fit({a, a, b, b}, {0, 0, 1, 1});
  return m;
}
TEST(Classifier, KnownLabels) {
  auto m = model();
  Features a{}, b{};
  b.fill(10);
  EXPECT_LT(m.score(a), 0);
  EXPECT_GT(m.score(b), 0);
}
TEST(Classifier, FiniteSmallVariance) {
  auto m = model();
  Features f;
  f.fill(1e5);
  EXPECT_TRUE(std::isfinite(m.score(f)));
}
TEST(Classifier, RoundTrip) {
  auto m = model();
  m.save("test_model.tmp");
  GaussianNB n;
  n.load("test_model.tmp");
  Features f{};
  EXPECT_DOUBLE_EQ(m.score(f), n.score(f));
  std::remove("test_model.tmp");
}
TEST(Classifier, RejectBadTraining) {
  GaussianNB m;
  Features f{};
  EXPECT_THROW(m.fit({f}, {1}), std::invalid_argument);
  EXPECT_THROW(m.fit({f, f}, {-1, 2}), std::invalid_argument);
  EXPECT_THROW(m.score(f), std::logic_error);
}
TEST(Classifier, RejectBadModel) {
  GaussianNB m;
  EXPECT_THROW(m.load("does-not-exist"), std::runtime_error);
  std::ofstream f("bad_model.tmp");
  f << "BAD";
  f.close();
  EXPECT_THROW(m.load("bad_model.tmp"), std::runtime_error);
  std::remove("bad_model.tmp");
}
TEST(Kalman, AnalyticUpdate) {
  Kalman k;
  k.initialize(Point(1, 2));
  k.update(Point(1.2, 1.8));
  EXPECT_NEAR(k.position().x(), 1.1, 1e-12);
  EXPECT_NEAR(k.position().y(), 1.9, 1e-12);
  EXPECT_NEAR(k.covariance()(0, 0), .005, 1e-12);
}
TEST(Kalman, ConstantVelocity) {
  Kalman k;
  k.initialize(Point(0, 0));
  for (int i = 1; i <= 100; ++i) {
    k.predict(.1);
    k.update(Point(i * .1, i * .05));
  }
  EXPECT_NEAR(k.state()(1), 1, .02);
  EXPECT_NEAR(k.state()(3), .5, .02);
}
TEST(Kalman, CovarianceStability) {
  Kalman k;
  k.initialize(Point(0, 0));
  for (int i = 0; i < 5000; ++i) {
    k.predict(.03 + (i % 4) * .01);
    if (i % 7)
      k.update(Point(std::sin(i * .01), std::cos(i * .01)));
    EXPECT_LT((k.covariance() - k.covariance().transpose()).norm(), 1e-10);
    Eigen::SelfAdjointEigenSolver<Kalman::Covariance> eig(k.covariance());
    ASSERT_GT(eig.eigenvalues().minCoeff(), -1e-12);
  }
}
TEST(Kalman, InvalidTime) {
  Kalman k;
  EXPECT_THROW(k.predict(0), std::invalid_argument);
  EXPECT_THROW(k.predict(-1), std::invalid_argument);
  EXPECT_THROW(k.predict(NAN), std::invalid_argument);
  EXPECT_THROW(k.update(Point(NAN, 0)), std::invalid_argument);
}
TEST(Association, GateAndNearest) {
  Kalman k;
  k.initialize(Point(1, 0));
  EXPECT_EQ(associate(k, {{Point(100, 0), 1}, {Point(1.01, 0), 1}}), 1);
  EXPECT_EQ(associate(k, {{Point(100, 0), 1}}), -1);
  EXPECT_EQ(associate(k, {}), -1);
}
TEST(Tracker, Lifecycle) {
  Tracker t;
  EXPECT_EQ(t.step(0, {}).status, Status::Empty);
  std::vector<Detection> d{{Point(1, 0), 1}};
  EXPECT_EQ(t.step(.1, d).status, Status::Tentative);
  t.step(.2, d);
  EXPECT_EQ(t.step(.3, d).status, Status::Tracking);
  EXPECT_EQ(t.step(.4, {}).status, Status::Coasting);
  EXPECT_EQ(t.step(1, {}).status, Status::Lost);
  EXPECT_EQ(t.step(1.1, d).status, Status::Tentative);
}
TEST(Tracker, DuplicateIgnored) {
  Tracker t;
  auto a = t.step(1, {{Point(1, 0), 1}});
  auto b = t.step(1, {{Point(2, 0), 1}});
  EXPECT_DOUBLE_EQ(a.position.x(), b.position.x());
  EXPECT_FALSE(b.observed);
  EXPECT_DOUBLE_EQ(t.step(.5, {}).stamp, 1);
}
TEST(Tracker, ConfirmationMustBeConsecutive) {
  Tracker t;
  std::vector<Detection> d{{Point(1, 0), 1}};
  t.step(0, d);
  t.step(.1, {});
  t.step(.2, d);
  EXPECT_EQ(t.step(.3, d).status, Status::Tentative);
  EXPECT_EQ(t.step(.4, d).status, Status::Tracking);
}
TEST(Controller, DeadbandAndLimits) {
  Controller c;
  EXPECT_DOUBLE_EQ(c.step(Point(.8, 0), true, .1).linear, 0);
  for (int i = 0; i < 100; ++i) {
    auto u = c.step(Point(3, 0), true, .1);
    EXPECT_LE(u.linear, .25);
    EXPECT_LE(std::abs(u.angular), .8);
  }
}
TEST(Controller, AccelerationAndTurn) {
  Controller c;
  auto a = c.step(Point(3, 0), true, .1);
  EXPECT_NEAR(a.linear, .05, 1e-12);
  c.reset();
  EXPECT_DOUBLE_EQ(c.step(Point(0, 3), true, .1).linear, 0);
}
TEST(Controller, ImmediateStop) {
  Controller c;
  c.step(Point(2, 0), true, .1);
  auto u = c.step(Point(2, 0), false, .1);
  EXPECT_DOUBLE_EQ(u.linear, 0);
  EXPECT_DOUBLE_EQ(u.angular, 0);
  EXPECT_DOUBLE_EQ(c.step(Point(.1, .1), true, .1).angular, 0);
}
TEST(Safety, StaleLostAndTF) {
  Track t;
  t.status = Status::Tracking;
  t.observed = true;
  t.stamp = 1;
  EXPECT_TRUE(safe_to_move(t, 1.1, true, false));
  EXPECT_FALSE(safe_to_move(t, 1.4, true, false));
  EXPECT_FALSE(safe_to_move(t, .9, true, false));
  EXPECT_FALSE(safe_to_move(t, 1.1, false, false));
  EXPECT_FALSE(safe_to_move(t, 1.1, true, true));
  t.status = Status::Coasting;
  EXPECT_FALSE(safe_to_move(t, 1.1, true, false));
}
TEST(Safety, Clearance) {
  Scan s;
  s.ranges = {NAN, INFINITY, 1};
  EXPECT_FALSE(obstacle_near(s));
  s.ranges.push_back(.2);
  EXPECT_TRUE(obstacle_near(s));
}
TEST(Pipeline, EmptyScan) {
  auto m = model();
  Scan s;
  s.angle_increment = .1;
  EXPECT_TRUE(detect(s, m).empty());
  s.ranges.assign(720, INFINITY);
  EXPECT_TRUE(detect(s, m).empty());
}
TEST(Classifier, CalibrationAndPersistence) {
  auto m = model();
  Features a{}, b{};
  b.fill(10);
  m.calibrate({a, a, b, b}, {0, 0, 1, 1});
  EXPECT_LT(m.score(a), 1e-8);
  EXPECT_GT(m.score(b), 0);
  m.save("calibration.tmp");
  GaussianNB n;
  n.load("calibration.tmp");
  EXPECT_DOUBLE_EQ(m.threshold(), n.threshold());
  std::remove("calibration.tmp");
  EXPECT_THROW(m.calibrate({a}, {0}), std::invalid_argument);
}
