#include "leg_tracking/core.hpp"
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/Twist.h>
#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <stdexcept>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_listener.h>
#include <visualization_msgs/Marker.h>

class Node {
  ros::NodeHandle nh_, pnh_{"~"};
  tf2_ros::Buffer tf_;
  tf2_ros::TransformListener listener_{tf_};
  ros::Subscriber sub_;
  ros::Publisher cmd_, marker_;
  ros::WallTimer timer_;
  leg::GaussianNB model_;
  leg::Tracker tracker_;
  leg::Controller controller_;
  leg::Track track_;
  bool usable_ = false, obstacle_ = true;
  ros::WallTime received_;
  std::string tracking_frame_, base_frame_;
  double last_stamp_ = -1;

public:
  Node() {
    std::string path, output;
    pnh_.param<std::string>("model", path, "");
    model_.load(path);
    pnh_.param<std::string>("tracking_frame", tracking_frame_, "odom");
    pnh_.param<std::string>("base_frame", base_frame_, "base_link");
    pnh_.param<std::string>("command_topic", output, "cmd_vel_preview");
    cmd_ = nh_.advertise<geometry_msgs::Twist>(output, 1);
    marker_ = nh_.advertise<visualization_msgs::Marker>("tracked_leg", 1);
    sub_ = nh_.subscribe("scan", 1, &Node::scan, this);
    timer_ = nh_.createWallTimer(ros::WallDuration(.05), &Node::tick, this);
  }
  void scan(const sensor_msgs::LaserScan::ConstPtr &msg) {
    double stamp = msg->header.stamp.toSec();
    if (stamp <= last_stamp_) {
      usable_ = false;
      return;
    }
    last_stamp_ = stamp;
    received_ = ros::WallTime::now();
    usable_ = false;
    try {
      leg::Scan s;
      s.stamp = stamp;
      s.angle_min = msg->angle_min;
      s.angle_increment = msg->angle_increment;
      s.range_min = msg->range_min;
      s.range_max = msg->range_max;
      s.ranges.assign(msg->ranges.begin(), msg->ranges.end());
      auto detections = leg::detect(s, model_);
      obstacle_ = leg::obstacle_near(s);
      // Lookup even for empty detections: TF availability is a control
      // precondition.
      auto transform =
          tf_.lookupTransform(tracking_frame_, msg->header.frame_id,
                              msg->header.stamp, ros::Duration(0));
      for (auto &d : detections) {
        geometry_msgs::PointStamped a, b;
        a.header = msg->header;
        a.point.x = d.position.x();
        a.point.y = d.position.y();
        tf2::doTransform(a, b, transform);
        d.position = leg::Point(b.point.x, b.point.y);
      }
      track_ = tracker_.step(stamp, detections);
      usable_ = true;
      visualization_msgs::Marker m;
      m.header.frame_id = tracking_frame_;
      m.header.stamp = msg->header.stamp;
      m.ns = "leg_track";
      m.id = 0;
      m.type = visualization_msgs::Marker::SPHERE;
      m.action = (track_.status == leg::Status::Empty ||
                  track_.status == leg::Status::Lost)
                     ? visualization_msgs::Marker::DELETE
                     : visualization_msgs::Marker::ADD;
      m.pose.orientation.w = 1;
      m.pose.position.x = track_.position.x();
      m.pose.position.y = track_.position.y();
      m.scale.x = m.scale.y = m.scale.z = .12;
      m.color.g = 1;
      m.color.a = 1;
      m.lifetime = ros::Duration(.3);
      marker_.publish(m);
    } catch (const std::exception &e) {
      ROS_WARN_THROTTLE(2, "Scan rejected: %s", e.what());
    }
  }
  void tick(const ros::WallTimerEvent &event) {
    double dt = (event.current_real - event.last_real).toSec();
    bool allowed = usable_ && (ros::WallTime::now() - received_).toSec() < .3;
    allowed = allowed && leg::safe_to_move(track_, ros::Time::now().toSec(),
                                           true, obstacle_);
    leg::Point target = leg::Point::Zero();
    if (allowed) {
      try {
        geometry_msgs::PointStamped a, b;
        a.header.frame_id = tracking_frame_;
        a.header.stamp = ros::Time::now();
        a.point.x = track_.position.x();
        a.point.y = track_.position.y();
        b = tf_.transform(a, base_frame_, ros::Duration(0));
        target = leg::Point(b.point.x, b.point.y);
      } catch (const std::exception &) {
        allowed = false;
      }
    }
    auto u = controller_.step(target, allowed, dt);
    geometry_msgs::Twist cmd;
    cmd.linear.x = u.linear;
    cmd.angular.z = u.angular;
    cmd_.publish(cmd);
  }
};
int main(int argc, char **argv) {
  ros::init(argc, argv, "leg_tracking");
  try {
    Node node;
    ros::spin();
  } catch (const std::exception &e) {
    ROS_FATAL("%s", e.what());
    return 1;
  }
  return 0;
}
