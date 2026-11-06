#!/usr/bin/env python3
import math
import time
import unittest
import rospy
import rostest
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from visualization_msgs.msg import Marker

class Integration(unittest.TestCase):
    def test_pipeline_and_failures(self):
        self.commands=[];self.markers=[]
        rospy.Subscriber('cmd_vel_preview',Twist,lambda m:self.commands.append(m))
        rospy.Subscriber('tracked_leg',Marker,lambda m:self.markers.append(m))
        pub=rospy.Publisher('scan',LaserScan,queue_size=1)
        time.sleep(1)
        def publish(kind,frames=14):
            for _ in range(frames):
                s=LaserScan();s.header.stamp=rospy.Time.now();s.header.frame_id='missing' if kind=='tf' else 'laser'
                s.angle_min=-math.pi;s.angle_increment=2*math.pi/720;s.angle_max=s.angle_min+719*s.angle_increment
                s.range_min=.05;s.range_max=10;s.ranges=[float('inf')]*720
                if kind!='empty':
                    for i in range(720):
                        a=s.angle_min+i*s.angle_increment
                        disc=.07**2-(1.3*math.sin(a))**2
                        if disc>=0 and math.cos(a)>0:s.ranges[i]=1.3*math.cos(a)-math.sqrt(disc)
                if kind=='obstacle':s.ranges[360]=.2
                pub.publish(s);time.sleep(.05)
        publish('valid',24)
        self.assertTrue(self.markers,'no marker published')
        self.assertTrue(any(c.linear.x>0 for c in self.commands),'valid target never enabled preview control')
        for failure in ['empty','tf','obstacle']:
            publish('valid',14);publish(failure,12)
            self.assertTrue(self.commands)
            self.assertEqual(self.commands[-1].linear.x,0,failure)
            self.assertEqual(self.commands[-1].angular.z,0,failure)
        publish('valid',14);time.sleep(.6)
        self.assertEqual(self.commands[-1].linear.x,0,'stale input did not stop')
        self.assertEqual(self.commands[-1].angular.z,0)

if __name__=='__main__':
    rospy.init_node('integration_test');rostest.rosrun('ros_leg_tracking','integration',Integration)
