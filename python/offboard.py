from __future__ import division

import rospy
import math
import numpy as np
from geometry_msgs.msg import PoseStamped, Quaternion, TwistStamped
from mavros_msgs.msg import ParamValue, PositionTarget
from mavros_test_common import MavrosTestCommon
from pymavlink import mavutil
from six.moves import xrange
from std_msgs.msg import Header
from threading import Thread
from tf.transformations import quaternion_from_euler
from pynput import keyboard

from sensor_msgs.msg import LaserScan

from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.utils import get_device
import torch


class Modes:
    POSITION_CONTROL = 0
    VELOCITY_CONTROL = 1
    RL = 2


class MavrosOffboardPosctlTest(MavrosTestCommon):

    def setUp(self):
        super(MavrosOffboardPosctlTest, self).setUp()

        self.pos = PoseStamped()

        self.vel_global = TwistStamped()

        self.mode = Modes.POSITION_CONTROL

        self.vel_local = PositionTarget()

        self.pos_setpoint_pub = rospy.Publisher(
            '/mavros/setpoint_position/local', PoseStamped, queue_size=1)

        self.vel_local_pub = rospy.Publisher(
            '/mavros/setpoint_raw/local', PositionTarget, queue_size=1)  # prawdopodobnie da sie wszystkim sterowac

        # send setpoints in seperate thread to better prevent failsafe
        self.drone_control_thread = Thread(target=self.control_drone, args=())
        self.drone_control_thread.daemon = True
        self.drone_control_thread.start()

        self.v_x = 0
        self.v_y = 0
        self.v_z = 0
        self.v_yaw = 0
        self.speed = 0.5

        sub = rospy.Subscriber('/laser/scan', LaserScan, self.laser_callback)

        self.laser_max_range = 10
        self.laser_min_range = 0.15
        self.laser_resolution = 360

        self.laser_ranges = np.full(self.laser_resolution, self.laser_max_range, dtype=np.float32)
        self.laser_data = np.full(self.laser_resolution, self.laser_max_range, dtype=np.float32)

        device = get_device("auto")
        saved_variables = torch.load("ppo8_policy", map_location=device)

        self.model = MlpPolicy(**saved_variables["data"])
        self.model.load_state_dict(saved_variables["state_dict"], strict=False)
        self.model.to(device)

        self.yaw_i = 0
        self.yaw_p = 1
        self.yaw_p_i = 0.002

        self.pos_z_i = 0
        self.pos_z_p = 0.7
        self.pos_z_p_i = 0.002

        self.desired_heigth = 2.2
        self.desired_yaw = 0.0

    def tearDown(self):
        super(MavrosOffboardPosctlTest, self).tearDown()

    def preprocess_lasers(self):
        data = self.laser_data

        mask = np.isinf(data)
        data[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), data[~mask])

        data = np.maximum(data, self.laser_min_range)
        data = np.minimum(data, self.laser_max_range)
        self.laser_ranges = data

    def normalize_lasers(self, laser_ranges):
        divider = self.laser_max_range / 2
        laser_ranges = ((laser_ranges.copy() - divider) / divider).astype(np.float32)
        return laser_ranges

    def laser_callback(self, msg):
        self.laser_data = np.array(msg.ranges).astype(np.float32)

    def euler_from_quaternion(self, x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return roll_x, pitch_y, yaw_z

    #
    # Helper methods
    #
    def control_drone(self):
        rate = rospy.Rate(30)  # Hz
        self.pos.header = Header()
        self.pos.header.frame_id = "base_footprint"

        self.vel_local.header = Header()
        self.vel_local.header.frame_id = "base_footprint"

        self.vel_global.header = Header()
        self.vel_global.header.frame_id = "base_footprint"

        while not rospy.is_shutdown():

            roll, pitch, yaw = self.euler_from_quaternion(self.local_position.pose.orientation.x,
                                                          self.local_position.pose.orientation.y,
                                                          self.local_position.pose.orientation.z,
                                                          self.local_position.pose.orientation.w)

            rospy.loginfo(str(roll) + " " + str(pitch) + " " + str(yaw))

            if self.mode == Modes.POSITION_CONTROL:
                self.pos.header.stamp = rospy.Time.now()
                self.pos_setpoint_pub.publish(self.pos)
            elif self.mode == Modes.VELOCITY_CONTROL:
                self.vel_local.header.stamp = rospy.Time.now()
                self.vel_local_pub.publish(self.vel_local)
            elif self.mode == Modes.RL:
                self.preprocess_lasers()

                obs = self.normalize_lasers(self.laser_ranges)

                action, _states = self.model.predict(obs, deterministic=True)

                # rospy.loginfo(str(action))

                e = (self.desired_yaw - self.local_position.pose.orientation.z)
                p = e * self.yaw_p
                self.yaw_i += self.yaw_p_i * e * (1 / 30)

                o = p + self.yaw_i

                e_z = self.desired_heigth - self.local_position.pose.position.z

                p_z = e_z * self.pos_z_p
                self.pos_z_i += self.pos_z_p_i * e_z * (1 / 30)

                o_z = p_z + self.pos_z_i

                self.set_velocity(action[0], -action[1], o_z, -o)
                self.vel_local.header.stamp = rospy.Time.now()
                self.vel_local_pub.publish(self.vel_local)

            try:  # prevent garbage in console output when thread is killed
                rate.sleep()
            except rospy.ROSInterruptException:
                pass

    def is_at_position(self, x, y, z, offset):
        """offset: meters"""
        rospy.logdebug(
            "current position | x:{0:.2f}, y:{1:.2f}, z:{2:.2f}".format(
                self.local_position.pose.position.x, self.local_position.pose.
                position.y, self.local_position.pose.position.z))

        desired = np.array((x, y, z))
        pos = np.array((self.local_position.pose.position.x,
                        self.local_position.pose.position.y,
                        self.local_position.pose.position.z))
        return np.linalg.norm(desired - pos) < offset

    def reach_position(self, x, y, z, timeout):
        """timeout(int): seconds"""
        # set a position setpoint
        self.pos.pose.position.x = x
        self.pos.pose.position.y = y
        self.pos.pose.position.z = z
        rospy.loginfo(
            "attempting to reach position | x: {0}, y: {1}, z: {2} | current position x: {3:.2f}, y: {4:.2f}, z: {5:.2f}".
            format(x, y, z, self.local_position.pose.position.x,
                   self.local_position.pose.position.y,
                   self.local_position.pose.position.z))

        # For demo purposes we will lock yaw/heading to north.
        yaw_degrees = 180  # North
        yaw = math.radians(yaw_degrees)
        quaternion = quaternion_from_euler(0, 0, yaw)
        self.pos.pose.orientation = Quaternion(*quaternion)

        # does it reach the position in 'timeout' seconds?
        loop_freq = 2  # Hz
        rate = rospy.Rate(loop_freq)
        reached = False
        for i in xrange(timeout * loop_freq):
            if self.is_at_position(self.pos.pose.position.x,
                                   self.pos.pose.position.y,
                                   self.pos.pose.position.z, 0.5):
                rospy.loginfo("position reached | seconds: {0} of {1}".format(
                    i / loop_freq, timeout))
                reached = True
                break

            try:
                rate.sleep()
            except rospy.ROSException as e:
                self.fail(e)

        self.assertTrue(reached, (
            "took too long to get to position | current position x: {0:.2f}, y: {1:.2f}, z: {2:.2f} | timeout(seconds): {3}".
            format(self.local_position.pose.position.x,
                   self.local_position.pose.position.y,
                   self.local_position.pose.position.z, timeout)))

    def take_off(self, z, azimuth, timeout, max_error):
        self.set_position(self.local_position.pose.position.x, self.local_position.pose.position.y, z, azimuth, timeout,
                          max_error)

    def set_velocity(self, v_x, v_y, v_z, v_yaw):

        self.vel_local.velocity.x = v_x
        self.vel_local.velocity.y = v_y
        self.vel_local.velocity.z = v_z

        self.vel_local.coordinate_frame = PositionTarget.FRAME_BODY_NED
        self.vel_local.yaw_rate = v_yaw

    def set_position(self, x, y, z, azimuth, timeout, max_error):
        self.mode = Modes.POSITION_CONTROL
        self.pos.pose.position.x = x
        self.pos.pose.position.y = y
        self.pos.pose.position.z = z

        yaw = math.radians(azimuth)
        quaternion = quaternion_from_euler(0, 0, yaw)
        self.pos.pose.orientation = Quaternion(*quaternion)

        loop_freq = 2  # Hz
        rate = rospy.Rate(loop_freq)
        reached = False
        for i in xrange(timeout * loop_freq):
            if self.is_at_position(self.pos.pose.position.x,
                                   self.pos.pose.position.y,
                                   self.pos.pose.position.z, max_error):
                rospy.loginfo("position reached | seconds: {0} of {1}".format(
                    i / loop_freq, timeout))
                reached = True
                break

            try:
                rate.sleep()
            except rospy.ROSException as e:
                self.fail(e)

        self.assertTrue(reached, (
            "took too long to get to position | current position x: {0:.2f}, y: {1:.2f}, z: {2:.2f} | timeout(seconds): {3}".
            format(self.local_position.pose.position.x,
                   self.local_position.pose.position.y,
                   self.local_position.pose.position.z, timeout)))
        return reached

    def on_release(self, key):

        if key == 'q':
            return True
        elif key == '1':
            self.set_arm(True, 5)
        elif key == '2':
            self.set_arm(False, 5)
        elif key == '3':
            self.set_arm(True, 5)
            rospy.loginfo("Start_RL")
            self.take_off(2.5, 0, 20, 0.5)
            self.mode = Modes.RL
        if self.state.armed:
            if key == 't':
                self.take_off(2.5, 0, 20, 0.5)
            elif key == 'p':
                self.rtl()
            elif key == 'l':
                self.land()
        return False

    def land(self):
        self.set_mode("AUTO.LAND", 5)

    def rtl(self):
        self.set_mode("AUTO.RTL", 5)

    def test_posctl(self):
        """Test offboard position control"""

        # make sure the simulation is ready to start the mission
        self.wait_for_topics(60)
        self.wait_for_landed_state(mavutil.mavlink.MAV_LANDED_STATE_ON_GROUND,
                                   10, -1)
        self.log_topic_vars()
        # exempting failsafe from lost RC to allow offboard
        rcl_except = ParamValue(1 << 2, 0.0)
        self.set_param("COM_RCL_EXCEPT", rcl_except, 5)
        self.set_mode("OFFBOARD", 5)

        rospy.loginfo("run mission")

        while True:
            key = input("Set new command!\n")
            ret = self.on_release(key)
            if ret:
                break


if __name__ == '__main__':
    import rostest

    rospy.init_node('test_node', anonymous=True)

    rostest.rosrun("navigation", 'mavros_offboard_posctl_test',
                   MavrosOffboardPosctlTest)
