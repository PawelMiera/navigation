#!/usr/bin/env python2mavros_test_common
from __future__ import division

import math
import unittest
from threading import Thread

import numpy as np
from pymavlink import mavutil
from six.moves import xrange

import rospy
from geometry_msgs.msg import PoseStamped, Quaternion
from mavros_msgs.msg import ExtendedState, State
from mavros_msgs.msg import ParamValue, PositionTarget
from mavros_msgs.srv import CommandBool, ParamGet, ParamSet, SetMode, WaypointClear, \
    WaypointPush

from std_msgs.msg import Header
from tf.transformations import quaternion_from_euler
from std_msgs.msg import Float32MultiArray


class Modes:
    POSITION_CONTROL = 0
    VELOCITY_CONTROL = 1
    RL = 2


class RL_Fly(unittest.TestCase):
    def __init__(self, *args):
        super(RL_Fly, self).__init__(*args)

    def setUp(self):
        self.extended_state = ExtendedState()
        # self.imu_data = Imu()
        # self.home_position = HomePosition()
        # self.local_position = PoseStamped()
        # self.local_velocity = TwistStamped()
        # self.local_acceleration = AccelStamped()
        # self.mission_wp = WaypointList()
        self.state = State()
        self.local_position = PoseStamped()
        self.mav_type = None

        self.sub_topics_ready = {
            key: False
            for key in [
                'ext_state', 'state', 'local_pos', 'rl_control'
            ]
        }

        # ROS services
        service_timeout = 30
        rospy.loginfo("waiting for ROS services")
        try:
            rospy.wait_for_service('mavros/param/get', service_timeout)
            rospy.wait_for_service('mavros/param/set', service_timeout)
            rospy.wait_for_service('mavros/cmd/arming', service_timeout)
            rospy.wait_for_service('mavros/mission/push', service_timeout)
            rospy.wait_for_service('mavros/mission/clear', service_timeout)
            rospy.wait_for_service('mavros/set_mode', service_timeout)
            rospy.loginfo("ROS services are up")
        except rospy.ROSException:
            self.fail("failed to connect to services")
        self.get_param_srv = rospy.ServiceProxy('mavros/param/get', ParamGet)
        self.set_param_srv = rospy.ServiceProxy('mavros/param/set', ParamSet)
        self.set_arming_srv = rospy.ServiceProxy('mavros/cmd/arming',
                                                 CommandBool)
        self.set_mode_srv = rospy.ServiceProxy('mavros/set_mode', SetMode)
        self.wp_clear_srv = rospy.ServiceProxy('mavros/mission/clear',
                                               WaypointClear)
        self.wp_push_srv = rospy.ServiceProxy('mavros/mission/push',
                                              WaypointPush)

        # ROS subscribers

        self.ext_state_sub = rospy.Subscriber('mavros/extended_state',
                                              ExtendedState,
                                              self.extended_state_callback)

        self.local_pos_sub = rospy.Subscriber('mavros/local_position/pose',
                                              PoseStamped,
                                              self.local_position_callback)

        self.state_sub = rospy.Subscriber('mavros/state', State,
                                          self.state_callback)

        self.rl_control_sub = rospy.Subscriber('/rl_control', Float32MultiArray, self.rl_control_callback)

        self.pos = PoseStamped()
        self.mode = Modes.VELOCITY_CONTROL

        self.vel_local = PositionTarget()

        self.set_velocity(0, 0, -1, 0)

        self.action = np.array([0, 0])

        self.vel_local.type_mask = PositionTarget.IGNORE_YAW

        self.last_pos_x = 0
        self.pos_no_change_count = 0
        self.last_rl_control_time = rospy.get_rostime().secs

        self.pos_setpoint_pub = rospy.Publisher(
            '/mavros/setpoint_position/local', PoseStamped, queue_size=1)

        self.vel_local_pub = rospy.Publisher(
            '/mavros/setpoint_raw/local', PositionTarget, queue_size=1)

        self.drone_control_thread = Thread(target=self.control_drone, args=())
        self.drone_control_thread.daemon = True
        self.drone_control_thread.start()

    def rl_control_callback(self, data):
        self.action = np.array(data.data).astype(np.float32)

        self.last_rl_control_time = rospy.get_rostime().secs
        if not self.sub_topics_ready['rl_control']:
            self.sub_topics_ready['rl_control'] = True

    def yaw_to_euler(self, x, y, z, w):

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return yaw_z

    def control_drone(self):
        my_rate = 30
        rate = rospy.Rate(my_rate)  # Hz
        self.pos.header = Header()
        self.pos.header.frame_id = "base_footprint"

        self.vel_local.header = Header()
        self.vel_local.header.frame_id = "base_footprint"

        i = 0

        while not rospy.is_shutdown():
            try:

                if self.local_position.pose.position.x == self.last_pos_x:
                    self.pos_no_change_count += 1
                else:
                    self.pos_no_change_count = 0

                if self.pos_no_change_count > 30:
                    self.land()
                    rospy.loginfo("Position estimate error, landing!")
                    continue
                elif rospy.get_rostime().secs - self.last_rl_control_time > 2:
                    self.land()
                    rospy.loginfo("Control time error, landing!")
                    continue

                self.last_pos_x = self.local_position.pose.position.x

                yaw = self.yaw_to_euler(self.local_position.pose.orientation.x,
                                        self.local_position.pose.orientation.y,
                                        self.local_position.pose.orientation.z,
                                        self.local_position.pose.orientation.w)
                i += 1
                if i % 30 == 0:
                    rospy.loginfo("x: " + str(round(self.local_position.pose.position.x, 2)) +
                                  " y: " + str(round(self.local_position.pose.position.y, 2)) + " z: "
                                  + str(round(self.local_position.pose.position.z, 2)) + " yaw: " + str(round(yaw, 2)) +
                                  " rx: " + str(round(self.vel_local.velocity.x, 2)) + " ry: " + str(
                        round(self.vel_local.velocity.y, 2)) +
                                  " rz: " + str(round(self.vel_local.velocity.z, 2)) + " y_rate: " + str(
                        round(self.vel_local.yaw_rate, 2)))

                if self.mode == Modes.POSITION_CONTROL:
                    self.pos.header.stamp = rospy.Time.now()
                    self.pos_setpoint_pub.publish(self.pos)
                elif self.mode == Modes.VELOCITY_CONTROL:
                    self.vel_local.header.stamp = rospy.Time.now()
                    self.vel_local_pub.publish(self.vel_local)

                elif self.mode == Modes.RL:

                    self.set_velocity(self.action[0], -self.action[1], 0, 0)
                    self.vel_local.header.stamp = rospy.Time.now()
                    self.vel_local_pub.publish(self.vel_local)
            except Exception as e:
                rospy.loginfo("LOOP error: " + str(e))

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
            "attempting to reach position | x: {0}, y: {1}, z: {2} | current position x: {3:.2f}, y: {4:.2f}, "
            "z: {5:.2f}".
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
            "took too long to get to position | current position x: {0:.2f}, y: {1:.2f}, z: {2:.2f} | timeout("
            "seconds): {3}".
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
            "took too long to get to position | current position x: {0:.2f}, y: {1:.2f}, z: {2:.2f} | timeout("
            "seconds): {3}".
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
            self.mode = Modes.RL
        if self.state.armed:
            if key == 't':
                self.take_off(2, 0, 20, 0.5)
            elif key == 'p':
                self.rtl()
            elif key == 'l':
                self.land()
            elif key == 'w':
                self.mode = Modes.VELOCITY_CONTROL
                self.set_velocity(0.3, 0, 0, 0)
            elif key == 's':
                self.mode = Modes.VELOCITY_CONTROL
                self.set_velocity(-0.3, 0, 0, 0)
            elif key == 'a':
                self.mode = Modes.VELOCITY_CONTROL
                self.set_velocity(0, 0.3, 0, 0)
            elif key == 'd':
                self.mode = Modes.VELOCITY_CONTROL
                self.set_velocity(0, -0.3, 0, 0)
            elif key == 'z':
                self.mode = Modes.VELOCITY_CONTROL
                self.set_velocity(0, 0, -0.7, 0)
            elif key == 'x':
                self.mode = Modes.VELOCITY_CONTROL
                self.set_velocity(0, 0, 0.9, 0)
            elif key == 'e':
                self.mode = Modes.VELOCITY_CONTROL
                self.set_velocity(0, 0, 0, 0.5)
            elif key == 'f':
                self.mode = Modes.VELOCITY_CONTROL
                self.set_velocity(0, 0, 0, 0)
            elif key == 'y':
                self.mode = Modes.VELOCITY_CONTROL
                self.test_yaw = True

        return False

    def land(self):
        self.set_mode("AUTO.LAND", 5)

    def rtl(self):
        self.set_mode("AUTO.RTL", 5)

    def test_posctl(self):
        """Test offboard position control"""
        rospy.loginfo("RL example code is starting...")
        # make sure the simulation is ready to start the mission
        self.wait_for_topics(10)
        self.wait_for_landed_state(mavutil.mavlink.MAV_LANDED_STATE_ON_GROUND,
                                   10, -1)
        self.log_topic_vars()
        # exempting failsafe from lost RC to allow offboard
        rcl_except = ParamValue(1 << 2, 0.0)
        self.set_param("COM_RCL_EXCEPT", rcl_except, 5)
        self.set_mode("OFFBOARD", 5)

        rospy.loginfo("run mission")

        while True:
            key = raw_input("Set new command!\n")
            ret = self.on_release(key)
            if ret:
                break

    def tearDown(self):
        self.log_topic_vars()

    def extended_state_callback(self, data):
        if self.extended_state.vtol_state != data.vtol_state:
            rospy.loginfo("VTOL state changed from {0} to {1}".format(
                mavutil.mavlink.enums['MAV_VTOL_STATE']
                [self.extended_state.vtol_state].name, mavutil.mavlink.enums[
                    'MAV_VTOL_STATE'][data.vtol_state].name))

        if self.extended_state.landed_state != data.landed_state:
            rospy.loginfo("landed state changed from {0} to {1}".format(
                mavutil.mavlink.enums['MAV_LANDED_STATE']
                [self.extended_state.landed_state].name, mavutil.mavlink.enums[
                    'MAV_LANDED_STATE'][data.landed_state].name))

        self.extended_state = data

        if not self.sub_topics_ready['ext_state']:
            self.sub_topics_ready['ext_state'] = True

    def local_position_callback(self, data):
        self.local_position = data

        if not self.sub_topics_ready['local_pos']:
            self.sub_topics_ready['local_pos'] = True

    def odometry_callback(self, data):
        self.odometry = data

        if not self.sub_topics_ready['odometry']:
            self.sub_topics_ready['odometry'] = True

    def state_callback(self, data):
        if self.state.armed != data.armed:
            rospy.loginfo("armed state changed from {0} to {1}".format(
                self.state.armed, data.armed))

        if self.state.connected != data.connected:
            rospy.loginfo("connected changed from {0} to {1}".format(
                self.state.connected, data.connected))

        if self.state.mode != data.mode:
            rospy.loginfo("mode changed from {0} to {1}".format(
                self.state.mode, data.mode))

        if self.state.system_status != data.system_status:
            rospy.loginfo("system_status changed from {0} to {1}".format(
                mavutil.mavlink.enums['MAV_STATE'][
                    self.state.system_status].name, mavutil.mavlink.enums[
                    'MAV_STATE'][data.system_status].name))

        self.state = data

        # mavros publishes a disconnected state message on init
        if not self.sub_topics_ready['state'] and data.connected:
            self.sub_topics_ready['state'] = True

    #
    # Helper methods
    #
    def set_arm(self, arm, timeout):
        """arm: True to arm or False to disarm, timeout(int): seconds"""
        rospy.loginfo("setting FCU arm: {0}".format(arm))
        old_arm = self.state.armed
        loop_freq = 1  # Hz
        rate = rospy.Rate(loop_freq)
        arm_set = False
        for i in xrange(timeout * loop_freq):
            if self.state.armed == arm:
                arm_set = True
                rospy.loginfo("set arm success | seconds: {0} of {1}".format(
                    i / loop_freq, timeout))
                break
            else:
                try:
                    res = self.set_arming_srv(arm)
                    if not res.success:
                        rospy.logerr("failed to send arm command")
                except rospy.ServiceException as e:
                    rospy.logerr(e)

            try:
                rate.sleep()
            except rospy.ROSException as e:
                self.fail(e)

        self.assertTrue(arm_set, (
            "failed to set arm | new arm: {0}, old arm: {1} | timeout(seconds): {2}".
                format(arm, old_arm, timeout)))

    def set_mode(self, mode, timeout):
        """mode: PX4 mode string, timeout(int): seconds"""
        rospy.loginfo("setting FCU mode: {0}".format(mode))
        old_mode = self.state.mode
        loop_freq = 1  # Hz
        rate = rospy.Rate(loop_freq)
        mode_set = False
        for i in xrange(timeout * loop_freq):
            if self.state.mode == mode:
                mode_set = True
                rospy.loginfo("set mode success | seconds: {0} of {1}".format(
                    i / loop_freq, timeout))
                break
            else:
                try:
                    res = self.set_mode_srv(0, mode)  # 0 is custom mode
                    if not res.mode_sent:
                        rospy.logerr("failed to send mode command")
                except rospy.ServiceException as e:
                    rospy.logerr(e)

            try:
                rate.sleep()
            except rospy.ROSException as e:
                self.fail(e)

        self.assertTrue(mode_set, (
            "failed to set mode | new mode: {0}, old mode: {1} | timeout(seconds): {2}".
                format(mode, old_mode, timeout)))

    def set_param(self, param_id, param_value, timeout):
        """param: PX4 param string, ParamValue, timeout(int): seconds"""
        if param_value.integer != 0:
            value = param_value.integer
        else:
            value = param_value.real
        rospy.loginfo("setting PX4 parameter: {0} with value {1}".
                      format(param_id, value))
        loop_freq = 1  # Hz
        rate = rospy.Rate(loop_freq)
        param_set = False
        for i in xrange(timeout * loop_freq):
            try:
                res = self.set_param_srv(param_id, param_value)
                if res.success:
                    rospy.loginfo("param {0} set to {1} | seconds: {2} of {3}".
                                  format(param_id, value, i / loop_freq, timeout))
                break
            except rospy.ServiceException as e:
                rospy.logerr(e)

            try:
                rate.sleep()
            except rospy.ROSException as e:
                self.fail(e)

        self.assertTrue(res.success, (
            "failed to set param | param_id: {0}, param_value: {1} | timeout(seconds): {2}".
                format(param_id, value, timeout)))

    def wait_for_topics(self, timeout):
        """wait for simulation to be ready, make sure we're getting topic info
        from all topics by checking dictionary of flag values set in callbacks,
        timeout(int): seconds"""
        rospy.loginfo("waiting for subscribed topics to be ready")
        loop_freq = 1  # Hz
        rate = rospy.Rate(loop_freq)
        simulation_ready = False
        for i in xrange(timeout * loop_freq):
            rospy.loginfo(str(self.sub_topics_ready))
            if all(value for value in self.sub_topics_ready.values()):
                simulation_ready = True
                rospy.loginfo("simulation topics ready | seconds: {0} of {1}".
                              format(i / loop_freq, timeout))
                break

            try:
                rate.sleep()
            except rospy.ROSException as e:
                self.fail(e)

        self.assertTrue(simulation_ready, (
            "failed to hear from all subscribed simulation topics | topic ready flags: {0} | timeout(seconds): {1}".
                format(self.sub_topics_ready, timeout)))

    def wait_for_landed_state(self, desired_landed_state, timeout, index):
        rospy.loginfo("waiting for landed state | state: {0}, index: {1}".
                      format(mavutil.mavlink.enums['MAV_LANDED_STATE'][
                                 desired_landed_state].name, index))
        loop_freq = 10  # Hz
        rate = rospy.Rate(loop_freq)
        landed_state_confirmed = False
        for i in xrange(timeout * loop_freq):
            if self.extended_state.landed_state == desired_landed_state:
                landed_state_confirmed = True
                rospy.loginfo("landed state confirmed | seconds: {0} of {1}".
                              format(i / loop_freq, timeout))
                break

            try:
                rate.sleep()
            except rospy.ROSException as e:
                self.fail(e)

        self.assertTrue(landed_state_confirmed, (
            "landed state not detected | desired: {0}, current: {1} | index: {2}, timeout(seconds): {3}".
                format(mavutil.mavlink.enums['MAV_LANDED_STATE'][
                           desired_landed_state].name, mavutil.mavlink.enums[
                           'MAV_LANDED_STATE'][self.extended_state.landed_state].name,
                       index, timeout)))

    def wait_for_vtol_state(self, transition, timeout, index):
        """Wait for VTOL transition, timeout(int): seconds"""
        rospy.loginfo(
            "waiting for VTOL transition | transition: {0}, index: {1}".format(
                mavutil.mavlink.enums['MAV_VTOL_STATE'][
                    transition].name, index))
        loop_freq = 10  # Hz
        rate = rospy.Rate(loop_freq)
        transitioned = False
        for i in xrange(timeout * loop_freq):
            if transition == self.extended_state.vtol_state:
                rospy.loginfo("transitioned | seconds: {0} of {1}".format(
                    i / loop_freq, timeout))
                transitioned = True
                break

            try:
                rate.sleep()
            except rospy.ROSException as e:
                self.fail(e)

        self.assertTrue(transitioned, (
            "transition not detected | desired: {0}, current: {1} | index: {2} timeout(seconds): {3}".
                format(mavutil.mavlink.enums['MAV_VTOL_STATE'][transition].name,
                       mavutil.mavlink.enums['MAV_VTOL_STATE'][
                           self.extended_state.vtol_state].name, index, timeout)))

    def wait_for_mav_type(self, timeout):
        """Wait for MAV_TYPE parameter, timeout(int): seconds"""
        rospy.loginfo("waiting for MAV_TYPE")
        loop_freq = 1  # Hz
        rate = rospy.Rate(loop_freq)
        res = False
        for i in xrange(timeout * loop_freq):
            try:
                res = self.get_param_srv('MAV_TYPE')
                if res.success:
                    self.mav_type = res.value.integer
                    rospy.loginfo(
                        "MAV_TYPE received | type: {0} | seconds: {1} of {2}".
                            format(mavutil.mavlink.enums['MAV_TYPE'][self.mav_type]
                                   .name, i / loop_freq, timeout))
                    break
            except rospy.ServiceException as e:
                rospy.logerr(e)

            try:
                rate.sleep()
            except rospy.ROSException as e:
                self.fail(e)

        self.assertTrue(res.success, (
            "MAV_TYPE param get failed | timeout(seconds): {0}".format(timeout)
        ))

    def log_topic_vars(self):
        """log the state of topic variables"""
        # rospy.loginfo("========================")
        # rospy.loginfo("===== topic values =====")
        # rospy.loginfo("========================")
        # rospy.loginfo("ALTITUDE: {}".format(self.altitude.local))
        # rospy.loginfo("========================")
        # rospy.loginfo("extended_state:\n{}".format(self.extended_state))
        # rospy.loginfo("========================")
        # rospy.loginfo("global_position:\n{}".format(self.global_position))
        # rospy.loginfo("========================")
        # rospy.loginfo("home_position:\n{}".format(self.home_position))
        # rospy.loginfo("========================")
        # rospy.loginfo("local_position:\n{}".format(self.local_position))
        # rospy.loginfo("========================")
        # rospy.loginfo("mission_wp:\n{}".format(self.mission_wp))
        # rospy.loginfo("========================")
        # rospy.loginfo("state:\n{}".format(self.state))
        # rospy.loginfo("========================")


if __name__ == '__main__':
    import rostest

    rospy.init_node('test_node', anonymous=True)

    rostest.rosrun("navigation", 'mavros_offboard_posctl_test',
                   RL_Fly)
