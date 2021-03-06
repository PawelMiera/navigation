from fast_process import preprocess_fast_median
import numpy as np
import torch
from stable_baselines3.common.utils import get_device
from stable_baselines3.ppo.policies import MlpPolicy

import rospy
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import LaserScan


class RlNode:

    def __init__(self):

        self.new_data = False
        self.ind = 0
        self.laser_max_range = 6
        self.laser_min_range = 0.15
        self.laser_resolution = 360
        self.laser_data = np.full(self.laser_resolution, self.laser_max_range, dtype=np.float32)

        self.corrupt_size = 0

        self.corrupt_ind = 0


        device = get_device("auto")
        saved_variables = torch.load("m_360_61_policy.zip", map_location=device)

        self.model = MlpPolicy(**saved_variables["data"])
        self.model.load_state_dict(saved_variables["state_dict"], strict=False)
        self.model.to(device)

        laser_ranges = preprocess_fast_median(self.laser_data, self.laser_resolution, self.laser_max_range,
                                              self.laser_min_range)

        self.pub = rospy.Publisher('/rl_control', Float32MultiArray, queue_size=1)
        self.sub = rospy.Subscriber('/scan', LaserScan, self.laser_callback)

        rate = rospy.Rate(1)

        while not self.new_data:
            rospy.loginfo("Waiting for data")
            try:  # prevent garbage in console output when thread is killed
                rate.sleep()
            except rospy.ROSInterruptException:
                pass

    def laser_callback(self, msg):
        if self.ind % 30 == 0:
            rospy.loginfo(str(len(msg.ranges)))
        self.ind += 1
        if len(msg.ranges) == 360:
            self.corrupt_ind = 0
            self.new_data = True
            self.laser_data = np.array(msg.ranges).astype(np.float32)
        else:
            self.corrupt_size += 1
            self.corrupt_ind += 1
            rospy.loginfo("corrupt data size " + str(self.corrupt_size) + " / " + str(self.ind))

        if self.corrupt_ind > 20:
            data_to_send = Float32MultiArray()
            data_to_send.data = np.array([0, 0])
            self.pub.publish(data_to_send)
            rospy.loginfo("stopping vehicle!")

    def normalize_lasers(self, laser_ranges):
        divider = self.laser_max_range / 2
        laser_ranges = ((laser_ranges.copy() - divider) / divider).astype(np.float32)
        return laser_ranges

    def run(self):

        loop_rate = rospy.Rate(30)
        rospy.loginfo("Node running!")
        while not rospy.is_shutdown():
            if self.new_data:
                self.new_data = False

                data = self.laser_data.copy()
                data = np.subtract(data, 0.2)

                data = np.minimum(data, self.laser_max_range)
                data = np.maximum(data, self.laser_min_range)
                laser_ranges = preprocess_fast_median(data, self.laser_resolution, self.laser_max_range,
                                                      self.laser_min_range)

                obs = self.normalize_lasers(laser_ranges)

                action, _states = self.model.predict(obs, deterministic=True)

                data_to_send = Float32MultiArray()  # the data to be sent, initialise the array
                data_to_send.data = action  # assign the array with the value you want to send
                self.pub.publish(data_to_send)


            try:  # prevent garbage in console output when thread is killed
                loop_rate.sleep()
            except rospy.ROSInterruptException:
                pass


if __name__ == '__main__':
    rospy.init_node('RL_Node', anonymous=True)
    rlnode = RlNode()
    rlnode.run()
