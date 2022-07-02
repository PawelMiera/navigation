import rospy
from sensor_msgs.msg import LaserScan
import cv2
from math import pi, sin, cos
import numpy as np
from slow_process import preprocess_slow_median
import os

class Laser:
    def __init__(self):
        self.window_size = (900, 900)
        self.window_size_half = (int(self.window_size[0] / 2), int(self.window_size[1] / 2))
        self.pixels_per_meter = 70

        self.laser_resolution = 360
        self.laser_angle_per_step = 2 * pi / self.laser_resolution
        self.laser_max_range = 6
        self.laser_min_range = 0.15
        self.laser_ranges = np.full(self.laser_resolution, self.laser_max_range, dtype=np.float32)
        self.laser_data = np.full(self.laser_resolution, self.laser_max_range, dtype=np.float32)
        rospy.init_node('scan_values')
        sub = rospy.Subscriber('/scan', LaserScan, self.laser_callback)

        self.start_save = False
        self.save_ind = 0

        while True:
            data = self.laser_data.copy()
            data = np.subtract(data, 0.1)

            data = np.minimum(data, self.laser_max_range)
            data = np.maximum(data, self.laser_min_range)
            self.laser_ranges = preprocess_slow_median(data, self.laser_resolution, self.laser_max_range,
                                                  self.laser_min_range)

            key = self.render()
            if key == ord("q"):
                break
            elif key == ord("s"):
                self.start_save =True
                print("saving")
                if not os.path.exists:
                    os.mkdir("saved")


    def laser_callback(self, msg):
        self.laser_data = np.array(msg.ranges).astype(np.float32)
        if self.start_save:
            np.save("saved/" + str(self.save_ind) + ".npz", self.laser_data)
            self.save_ind += 1
        print(len(self.laser_data))


    def preprocess_lasers2(self):
        data = self.laser_data.copy()

        mask = np.isinf(data)
        mask_min = data < self.laser_min_range + 0.2

        #print("max: ", np.max(data[np.logical_not(mask)]), " min ", np.min(data))

        data = np.maximum(data, self.laser_min_range)
        data = np.minimum(data, self.laser_max_range)

        #print(data[np.logical_not(mask)])

        out_list = [np.array([data[self.laser_resolution-1], data[1]])]
        for i in range(1, self.laser_resolution - 1):
            curr = np.concatenate([data[i - 1:i], data[i+1:i + 2]])
            out_list.append(curr)

        out_list.append([data[self.laser_resolution-2], data[0]])

        neighbours_closest_min = np.min(out_list, axis=1)

        neighbours_closest_max = np.max(out_list, axis=1)

        data[mask_min] = neighbours_closest_max[mask_min]
        data[mask] = neighbours_closest_min[mask]

        #print("2max: ", np.max(data), " min ", np.min(data))qq

    def preprocess_lasers(self):
        data = self.laser_data.copy()

        mask = np.isnan(data)

        data[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), data[~mask])

        # print("max: ", np.max(data[np.logical_not(mask)]))
        #
        # #print(data[np.logical_not(mask)])
        #
        # out_list = [np.array([data[self.laser_resolution-1], data[1]])]
        # for i in range(1, self.laser_resolution - 1):
        #     curr = np.concatenate([data[i - 1:i], data[i+1:i + 2]])
        #     out_list.append(curr)
        #
        # out_list.append([data[self.laser_resolution-2], data[0]])
        #
        # neighbours_closest_min = np.min(out_list, axis=1)
        #
        # data[mask] = neighbours_closest_min[mask]

        # data = np.maximum(data, self.laser_min_range)
        # data = np.minimum(data, self.laser_max_range)

        self.laser_ranges = data

    def render(self, mode='human'):
        background = np.zeros((self.window_size[0], self.window_size[1], 3), np.uint8)
        background[:] = (0, 255, 0)

        for i in range(self.laser_resolution):
            A = int(self.laser_ranges[i] * self.pixels_per_meter * sin(i * self.laser_angle_per_step)) + \
                self.window_size_half[0]
            B = -int(self.laser_ranges[i] * self.pixels_per_meter * cos(i * self.laser_angle_per_step)) + \
                self.window_size_half[1]
            cv2.line(background, self.window_size_half, (A, B), (0, 0, 255), 1)
            cv2.circle(background, (A, B), 2, (0, 0, 255), -1)

        drone_x_min = self.window_size_half[0] - 15
        drone_y_min = self.window_size_half[1] - 10
        background = cv2.rectangle(background, (drone_y_min, drone_x_min), (drone_y_min + 20, drone_x_min + 30),
                                   (0, 0, 0), -1)
        background = cv2.rectangle(background, (drone_y_min, drone_x_min), (drone_y_min + 20, drone_x_min + 5),
                                   (255, 0, 0), -1)

        cv2.imshow("game", background)
        return cv2.waitKey(1)


laser = Laser()
