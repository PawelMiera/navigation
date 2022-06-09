import rospy
from sensor_msgs.msg import LaserScan
import cv2


class Laser:
    def __init__(self):
        self.window_size = (1000, 1000)
        self.window_size_half = (int(self.window_size[0] / 2), int(self.window_size[1] / 2))
        self.pixels_per_meter = 50
        self.laser_angle_per_step = 2 * pi / self.laser_resolution
        self.laser_max_range = 10
        self.laser_min_range = 0.15
        self.laser_resolution = 360
        self.laser_ranges = np.full(self.laser_resolution, self.laser_max_range, dtype=np.float32)
        self.laser_data = np.full(self.laser_resolution, self.laser_max_range, dtype=np.float32)
        rospy.init_node('scan_values')
        sub = rospy.Subscriber('/scan', LaserScan, self.laser_callback)

        while True:
            self.preprocess_lasers()
            self.render()

    def laser_callback(self, msg):
        self.laser_data = np.array(msg.ranges).astype(np.float32)


    def preprocess_lasers(self):
        data = self.laser_data
        data[data == np.inf] = self.laser_max_range

        data = data[::-1]
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