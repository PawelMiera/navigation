from slow_process import preprocess_slow_median
import numpy as np
import cv2
from math import sin, cos, pi

def render(laser_ranges, window_name):
    window_size = (900, 900)
    window_size_half = (int(window_size[0] / 2), int(window_size[1] / 2))
    laser_resolution = 360
    pixels_per_meter = 70
    laser_angle_per_step = 2 * pi / laser_resolution

    background = np.zeros((window_size[0], window_size[1], 3), np.uint8)
    background[:] = (0, 255, 0)

    for i in range(laser_resolution):
        A = int(laser_ranges[i] * pixels_per_meter * sin(i * laser_angle_per_step)) + \
            window_size_half[0]
        B = -int(laser_ranges[i] * pixels_per_meter * cos(i * laser_angle_per_step)) + \
            window_size_half[1]
        cv2.line(background, window_size_half, (A, B), (0, 0, 255), 1)
        cv2.circle(background, (A, B), 2, (0, 0, 255), -1)

    drone_x_min = window_size_half[0] - 15
    drone_y_min = window_size_half[1] - 10
    background = cv2.rectangle(background, (drone_y_min, drone_x_min), (drone_y_min + 20, drone_x_min + 30),
                               (0, 0, 0), -1)
    background = cv2.rectangle(background, (drone_y_min, drone_x_min), (drone_y_min + 20, drone_x_min + 5),
                               (255, 0, 0), -1)

    cv2.imshow(window_name, background)

i = 0
while i < 172:
    if i <0:
        i = 0
    print(i)
    a = np.load("saved/" + str(i) + ".npz.npy")

    a = np.subtract(a, 0.1)

    a = np.minimum(a, 6.0)
    a = np.maximum(a, 0.15)

    processed = preprocess_slow_median(a, 360, 6.0, 0.15)



    render(a, "before")
    render(processed, "after")
    key = cv2.waitKey(0)


    if key == ord("q"):
        break
    elif key == ord("b"):
        i -= 2

    i += 1
