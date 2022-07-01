from numba import jit
import numpy as np

@jit(nopython=True)
def preprocess_fast(laser_ranges, laser_resolution, laser_max_range, laser_min_range):
    data2 = laser_ranges.copy()

    for i in range(laser_resolution):
        if laser_ranges[i] >= laser_max_range or laser_ranges[i] < laser_min_range:
            curr_sum = 0
            count = 0
            for j in range(i - 2, i + 3):
                if j != i:
                    ind = j
                    if j < 0:
                        ind = laser_resolution + j
                    elif j >= laser_resolution:
                        ind = j - laser_resolution
                    if laser_min_range < laser_ranges[ind] < laser_max_range:
                        curr_sum += laser_ranges[ind]
                        count += 1
            if curr_sum != 0:
                data2[i] = curr_sum / count

            if data2[i] > laser_max_range:
                data2[i] = laser_max_range
            elif data2[i] < laser_min_range:
                data2[i] = laser_min_range

    for i in range(laser_resolution):
        mean_filter_sum = 0
        mean_filter_count = 0
        for j in range(i - 1, i + 2):
            ind = j
            if j < 0:
                ind = laser_resolution + j
            elif j >= laser_resolution:
                ind = j - laser_resolution

            mean_filter_sum += data2[ind]
            mean_filter_count += 1

        data2[i] = mean_filter_sum / mean_filter_count
        data2[i] = data2[i] - 0.1

        if data2[i] < laser_min_range:
            data2[i] = laser_min_range

    return data2


@jit(nopython=True)
def preprocess_fast_median(laser_ranges, laser_resolution, laser_max_range, laser_min_range):
    data2 = laser_ranges.copy()
    out_data = laser_ranges.copy()

    for i in range(laser_resolution):
        if laser_ranges[i] >= laser_max_range or laser_ranges[i] < laser_min_range:
            curr_sum = 0
            count = 0
            for j in range(i - 2, i + 3):
                if j != i:
                    ind = j
                    if j < 0:
                        ind = laser_resolution + j
                    elif j >= laser_resolution:
                        ind = j - laser_resolution
                    if laser_min_range < laser_ranges[ind] < laser_max_range:
                        curr_sum += laser_ranges[ind]
                        count += 1
            if curr_sum != 0:
                data2[i] = curr_sum / count

            if data2[i] > laser_max_range:
                data2[i] = laser_max_range
            elif data2[i] < laser_min_range:
                data2[i] = laser_min_range

    for i in range(laser_resolution):
        values_list = []
        for j in range(i - 4, i + 5):
            ind = j
            if j < 0:
                ind = laser_resolution + j
            elif j >= laser_resolution:
                ind = j - laser_resolution

            values_list.append(data2[ind])

        values_list = sorted(values_list)

        out_data[i] = values_list[4] - 0.1

        if out_data[i] < laser_min_range:
            out_data[i] = laser_min_range

    return out_data

    return out_data




