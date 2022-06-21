def preprocess_slow(laser_ranges, laser_resolution, laser_max_range, laser_min_range):
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

    return data2