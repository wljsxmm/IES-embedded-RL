# 每小时的负荷
hourly_demand = [2130, 2208, 2296, 2254, 2112, 2140, 2262, 2400, 2350, 2182, 2098, 2038, 1915, 1860, 1800, 1782, 1702,
                 1696, 1694, 1716, 1770, 1792, 1864, 1946]
# hourly_demand = [2130, 2208, 2296, 2254, 1500, 1040, 562, 800, 1400, 2182, 2098, 2038, 1915, 1860, 1800, 1782, 1702, 1696, 1694, 1716, 1770, 1792, 1864, 1946]
# hourly_demand = [2130, 2208, 2296, 2254, 2112, 1640, 2262, 2400, 2350, 2182, 2098, 2038, 1915, 1860, 1800, 1782, 1702, 1696, 1694, 1716, 1770, 1792, 1864, 1946]
# 生成每小时的热负荷都为900
hourly_heat_demand = [900 for i in range(24)]


hourly_wind_power_low = [200 for i in range(24)]
hourly_wind_power_middle = [400 for i in range(24)]
hourly_wind_power_high = [650 for i in range(24)]


hourly_wind_power_available_low = [255, 233, 194, 186, 202, 190, 181, 217, 223, 235, 255, 260, 268, 270, 269, 150, 141, 128, 168, 178, 188, 200, 280, 262]

hourly_wind_power_available = [255, 233, 194, 186, 202, 190, 181, 217, 223, 235, 255, 260, 368, 370, 369, 350, 341, 358, 368, 278, 288, 300, 280, 262]

hourly_wind_power_available_high = [255, 233, 194, 186, 202, 190, 181, 217, 223, 235, 355, 360, 368, 370, 369, 350, 341, 358, 368, 378, 388, 300, 380, 362]

# hourly_wind_power_available的值变成2倍
# hourly_wind_power_available = [i * 4 for i in hourly_wind_power_available]


# hourly_wind_power_available = [255, 233, 194, 186, 567, 786, 678, 797, 381, 381, 255, 260, 268, 270, 369, 350, 241, 258,
#                                268, 278, 288, 300, 280, 262]
hourly_wind_power_available1 = [255, 733, 694, 786, 567, 786, 678, 797, 381, 381, 255, 890, 986, 948, 869, 650, 241,
                                258, 268, 278, 288, 300, 280, 262]

hourly_wind_power_available2 = [255, 833, 794, 886, 867, 886, 978, 797, 381, 381, 255, 890, 986, 948, 869, 650, 241,
                                258, 268, 278, 788, 800, 680, 762]
# hourly_wind_power_available_all = [hourly_wind_power_available, hourly_wind_power_available[::-1]]

# 生成0 300 600 100 的不同组合
# hst_status = [
#     [0, 0], [0, 300], [0, 600], [0, 900],
#     [300, 0], [300, 300], [300, 600], [300, 900],
#     [600, 0], [600, 300], [600, 600], [600, 900],
#     [900, 0], [900, 300], [900, 600], [900, 900]
#     ]
# 生成一个区域的HST的松弛热状态
# hst_status = [
#     [0, 100], [100, 200], [200, 300], [300, 400],
#     [400, 500], [500, 600], [600, 700], [700, 800],
#     [800, 900], [900, 1000]
#     ]

# hst_status = [
#     [0, 200], [200, 400], [400, 600], [600, 800],
#     [800, 1000], [1000, 1200], [1200, 1400], [1400, 1600],
#     [1600, 1800], [1800, 2000]
#     ]

import numpy as np

interval_size = 40

# Generate the intervals
hst_status = [(i, i + interval_size) for i in np.linspace(0, 2000, 50, endpoint=False)]


# Initialize an empty list to store the combinations
# hst_status_combinations = []
#
# # Loop over all statuses of the first HST
# for status1 in hst_status:
#     # Loop over all statuses of the second HST
#     for status2 in hst_status:
#         # Add the combination to the list
#         hst_status_combinations.append([status1, status2])


# 发电机参数
generators_chp = {
    "G1": {"a": 0.000171, "b": 0.2705, "c": 11.537, "Pmin": 100, "Pmax": 200, "heat_ratio": 0.5, "cv1": 0.15,
           "cv2": 0.15, "cm": 0.75, "Hmax": 250, "ramp_up": 50, "ramp_down": 50},
    "G2": {"a": 0.000072, "b": 0.2292, "c": 14.618, "Pmin": 175, "Pmax": 350, "heat_ratio": 0.4, "cv1": 0.15,
           "cv2": 0.15, "cm": 0.75, "Hmax": 450, "ramp_up": 70, "ramp_down": 70},
    "G3": {"a": 0.000072, "b": 0.2292, "c": 14.618, "Pmin": 175, "Pmax": 350, "heat_ratio": 0.3, "cv1": 0.15,
           "cv2": 0.15, "cm": 0.75, "Hmax": 450, "ramp_up": 70, "ramp_down": 70},
    "G4": {"a": 0.000076, "b": 0.2716, "c": 18.822, "Pmin": 30, "Pmax": 300, "heat_ratio": 0.3, "cv1": 0.15,
           "cv2": 0.15, "cm": 0.75, "Hmax": 400, "ramp_up": 150, "ramp_down": 80},
    "G5": {"a": 0.000076, "b": 0.2716, "c": 18.822, "Pmin": 30, "Pmax": 300, "heat_ratio": 0.3, "cv1": 0.15,
           "cv2": 0.15, "cm": 0.75, "Hmax": 400, "ramp_up": 150, "ramp_down": 80},
    "G6": {"a": 0.000076, "b": 0.2716, "c": 18.822, "Pmin": 30, "Pmax": 300, "heat_ratio": 0.3, "cv1": 0.15,
           "cv2": 0.15, "cm": 0.75, "Hmax": 400, "ramp_up": 150, "ramp_down": 80},
    # "G7": {"a": 0.000171, "b": 0.2705, "c": 11.537, "Pmin": 80, "Pmax": 200, "heat_ratio": 0.3, "cv1": 0, "cv2": 0, "cm":0},
    # "G8": {"a": 0.000038, "b": 0.2716, "c": 37.645, "Pmin": 200, "Pmax": 500, "heat_ratio": 0.3, "cv1": 0, "cv2": 0, "cm":0},
}


generators_condensing = {
    "G7": {"a": 0.001171, "b": 0.2705, "c": 11.537, "Pmin": 80, "Pmax": 200, "heat_ratio": 0.3, "cv1": 0, "cv2": 0,
           "cm": 0, "ramp_up": 50, "ramp_down": 50},
    "G8": {"a": 0.000138, "b": 0.2716, "c": 37.645, "Pmin": 200, "Pmax": 500, "heat_ratio": 0.3, "cv1": 0, "cv2": 0,
           "cm": 0, "ramp_up": 130, "ramp_down": 130},
    # "Wind": {"a": 0, "b": 0, "c": 0, "Pmin": 0, "Pmax": 10000, "heat_ratio": 0, "cv1": 0,
    #          "cv2": 0, "cm": 0, "Hmax": 0, "ramp_up": 10000, "ramp_down": 10000},
}

# 区域a和区域b的储热设备参数
thermal_storage = {
    "a": {"capacity": 2000, "initial_heat": 0, "charge_capacity": 200, "discharge_capacity": 200},
    "b": {"capacity": 2000, "initial_heat": 0, "charge_capacity": 200, "discharge_capacity": 200},
}

# thermal_storage = {
#     "a": {"capacity": 0, "initial_heat": 0, "charge_capacity": 0, "discharge_capacity": 0},
#     "b": {"capacity": 0, "initial_heat": 0, "charge_capacity": 0, "discharge_capacity": 0},
# }