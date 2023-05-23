import matplotlib
# matplotlib.use('WebAgg')

import matplotlib.pyplot as plt
from parameters import *
import numpy as np
from collections import defaultdict


# plt.style.use(['science', 'ieee', 'no-latex'])
# plt.rcParams.update({
#     "font.family": "serif",  # specify font family here
#     "font.serif": ["Times"],  # specify font here
#     "font.size": 6})  # specify font size here

# TODO plot_storage plot_heat修改
def results_process_chp_power(results, day):
    # 使用 defaultdict 初始化一个空字典，使默认值为一个空列表
    classified_data = defaultdict(list)
    for i in range(day):
        power_chp = results[i]["CHP_Power"]
        for key, value in power_chp.items():
            hour, generator_type = key
            classified_data[generator_type].append((hour + 24 * i, value))
    classified_data = dict(classified_data)
    return classified_data


def results_process_condensing(results, day):
    # 使用 defaultdict 初始化一个空字典，使默认值为一个空列表
    classified_data = defaultdict(list)
    for i in range(day):
        power_condensing = results[i]["Condensing_Power"]
        # 遍历给定的数据字典
        for key, value in power_condensing.items():
            hour, generator_type = key
            classified_data[generator_type].append((hour + 24 * i, value))
        # 将 defaultdict 转换回普通字典
    classified_data = dict(classified_data)
    return classified_data


def plot_power(results, day):
    classified_data_chp = results_process_chp_power(results, day)
    classified_data_condensing = results_process_condensing(results, day)
    # 提取各发电机的出力数据
    g1_output = [output for _, output in classified_data_chp['G1']]
    g2_output = [output for _, output in classified_data_chp['G2']]
    g3_output = [output for _, output in classified_data_chp['G3']]
    g4_output = [output for _, output in classified_data_chp['G4']]
    g5_output = [output for _, output in classified_data_chp['G5']]
    g6_output = [output for _, output in classified_data_chp['G6']]
    g7_output = [output for _, output in classified_data_condensing['G7']]
    g8_output = [output for _, output in classified_data_condensing['G8']]

    # 提取小时数据
    hours = [hour for hour, _ in classified_data_chp['G1']]

    # 绘制堆叠柱状图
    plt.bar(hours, g1_output, label='G1')
    plt.bar(hours, g2_output, bottom=g1_output, label='G2')
    plt.bar(hours, g3_output, bottom=[sum(x) for x in zip(g1_output, g2_output)], label='G3')
    plt.bar(hours, g4_output, bottom=[sum(x) for x in zip(g1_output, g2_output, g3_output)], label='G4')
    plt.bar(hours, g5_output, bottom=[sum(x) for x in zip(g1_output, g2_output, g3_output, g4_output)], label='G5')
    plt.bar(hours, g6_output, bottom=[sum(x) for x in zip(g1_output, g2_output, g3_output, g4_output, g5_output)],
            label='G6')
    plt.bar(hours, g7_output, bottom=[sum(x) for x in
                                      zip(g1_output, g2_output, g3_output, g4_output, g5_output, g6_output)],
            label='G7')
    plt.bar(hours, g8_output, bottom=[sum(x) for x in
                                      zip(g1_output, g2_output, g3_output, g4_output, g5_output, g6_output,
                                          g7_output)],
            label='G8')

    # 绘制load折线图
    plt.plot(hours, hourly_demand * 3, label='load', linestyle="-.")

    # 添加图例和轴标签
    plt.xlabel('Hours')
    plt.ylabel('Output')

    # plt.legend(bbox_to_anchor=(0, 0), loc='upper left', ncol=3)
    plt.legend(loc='upper left', ncol=3)
    # 显示图形
    plt.show()


def results_process_chp_heat(results, day):
    # 使用 defaultdict 初始化一个空字典，使默认值为一个空列表
    classified_data = defaultdict(list)
    # heat_chp = results[0]["Heat_CHP"]

    for day in range(day):
        heat_chp = (results[day]["Heat_CHP"])
        for key, value in heat_chp.items():
            hour, generator_type = key
            classified_data[generator_type].append((hour + 24 * day, value))

    # 将 defaultdict 转换回普通字典
    classified_data = dict(classified_data)
    return classified_data


def results_process_hst_power(results, day):
    # 使用 defaultdict 初始化一个空字典，使默认值为一个空列表
    classified_data = defaultdict(list)
    # 遍历给定的数据字典
    for day in range(day):
        heat_hst = results[day]["Storage_Energy"]
        for key, value in heat_hst.items():
            hour, generator_type = key
            classified_data[generator_type].append((hour + 24 * day, value))
    # 将 defaultdict 转换回普通字典
    classified_data = dict(classified_data)
    return classified_data


def plot_storage(results):
    storage_charge_data = results["Storage_Charge"]
    storage_discharge_data = results["Storage_Discharge"]
    storage_energy_data = results["Storage_Energy"]

    hours = range(len(hourly_demand))

    plt.plot(hours, [storage_charge_data[hour] for hour in hours], label="storage_charge")
    plt.plot(hours, [storage_discharge_data[hour] for hour in hours], label="storage_discharge")
    plt.plot(range(25), [storage_energy_data[hour] for hour in range(25)], label="storage_energy", linestyle="--")

    plt.xlabel("Hour")
    plt.ylabel("Power (MW) / Energy (MWh)")
    plt.title("Storage Dispatch Results")
    plt.legend()
    plt.grid()
    plt.show()


def plot_heat(results, day):
    classified_data_chp = results_process_chp_heat(results)

    # 提取各发电机的出力数据
    g1_heat = [heat for _, heat in classified_data_chp['G1']]
    g2_heat = [heat for _, heat in classified_data_chp['G2']]
    g3_heat = [heat for _, heat in classified_data_chp['G3']]
    g4_heat = [heat for _, heat in classified_data_chp['G4']]
    g5_heat = [heat for _, heat in classified_data_chp['G5']]
    g6_heat = [heat for _, heat in classified_data_chp['G6']]

    # 储热充放热数据
    storage_discharge_data = results["Storage_Discharge"]
    storage_charge_data = results["Storage_Charge"]
    net_storage_heat = [storage_discharge_data[hour] - storage_charge_data[hour] for hour in range(len(hourly_demand))]

    hours = [hour for hour, _ in classified_data_chp['G1']]

    # 绘制堆叠柱状图
    plt.bar(hours, g1_heat, label="G1_heat")
    plt.bar(hours, g2_heat, bottom=g1_heat, label="G2_heat")
    plt.bar(hours, g3_heat, bottom=[g1_heat[i] + g2_heat[i] for i in range(len(g1_heat))], label="G3_heat")
    plt.bar(hours, g4_heat, bottom=[g1_heat[i] + g2_heat[i] + g3_heat[i] for i in range(len(g1_heat))], label="G4_heat")
    plt.bar(hours, g5_heat, bottom=[g1_heat[i] + g2_heat[i] + g3_heat[i] + g4_heat[i] for i in range(len(g1_heat))],
            label="G5_heat")
    plt.bar(hours, g6_heat,
            bottom=[g1_heat[i] + g2_heat[i] + g3_heat[i] + g4_heat[i] + g5_heat[i] for i in range(len(g1_heat))],
            label="G6_heat")
    plt.bar(hours, net_storage_heat,
            bottom=[g1_heat[i] + g2_heat[i] + g3_heat[i] + g4_heat[i] + g5_heat[i] + g6_heat[i] for i in
                    range(len(g1_heat))], label="Net_Storage_Discharge")

    # 绘制热负荷曲线
    plt.plot(hours, hourly_heat_demand, label="Hourly_Demand", linestyle="-.")

    plt.xlabel("Hour")
    plt.ylabel("Heat (MW)")
    plt.title("CHP and TST Heat Output")
    plt.legend()
    plt.grid()
    plt.show()


def plot_wind_power(results, wind_scenarios, day):
    wind_power = []
    for i in range(day):
        wind_power.append(results[i]["Wind_Power"])
    wind_power = [wind_power[i][hour] for i in range(day) for hour in range(24)]

    # wind_power = results["Wind_Power"]
    wind_power_forecast = [wind_scenarios[day][hour] for day in range(day) for hour in range(24)]

    plt.figure(figsize=(10, 6))

    hours = range(0, day * len(hourly_demand))
    plt.plot(range(1, day * len(hourly_demand) + 1), wind_power, label="Wind_Power", linestyle="-", color="black",
             linewidth=1)
    plt.plot(range(1, day * len(hourly_demand) + 1), wind_power_forecast, label="wind_power_forecast", linestyle="--",
             color="black", linewidth=2)

    # 计算出总的弃风量
    wind_power_discard = [wind_power_forecast[hour] - wind_power[hour] for hour in hours]
    # print("===========Wind Power Discard===========: ", sum(wind_power_discard))
    # 计算总的消纳量
    wind_power_accept = sum(wind_power)
    # print("===========Wind Power Accept===========: ", wind_power_accept)
    # 计算风电消纳率
    wind_power_accept_rate = wind_power_accept / sum(wind_power_forecast)
    # print("===========Wind Power Accept Rate===========: ", wind_power_accept_rate)

    # # 加密横坐标
    # plt.xticks(np.arange(1, day * len(hourly_demand) + 2, day))
    # # x轴从0开始
    # plt.xlim(1, day * len(hourly_demand) + 2)
    # plt.yticks(np.arange(0, 1000, 100))
    # plt.ylim(0, 1000)
    #
    # plt.xlabel("Hour")
    # plt.ylabel("Power (MW)")
    # plt.title("Wind Power Output")
    # plt.legend()
    # plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    # plt.tight_layout()
    # plt.show()
    return wind_power_accept


def plot_area_b_multi(results, day):
    heat_load = [900 for i in range(24 * day)]

    classified_data_chp = results_process_chp_heat(results, day)
    classified_data_hst = results_process_hst_power(results, day)

    g4_heat = [heat for _, heat in classified_data_chp['G4']]
    g5_heat = [heat for _, heat in classified_data_chp['G5']]
    g6_heat = [heat for _, heat in classified_data_chp['G6']]

    b_heat = [heat for _, heat in classified_data_hst['b']]

    # 删除b_heat 第24、49个元素
    if day > 1:
        if day == 2:
            b_heat.pop(24)  # 删除第24个元素
        elif day == 3:
            b_heat.pop(24)
            b_heat.pop(49)  # 删除第49个元素
        else:
            print("day is wrong")

    # 加总各发电机的产热数据
    area_b_heat = [g4_heat[i] + g5_heat[i] + g6_heat[i] for i in range(len(g4_heat))]

    hours = range(day * len(hourly_demand))
    plt.figure(figsize=(10, 6))

    plt.plot(range(1, day * len(hourly_demand) + 1), [heat_load[hour] for hour in hours], label="heat_load",
             linestyle="-.", color="black", linewidth=1)
    plt.plot(range(1, day * len(hourly_demand) + 2), [b_heat[hour] for hour in range(day * len(hourly_demand) + 1)],
             label="heat_storage", linestyle="--", color="black", linewidth=1, marker="v")
    plt.plot(range(1, day * len(hourly_demand) + 1), [area_b_heat[hour] for hour in hours], label="area_b_heat",
             linestyle="-", color="black", linewidth=1, marker="^")

    # 加密横坐标
    plt.xticks(np.arange(1, day * len(hourly_demand) + 2, day))
    # x轴从0开始
    plt.xlim(1, day * len(hourly_demand) + 2)
    plt.ylim(0, 1100)

    plt.xlabel("Hour")
    plt.ylabel("Heat (MW)")
    plt.title("Area B Heat Output")
    plt.legend()
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    plt.tight_layout()
    plt.show()
