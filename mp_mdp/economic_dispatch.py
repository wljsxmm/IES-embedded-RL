import pulp
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt

plt.style.use(['science', 'ieee', 'no-latex'])
plt.rcParams.update({
    "font.family": "serif",  # specify font family here
    "font.serif": ["Times"],  # specify font here
    "font.size": 6})  # specify font size here

import math


def generate_hourly_demand(min_demand, max_demand, hours):
    amplitude = (max_demand - min_demand) / 2
    base_demand = min_demand + amplitude
    hourly_demand = [int(base_demand + amplitude * math.sin(2 * math.pi * hour / 24)) for hour in range(hours)]
    return hourly_demand


hourly_demand_all = []
for i in range(5):
    min_demand = 150
    max_demand = 400
    hours = 24
    demand_temp = generate_hourly_demand(min_demand, max_demand, hours)
    hourly_demand_all.append(demand_temp)

# 发电机参数
generators_chp = {
    "G1": {"a": 0.000171, "b": 0.2705, "c": 11.537, "Pmin": 10, "Pmax": 200, "heat_ratio": 0.5, "cv1": 0.15,
           "cv2": 0.15, "cm": 0.75, "Hmax": 250},
    "G2": {"a": 0.000072, "b": 0.2292, "c": 14.618, "Pmin": 20, "Pmax": 150, "heat_ratio": 0.4, "cv1": 0.15,
           "cv2": 0.15, "cm": 0.75, "Hmax": 450},
    "G3": {"a": 0.000072, "b": 0.2292, "c": 14.618, "Pmin": 30, "Pmax": 400, "heat_ratio": 0.3, "cv1": 0.15,
           "cv2": 0.15, "cm": 0.75, "Hmax": 450},
    "G4": {"a": 0.000076, "b": 0.2716, "c": 18.822, "Pmin": 30, "Pmax": 400, "heat_ratio": 0.3, "cv1": 0.15,
           "cv2": 0.15, "cm": 0.75, "Hmax": 400},
    "G5": {"a": 0.000076, "b": 0.2716, "c": 18.822, "Pmin": 30, "Pmax": 400, "heat_ratio": 0.3, "cv1": 0.15,
           "cv2": 0.15, "cm": 0.75, "Hmax": 400},
    "G6": {"a": 0.000076, "b": 0.2716, "c": 18.822, "Pmin": 30, "Pmax": 400, "heat_ratio": 0.3, "cv1": 0.15,
           "cv2": 0.15, "cm": 0.75, "Hmax": 400},
    # "G7": {"a": 0.000171, "b": 0.2705, "c": 11.537, "Pmin": 80, "Pmax": 200, "heat_ratio": 0.3, "cv1": 0, "cv2": 0, "cm":0},
    # "G8": {"a": 0.000038, "b": 0.2716, "c": 37.645, "Pmin": 200, "Pmax": 500, "heat_ratio": 0.3, "cv1": 0, "cv2": 0, "cm":0},
    # "Wind": {"a": 0, "b": 0, "c": 0, "Pmin": 0, "Pmax": 1000},
}

generators_condensing = {
    "G7": {"a": 0.000171, "b": 0.2705, "c": 11.537, "Pmin": 80, "Pmax": 200, "heat_ratio": 0.3, "cv1": 0, "cv2": 0,
           "cm": 0},
    "G8": {"a": 0.000038, "b": 0.2716, "c": 37.645, "Pmin": 200, "Pmax": 500, "heat_ratio": 0.3, "cv1": 0, "cv2": 0,
           "cm": 0},
}

# 储热设备参数
thermal_storage = {
    "capacity": 1000,
    "initial_energy": 500,
    "charge_capacity": 100,
    "discharge_capacity": 100,
}

# 每小时的负荷
hourly_demand = [2130, 2208, 2296, 2254, 2112, 2140, 2262, 2400, 2350, 2182, 2098, 2038, 1915, 1860, 1800, 1782, 1702,
                 1696, 1694, 1716,
                 1770, 1792, 1864, 1946]
# hourly_demand_all = [hourly_demand, hourly_demand[::-1]]

# 生成每小时的负荷都为900
hourly_heat_demand = [900 for i in range(24)]
# hourly_heat_demand = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240,
#                       250, 260, 270, 280]
hourly_heat_demand_all = [hourly_heat_demand, hourly_heat_demand[::-1]]

hourly_wind_power_available = [255, 233, 194, 186, 202, 190, 181, 217, 223, 235, 255, 260, 268, 270, 269, 250, 241, 258,
                               268, 278, 288, 300, 280, 262]
hourly_wind_power_available_all = [hourly_wind_power_available, hourly_wind_power_available[::-1]]


def economic_dispatch_24h(hourly_demand, hourly_heat_demand, hourly_wind_power_available, initial_energy):
    power_vars_chp = pulp.LpVariable.dicts("Power", ((hour, name) for hour in range(len(hourly_demand)) for name in
                                                     generators_chp.keys()), lowBound=0)
    power_vars_condensing = pulp.LpVariable.dicts("Power",
                                                  ((hour, name) for hour in range(len(hourly_demand)) for name in
                                                   generators_condensing.keys()), lowBound=0)
    heat_vars_chp = pulp.LpVariable.dicts("Heat", ((hour, name) for hour in range(len(hourly_demand)) for name in
                                                   generators_chp.keys()), lowBound=0)

    storage_charge_vars = pulp.LpVariable.dicts("StorageCharge", (hour for hour in range(len(hourly_demand))),
                                                lowBound=0, upBound=thermal_storage["charge_capacity"])
    storage_discharge_vars = pulp.LpVariable.dicts("StorageDischarge", (hour for hour in range(len(hourly_demand))),
                                                   lowBound=0, upBound=thermal_storage["discharge_capacity"])
    storage_energy_vars = pulp.LpVariable.dicts("StorageEnergy", (hour for hour in range(len(hourly_demand) + 1)),
                                                lowBound=0, upBound=thermal_storage["capacity"])
    wind_power_vars = pulp.LpVariable.dicts("WindPower", (hour for hour in range(len(hourly_demand))),
                                            lowBound=0, upBound=hourly_wind_power_available)
    quadratic_term_vars = {(hour, name): pulp.LpVariable(f"quadratic_term_{hour}_{name}", 0) for hour in
                           range(len(hourly_demand)) for name in generators_condensing.keys()}

    lp_problem = pulp.LpProblem("EconomicDispatch24h", pulp.LpMinimize)
    # 添加目标函数
    lp_problem += pulp.lpSum([(generators_condensing[name]["a"] * quadratic_term_vars[hour, name] + generators_condensing[name]["b"] *
                               power_vars_condensing[hour, name] + generators[name]["c"]) for hour in range(len(hourly_demand)) for
                              name in generators.keys()])

    # 添加约束条件
    for hour in range(len(hourly_demand)):
        # 电力平衡
        lp_problem += (pulp.lpSum([power_vars_chp[hour, name] for name in generators_chp.keys()]) +
                       pulp.lpSum([power_vars_condensing[hour, name] for name in generators_condensing.keys()])) \
                      == hourly_demand[hour] + wind_power_vars[hour]
        # 热力平衡
        lp_problem += pulp.lpSum([heat_vars_chp[hour, name] for name in generators_chp.keys()]) == hourly_heat_demand[hour]

        # 热电比的写法
        # lp_problem += pulp.lpSum(
        #     [generators[name]["heat_ratio"] * power_vars[hour, name] for name in generators.keys() if
        #      "heat_ratio" in generators[name]]) - storage_discharge_vars[
        #     hour] + storage_charge_vars[hour] == hourly_heat_demand[hour]

        # 发电机输出范围约束
        for name in generators_condensing.keys():
            lp_problem += generators_condensing[name]["Pmin"] <= power_vars_condensing[hour, name] <= generators_condensing[name]["Pmax"]
        for name in generators_chp.keys():
            lp_problem += generators_chp[name]["Pmin"] <= power_vars_chp[hour, name] <= generators_chp[name]["Pmax"]
            lp_problem += generators_chp[name]["Qmin"] <= heat_vars_chp[hour, name] <= generators_chp[name]["Qmax"]
        # 储热设备充放电约束
        lp_problem += storage_energy_vars[hour] - storage_energy_vars[hour + 1] == storage_discharge_vars[hour] - \
                      storage_charge_vars[hour]
        # 储热设备容量约束
        lp_problem += storage_energy_vars[hour] <= thermal_storage["capacity"]

        # 风电出力约束
        lp_problem += wind_power_vars[hour] <= hourly_wind_power_available[hour]

    # 首尾储热量相等约束
    lp_problem += storage_energy_vars[0] == storage_energy_vars[len(hourly_demand)]

    # 初始储热量约束
    lp_problem += storage_energy_vars[0] == initial_energy

    lp_problem.solve()
    if pulp.LpStatus[lp_problem.status] == "Optimal":
        results = {
            "power": {(hour, name): power_vars[hour, name].value() for hour in range(len(hourly_demand)) for name in
                      generators.keys()},
            "storage_charge": {hour: storage_charge_vars[hour].value() for hour in range(len(hourly_demand))},
            "storage_discharge": {hour: storage_discharge_vars[hour].value() for hour in range(len(hourly_demand))},
            "storage_energy": {hour: storage_energy_vars[hour].value() for hour in range(len(hourly_demand) + 1)},
            "total_cost": pulp.value(lp_problem.objective),
            "status": pulp.LpStatus[lp_problem.status],
        }
        return results
    else:
        return None


def save_results_to_csv(results, filename):
    power_data = results["power"]
    storage_charge_data = results["storage_charge"]
    storage_discharge_data = results["storage_discharge"]
    storage_energy_data = results["storage_energy"]
    optimization_status = results["status"]

    data = []
    for hour in range(len(hourly_demand)):
        row = {"hour": hour}
        for name in generators.keys():
            row[name] = power_data[hour, name]
        row["storage_charge"] = storage_charge_data[hour]
        row["storage_discharge"] = storage_discharge_data[hour]
        row["storage_energy"] = storage_energy_data[hour]
        row["status"] = optimization_status
        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)


# 添加风电功率数据到results字典
# results["wind_power"] = hourly_wind_power_available
# save_results_to_csv(results, "economic_dispatch_results.csv")

def plot_results(results):
    power_data = results["power"]
    storage_charge_data = results["storage_charge"]
    storage_discharge_data = results["storage_discharge"]
    storage_energy_data = results["storage_energy"]
    wind_power_data = results["wind_power"]

    hours = range(len(hourly_demand))

    # 绘制发电机组功率输出
    for name in generators.keys():
        plt.plot(hours, [power_data[hour, name] for hour in hours], label=name)

    # 绘制风电出力曲线
    plt.plot(hours, [wind_power_data[hour] for hour in hours], label="wind_power", linestyle="-.")

    # 绘制储热设备的充放电
    plt.plot(hours, [storage_charge_data[hour] for hour in hours], label="storage_charge")
    plt.plot(hours, [storage_discharge_data[hour] for hour in hours], label="storage_discharge")

    # 绘制储热量
    plt.plot(hours, [storage_energy_data[hour] for hour in hours], label="storage_energy", linestyle="--")

    plt.xlabel("Hour")
    plt.ylabel("Power (MW) / Energy (MWh)")
    plt.title("Economic Dispatch Results")
    plt.legend()
    plt.grid()
    plt.show()


# 使用新函数绘制储热设备的充放电和储热量
def plot_storage(results):
    storage_charge_data = results["storage_charge"]
    storage_discharge_data = results["storage_discharge"]
    storage_energy_data = results["storage_energy"]

    hours = range(len(hourly_demand))

    plt.plot(hours, [storage_charge_data[hour] for hour in hours], label="storage_charge")
    plt.plot(hours, [storage_discharge_data[hour] for hour in hours], label="storage_discharge")
    plt.plot(hours, [storage_energy_data[hour] for hour in hours], label="storage_energy", linestyle="--")

    plt.xlabel("Hour")
    plt.ylabel("Power (MW) / Energy (MWh)")
    plt.title("Storage Dispatch Results")
    plt.legend()
    plt.grid()
    plt.show()


def results_process(results):
    # 使用 defaultdict 初始化一个空字典，使默认值为一个空列表
    classified_data = defaultdict(list)
    power = results["power"]
    # 遍历给定的数据字典
    for key, value in power.items():
        hour, generator_type = key
        classified_data[generator_type].append((hour, value))
    # 将 defaultdict 转换回普通字典
    classified_data = dict(classified_data)
    return classified_data


def plot_wind_power(results):
    wind_power_data = results["wind_power"]
    classified_data = results_process(results)
    wind_power_forecast = classified_data["Wind"]

    hours = range(len(hourly_demand))
    plt.plot(hours, [wind_power_data[hour] for hour in hours], label="wind_power", linestyle="-.")
    plt.plot(hours, [wind_power_forecast[hour] for hour in hours], label="wind_power_forecast", linestyle="--")

    plt.xlabel("Hour")
    plt.ylabel("Power (MW)")
    plt.title("Wind Power Output")
    plt.legend()
    plt.grid()
    plt.show()


def plot_generators(results):
    classified_data = results_process(results)
    # 提取各发电机的出力数据
    g1_output = [output for _, output in classified_data['G1']]
    g2_output = [output for _, output in classified_data['G2']]
    g3_output = [output for _, output in classified_data['G3']]

    # 提取小时数据
    hours = [hour for hour, _ in classified_data['G1']]

    # 绘制堆叠柱状图
    plt.bar(hours, g1_output, label='G1')
    plt.bar(hours, g2_output, bottom=g1_output, label='G2')
    plt.bar(hours, g3_output, bottom=[sum(x) for x in zip(g1_output, g2_output)], label='G3')

    # 添加图例和轴标签
    plt.xlabel('Hours')
    plt.ylabel('Output')
    plt.legend()

    # 显示图形
    plt.show()


# main 主函数
if __name__ == "__main__":
    for day in range(2):
        # 24小时的经济调度
        results = economic_dispatch_24h(hourly_demand_all[day], hourly_heat_demand_all[day],
                                        hourly_wind_power_available_all[day],
                                        thermal_storage["initial_energy"])
        # 更新储热设备初始储热量
        thermal_storage["initial_energy"] = results["storage_energy"][0]

        # 输出总成本
        if results is not None:
            print(f"第{day + 1}天")
            print("Total cost for 24 hours dispatch: ", results["total_cost"])
        else:
            print("Status: Infeasible")

        filename = f"results/day_{day + 1}.csv"
        results["wind_power"] = hourly_wind_power_available_all[day]
        save_results_to_csv(results, filename)
        plot_generators(results)
        plot_wind_power(results)
        plot_storage(results)
        plot_results(results)
