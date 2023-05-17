import gurobipy as gp
import pandas as pd
from collections import defaultdict
import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

# plt.style.use(['science', 'ieee', 'no-latex'])
# plt.rcParams.update({
#     "font.family": "serif",  # specify font family here
#     "font.serif": ["Times"],  # specify font here
#     "font.size": 6})  # specify font size here

# TODO 1 分区  2 分析储热罐对于弃风的影响

# 发电机参数
generators_chp = {
    "G1": {"a": 0.000171, "b": 0.2705, "c": 11.537, "Pmin": 10, "Pmax": 200, "heat_ratio": 0.5, "cv1": 0.15,
           "cv2": 0.15, "cm": 0.75, "Hmax": 250, "ramp_up": 50, "ramp_down": 50},
    "G2": {"a": 0.000072, "b": 0.2292, "c": 14.618, "Pmin": 20, "Pmax": 150, "heat_ratio": 0.4, "cv1": 0.15,
           "cv2": 0.15, "cm": 0.75, "Hmax": 450, "ramp_up": 70, "ramp_down": 70},
    "G3": {"a": 0.000072, "b": 0.2292, "c": 14.618, "Pmin": 30, "Pmax": 400, "heat_ratio": 0.3, "cv1": 0.15,
           "cv2": 0.15, "cm": 0.75, "Hmax": 450, "ramp_up": 70, "ramp_down": 70},
    "G4": {"a": 0.000076, "b": 0.2716, "c": 18.822, "Pmin": 30, "Pmax": 400, "heat_ratio": 0.3, "cv1": 0.15,
           "cv2": 0.15, "cm": 0.75, "Hmax": 400, "ramp_up": 80, "ramp_down": 80},
    "G5": {"a": 0.000076, "b": 0.2716, "c": 18.822, "Pmin": 30, "Pmax": 400, "heat_ratio": 0.3, "cv1": 0.15,
           "cv2": 0.15, "cm": 0.75, "Hmax": 400, "ramp_up": 80, "ramp_down": 80},
    "G6": {"a": 0.000076, "b": 0.2716, "c": 18.822, "Pmin": 30, "Pmax": 400, "heat_ratio": 0.3, "cv1": 0.15,
           "cv2": 0.15, "cm": 0.75, "Hmax": 400, "ramp_up": 80, "ramp_down": 80},
    # "G7": {"a": 0.000171, "b": 0.2705, "c": 11.537, "Pmin": 80, "Pmax": 200, "heat_ratio": 0.3, "cv1": 0, "cv2": 0, "cm":0},
    # "G8": {"a": 0.000038, "b": 0.2716, "c": 37.645, "Pmin": 200, "Pmax": 500, "heat_ratio": 0.3, "cv1": 0, "cv2": 0, "cm":0},
}

generators_condensing = {
    "G7": {"a": 0.000171, "b": 0.2705, "c": 11.537, "Pmin": 80, "Pmax": 200, "heat_ratio": 0.3, "cv1": 0, "cv2": 0,
           "cm": 0, "ramp_up": 50, "ramp_down": 50},
    "G8": {"a": 0.000038, "b": 0.2716, "c": 37.645, "Pmin": 200, "Pmax": 500, "heat_ratio": 0.3, "cv1": 0, "cv2": 0,
           "cm": 0, "ramp_up": 130, "ramp_down": 130},
    "Wind": {"a": 0, "b": 0, "c": 0, "Pmin": 0, "Pmax": 10000, "heat_ratio": 0, "cv1": 0,
             "cv2": 0, "cm": 1, "Hmax": 0, "ramp_up": 10000, "ramp_down": 10000},
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
hourly_demand_all = [hourly_demand, hourly_demand[::-1]]

# 生成每小时的负荷都为900
hourly_heat_demand = [1800 for i in range(24)]
# hourly_heat_demand = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240,
#                       250, 260, 270, 280]
hourly_heat_demand_all = [hourly_heat_demand, hourly_heat_demand[::-1]]

hourly_wind_power_available = [255, 233, 194, 186, 202, 190, 181, 217, 223, 235, 255, 260, 268, 270, 269, 250, 241, 258,
                               268, 278, 288, 300, 280, 262]
hourly_wind_power_available_all = [hourly_wind_power_available, hourly_wind_power_available[::-1]]


def economic_dispatch_24h_gurobi(hourly_demand, hourly_heat_demand, hourly_wind_power_available, initial_energy):
    model = gp.Model("EconomicDispatch24h")

    # 创建变量
    power_vars_chp = model.addVars(range(len(hourly_demand)), generators_chp.keys(), lb=0, name="Power_chp")
    power_vars_condensing = model.addVars(range(len(hourly_demand)), generators_condensing.keys(), lb=0,
                                          name="Power_condensing")
    heat_vars_chp = model.addVars(range(len(hourly_demand)), generators_chp.keys(), lb=0, name="Heat_chp")

    storage_charge_vars = model.addVars(range(len(hourly_demand)), lb=0, ub=thermal_storage["charge_capacity"],
                                        name="StorageCharge")
    storage_discharge_vars = model.addVars(range(len(hourly_demand)), lb=0, ub=thermal_storage["discharge_capacity"],
                                           name="StorageDischarge")
    storage_energy_vars = model.addVars(range(len(hourly_demand) + 1), lb=0, ub=thermal_storage["capacity"],
                                        name="StorageEnergy")

    # wind_power_vars = model.addVars(range(len(hourly_demand)), lb=0, ub=hourly_wind_power_available, name="WindPower")

    # 设置目标函数
    obj = gp.quicksum(
        generators_condensing[name]["a"] * power_vars_condensing[hour, name] * power_vars_condensing[hour, name] + generators_condensing[name]["b"] * power_vars_condensing[hour, name] + generators_condensing[name]["c"] for hour in range(len(hourly_demand)) for name in generators_condensing.keys()) \
        + gp.quicksum(generators_chp[name]["a"] * (power_vars_chp[hour, name] + generators_chp[name]["cv1"] * heat_vars_chp[hour, name]) * (power_vars_chp[hour, name] + generators_chp[name]["cv1"] * heat_vars_chp[hour, name]) + generators_chp[name]["b"] * (power_vars_chp[hour, name] + generators_chp[name]["cv1"] * heat_vars_chp[hour, name]) + generators_chp[name]["c"] for hour in range(len(hourly_demand)) for name in generators_chp.keys())  \
        # + gp.quicksum(1 * wind_power_vars[hour] for hour in range(len(hourly_demand)))

    model.setObjective(obj, gp.GRB.MINIMIZE)

    # 添加约束条件
    for hour in range(len(hourly_demand)):
        # 电力平衡
        model.addConstr((gp.quicksum(power_vars_chp[hour, name] for name in generators_chp.keys()) + \
                         gp.quicksum(power_vars_condensing[hour, name] for name in generators_condensing.keys())) == \
                        hourly_demand[hour])


        # 热力平衡
        model.addConstr(
            gp.quicksum(heat_vars_chp[hour, name] for name in generators_chp.keys()) + storage_discharge_vars[hour] \
            - storage_charge_vars[hour] == hourly_heat_demand[hour])

        # 发电机输出范围约束
        for name in generators_condensing.keys():
            model.addConstr(power_vars_condensing[hour, name] >= generators_condensing[name]["Pmin"])
            model.addConstr(power_vars_condensing[hour, name] <= generators_condensing[name]["Pmax"])
            if name == "Wind":
                model.addConstr(power_vars_condensing[hour, name] <= hourly_wind_power_available[hour])
            # 机组爬坡约束
            if hour > 0:
                model.addConstr(power_vars_condensing[hour, name] - power_vars_condensing[hour - 1, name] <=
                                generators_condensing[name]["ramp_up"])
                model.addConstr(power_vars_condensing[hour - 1, name] - power_vars_condensing[hour, name] <=
                                generators_condensing[name]["ramp_down"])
        for name in generators_chp.keys():
            model.addConstr(power_vars_chp[hour, name] >= generators_chp[name]["Pmin"])
            model.addConstr(power_vars_chp[hour, name] <= generators_chp[name]["Pmax"])
            model.addConstr(heat_vars_chp[hour, name] >= 0)
            model.addConstr(heat_vars_chp[hour, name] <= generators_chp[name]["Hmax"])
            # 机组爬坡约束
            if hour > 0:
                model.addConstr(
                    power_vars_chp[hour, name] - power_vars_chp[hour - 1, name] <= generators_chp[name]["ramp_up"])
                model.addConstr(
                    power_vars_chp[hour - 1, name] - power_vars_chp[hour, name] <= generators_chp[name]["ramp_down"])
            # 热电联供约束
            model.addConstr(power_vars_chp[hour, name] <= generators_chp[name]["Pmax"] - generators_chp[name]["cv1"] *
                            heat_vars_chp[hour, name])
            # model.addConstr(power_vars_chp[hour, name] >= gp.min_(generators_chp[name]["Pmin"] - generators_chp[name]["cv1"] * heat_vars_chp[hour, name], generators_chp[name]["cm"] * heat_vars_chp[hour, name]))
            aux_var = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name=f"aux_var_{hour}_{name}")
            model.addConstr(
                aux_var <= generators_chp[name]["Pmin"] - generators_chp[name]["cv1"] * heat_vars_chp[hour, name])
            model.addConstr(aux_var <= generators_chp[name]["cm"] * heat_vars_chp[hour, name])
            model.addConstr(power_vars_chp[hour, name] >= aux_var)

        # 储热设备充放电约束
        model.addConstr(storage_energy_vars[hour] - storage_energy_vars[hour + 1] == storage_discharge_vars[hour] -
                        storage_charge_vars[hour])
        # 储热设备容量约束
        model.addConstr(storage_energy_vars[hour] <= thermal_storage["capacity"])
        model.addConstr(storage_energy_vars[hour] >= 0)
        # 风电出力约束
        # model.addConstr(wind_power_vars[hour] <= hourly_wind_power_available[hour])

    # 首尾储热量相等约束
    model.addConstr(storage_energy_vars[0] == storage_energy_vars[len(hourly_demand)])

    # 初始储热量约束
    model.addConstr(storage_energy_vars[0] == initial_energy)

    # 求解模型
    model.optimize()

    if model.status == gp.GRB.OPTIMAL:
        results = {
            "power_chp": {(hour, name): power_vars_chp[hour, name].x for hour in range(len(hourly_demand)) for name in
                          generators_chp.keys()},
            "wind_power": {hour: power_vars_condensing[hour, "Wind"].x for hour in range(len(hourly_demand))},
            "power_condensing": {(hour, name): power_vars_condensing[hour, name].x for hour in range(len(hourly_demand))
                                 for name in generators_condensing.keys()},
            "heat_chp": {(hour, name): heat_vars_chp[hour, name].x for hour in range(len(hourly_demand)) for name in
                         generators_chp.keys()},
            "storage_charge": {hour: storage_charge_vars[hour].x for hour in range(len(hourly_demand))},
            "storage_discharge": {hour: storage_discharge_vars[hour].x for hour in range(len(hourly_demand))},
            "storage_energy": {hour: storage_energy_vars[hour].x for hour in range(len(hourly_demand) + 1)},
            # "wind_power": {hour: wind_power_vars[hour].x for hour in range(len(hourly_demand))},
            "total_cost": model.objVal,
            "status": model.status
        }
        return results
    else:
        return None


def save_results_to_csv(results, filename):
    chp_power_data = results["power_chp"]
    condensing_power_data = results["power_condensing"]
    chp_heat_data = results["heat_chp"]
    storage_charge_data = results["storage_charge"]
    storage_discharge_data = results["storage_discharge"]
    storage_energy_data = results["storage_energy"]
    wind_power_data = results["wind_power"]
    optimization_status = results["status"]

    data = []
    for hour in range(len(hourly_demand)):
        row = {"hour": hour}
        for name in generators_chp.keys():
            row[name] = chp_power_data[hour, name]
            row[name + "_heat"] = chp_heat_data[hour, name]
        for name in generators_condensing.keys():
            row[name] = condensing_power_data[hour, name]
        row["storage_charge"] = storage_charge_data[hour]
        row["storage_discharge"] = storage_discharge_data[hour]
        row["storage_energy"] = storage_energy_data[hour]
        row["wind_power"] = wind_power_data[hour]
        row["status"] = optimization_status
        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)


# 使用新函数绘制储热设备的充放电和储热量
def plot_storage(results):
    storage_charge_data = results["storage_charge"]
    storage_discharge_data = results["storage_discharge"]
    storage_energy_data = results["storage_energy"]

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


def results_process_chp_power(results):
    # 使用 defaultdict 初始化一个空字典，使默认值为一个空列表
    classified_data = defaultdict(list)
    power_chp = results["power_chp"]
    # 遍历给定的数据字典
    for key, value in power_chp.items():
        hour, generator_type = key
        classified_data[generator_type].append((hour, value))
    # 将 defaultdict 转换回普通字典
    classified_data = dict(classified_data)
    return classified_data

def results_process_chp_heat(results):
    # 使用 defaultdict 初始化一个空字典，使默认值为一个空列表
    classified_data = defaultdict(list)
    heat_chp = results["heat_chp"]
    # 遍历给定的数据字典
    for key, value in heat_chp.items():
        hour, generator_type = key
        classified_data[generator_type].append((hour, value))
    # 将 defaultdict 转换回普通字典
    classified_data = dict(classified_data)
    return classified_data


def results_process_condensing(results):
    # 使用 defaultdict 初始化一个空字典，使默认值为一个空列表
    classified_data = defaultdict(list)
    power_condensing = results["power_condensing"]
    # 遍历给定的数据字典
    for key, value in power_condensing.items():
        hour, generator_type = key
        classified_data[generator_type].append((hour, value))
    # 将 defaultdict 转换回普通字典
    classified_data = dict(classified_data)
    return classified_data


def plot_wind_power(results):
    # wind_power_data = results["wind_power"]
    classified_data = results_process_condensing(results)
    wind_power = [heat for _, heat in classified_data['Wind']]
    wind_power_forecast = hourly_wind_power_available

    hours = range(len(hourly_demand))
    plt.plot(hours, [wind_power[hour] for hour in hours], label="wind_power", linestyle="-.")
    plt.plot(hours, [wind_power_forecast[hour] for hour in hours], label="wind_power_forecast", linestyle="--")

    plt.xlabel("Hour")
    plt.ylabel("Power (MW)")
    plt.title("Wind Power Output")
    plt.legend()
    plt.grid()
    plt.show()

def plot_heat(results):
    classified_data_chp = results_process_chp_heat(results)

    # 提取各发电机的出力数据
    g1_heat = [heat for _, heat in classified_data_chp['G1']]
    g2_heat = [heat for _, heat in classified_data_chp['G2']]
    g3_heat = [heat for _, heat in classified_data_chp['G3']]
    g4_heat = [heat for _, heat in classified_data_chp['G4']]
    g5_heat = [heat for _, heat in classified_data_chp['G5']]
    g6_heat = [heat for _, heat in classified_data_chp['G6']]

    # 储热放热数据
    storage_discharge_data = results["storage_discharge"]

    hours = [hour for hour, _ in classified_data_chp['G1']]

    # 绘制堆叠柱状图
    plt.bar(hours, g1_heat, label="G1_heat")
    plt.bar(hours, g2_heat, bottom=g1_heat, label="G2_heat")
    plt.bar(hours, g3_heat, bottom=[g1_heat[i] + g2_heat[i] for i in range(len(g1_heat))], label="G3_heat")
    plt.bar(hours, g4_heat, bottom=[g1_heat[i] + g2_heat[i] + g3_heat[i] for i in range(len(g1_heat))], label="G4_heat")
    plt.bar(hours, g5_heat, bottom=[g1_heat[i] + g2_heat[i] + g3_heat[i] + g4_heat[i] for i in range(len(g1_heat))], label="G5_heat")
    plt.bar(hours, g6_heat, bottom=[g1_heat[i] + g2_heat[i] + g3_heat[i] + g4_heat[i] + g5_heat[i] for i in range(len(g1_heat))], label="G6_heat")
    plt.bar(hours, storage_discharge_data, bottom=[g1_heat[i] + g2_heat[i] + g3_heat[i] + g4_heat[i] + g5_heat[i] + g6_heat[i] for i in range(len(g1_heat))], label="storage_discharge")

    plt.plot(hours, g1_heat, label="G1_heat", linestyle="-.")


    plt.xlabel("Hour")
    plt.ylabel("Heat (MW)")
    plt.title("CHP Heat Output")
    plt.legend()
    plt.grid()
    plt.show()


def plot_power(results):
    classified_data_chp = results_process_chp_power(results)
    classified_data_condensing = results_process_condensing(results)
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

    # 添加图例和轴标签
    plt.xlabel('Hours')
    plt.ylabel('Output')
    plt.legend()

    # 显示图形
    plt.show()


if __name__ == "__main__":
    for day in range(1):
        # 24小时的经济调度
        results = economic_dispatch_24h_gurobi(hourly_demand_all[day], hourly_heat_demand_all[day],
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
        save_results_to_csv(results, filename)
        # plot_power(results)
        # plot_wind_power(results)
        plot_storage(resultsw)
        # plot_heat(results)