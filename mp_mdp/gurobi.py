import gurobipy as gp
from parameters import *


def economic_dispatch_continuous_optimal(hourly_demand, hourly_heat_demand, wind_scenarios, probabilities):
    """
    仅要求储热罐的状态是连续的，并在总的调度周期内约束储热罐的末容量状态
    """

    model = gp.Model("EconomicDispatchcontinuous")
    model.setParam('OutputFlag', 0)

    # 创建变量
    power_vars_chp = model.addVars(range(len(hourly_demand)), generators_chp.keys(), range(len(wind_scenarios)), lb=0,
                                   name="Power_chp")
    heat_vars_chp = model.addVars(range(len(hourly_demand)), generators_chp.keys(), range(len(wind_scenarios)), lb=0,
                                  name="Heat_chp")
    power_vars_condensing = model.addVars(range(len(hourly_demand)), generators_condensing.keys(),
                                          range(len(wind_scenarios)), lb=0,
                                          name="Power_condensing")
    heat_charge_vars = model.addVars(range(len(hourly_demand)), thermal_storage.keys(), range(len(wind_scenarios)),
                                     lb=0,
                                     name="HeatCharge")
    heat_discharge_vars = model.addVars(range(len(hourly_demand)), thermal_storage.keys(), range(len(wind_scenarios)),
                                        lb=0,
                                        name="HeatDischarge")
    storage_heat_vars = model.addVars(range(len(hourly_demand) + 1), thermal_storage.keys(), range(len(wind_scenarios)),
                                      lb=0,
                                      name="StorageHeat")
    charge_indicator_vars = model.addVars(range(len(hourly_demand)), thermal_storage.keys(), range(len(wind_scenarios)),
                                          vtype=gp.GRB.BINARY,
                                          name="ChargeIndicator")
    discharge_indicator_vars = model.addVars(range(len(hourly_demand)), thermal_storage.keys(),
                                             range(len(wind_scenarios)), vtype=gp.GRB.BINARY,
                                             name="DischargeIndicator")
    wind_power_vars = model.addVars(range(len(hourly_demand)), range(len(wind_scenarios)), lb=0, name="WindPower")

    # 设置目标函数
    expected_total_cost = 0
    # 遍历所有场景
    for scenario_idx, (wind_scenario, scenario_probability) in enumerate(zip(wind_scenarios, probabilities)):
        scenario_cost = 0
        for hour in range(len(hourly_demand)):
            for name in generators_chp.keys():
                scenario_cost += (heat_vars_chp[hour, name, scenario_idx] + power_vars_chp[
                    hour, name, scenario_idx]) * (heat_vars_chp[hour, name, scenario_idx] + power_vars_chp[
                    hour, name, scenario_idx]) * generators_chp[name]["a"] + \
                                 (heat_vars_chp[hour, name, scenario_idx] + power_vars_chp[hour, name, scenario_idx]) * \
                                 generators_chp[name]["b"] + \
                                 generators_chp[name]["c"]
            for name in generators_condensing.keys():
                scenario_cost += power_vars_condensing[hour, name, scenario_idx] * power_vars_condensing[
                    hour, name, scenario_idx] * generators_condensing[name]["a"] + \
                                 power_vars_condensing[hour, name, scenario_idx] * generators_condensing[name]["b"] + \
                                 generators_condensing[name]["c"]
            # scenario_cost += wind_power_vars[hour, scenario_idx] * 0.33  # 风电的成本 (0.33的值会产生弃风现象)

            # scenario_cost += storage_charge_vars[hour, scenario_idx] * thermal_storage["charge_cost"]
            # scenario_cost += storage_discharge_vars[hour, scenario_idx] * thermal_storage["discharge_cost"]

            # 将每个场景的目标函数乘以相应的概率，并将结果累加到期望总成本中
        # expected_total_cost += scenario_probability * scenario_cost
        expected_total_cost += 1 * scenario_cost

    model.setObjective(expected_total_cost, gp.GRB.MINIMIZE)

    # 添加约束条件
    for scenario_idx, wind_scenario in enumerate(wind_scenarios):

        for hour in range(len(hourly_demand)):
            # 电力平衡
            model.addConstr(
                gp.quicksum(power_vars_chp[hour, name, scenario_idx] for name in generators_chp.keys())
                + gp.quicksum(power_vars_condensing[hour, name, scenario_idx] for name in generators_condensing.keys())
                + wind_power_vars[hour, scenario_idx]
                == hourly_demand[hour],
                name=f"PowerBalance_hour{hour}_scenario{scenario_idx}"
            )

            # 热力平衡

            model.addConstr(
                gp.quicksum(heat_vars_chp[hour, name, scenario_idx] for name in ['G1', 'G2', 'G3'])
                + heat_discharge_vars[hour, 'a', scenario_idx]
                == hourly_heat_demand[hour] + heat_charge_vars[hour, 'a', scenario_idx],
                name=f"HeatBalance_hour{hour}_scenario{scenario_idx}"
            )

            model.addConstr(
                gp.quicksum(heat_vars_chp[hour, name, scenario_idx] for name in ['G4', 'G5', 'G6'])
                + heat_discharge_vars[hour, 'b', scenario_idx]
                == hourly_heat_demand[hour] + heat_charge_vars[hour, 'b', scenario_idx],
                name=f"HeatBalance_hour{hour}_scenario{scenario_idx}"
            )

            # 热电联产发电机的热电输出关系
            for name in generators_chp.keys():
                # # 选择两个中的较小值
                # aux_var = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY,
                #                        name=f"aux_var_{hour}_{name}_{scenario_idx}")
                # model.addConstr(
                #     aux_var <= generators_chp[name]["Pmin"] - generators_chp[name]["cv1"] * heat_vars_chp[
                #         hour, name, scenario_idx])
                # model.addConstr(aux_var <= generators_chp[name]["cm"] * heat_vars_chp[hour, name, scenario_idx])

                model.addConstr(power_vars_chp[hour, name, scenario_idx] >= generators_chp[name]["cm"] * heat_vars_chp[
                    hour, name, scenario_idx])

                # if generators_chp[name]["Pmin"] - generators_chp[name]["cv1"] * heat_vars_chp[hour, name, scenario_idx] > heat_vars_chp[hour, name, scenario_idx] * generators_chp[name]["cm"]:
                #     model.addConstr(power_vars_chp[hour, name, scenario_idx] >= heat_vars_chp[hour, name, scenario_idx] * generators_chp[name]["cm"])
                # else:
                #     model.addConstr(power_vars_chp[hour, name, scenario_idx] >= generators_chp[name]["Pmin"] - generators_chp[name]["cv1"] * heat_vars_chp[hour, name, scenario_idx])
                model.addConstr(
                    power_vars_chp[hour, name, scenario_idx] <= generators_chp[name]["Pmax"] - generators_chp[name][
                        "cv1"] * heat_vars_chp[hour, name, scenario_idx])

            # 发电机组与热电机组的出力约束
            for name in generators_chp.keys():
                model.addConstr(
                    power_vars_chp[hour, name, scenario_idx]
                    <= generators_chp[name]["Pmax"],
                    name=f"CHP_max_power_hour{hour}_generator{name}_scenario{scenario_idx}"
                )
                model.addConstr(
                    power_vars_chp[hour, name, scenario_idx]
                    >= generators_chp[name]["Pmin"],
                    name=f"CHP_min_power_hour{hour}_generator{name}_scenario{scenario_idx}"
                )
                model.addConstr(
                    heat_vars_chp[hour, name, scenario_idx]
                    <= generators_chp[name]["Hmax"],
                    name=f"CHP_max_heat_hour{hour}_generator{name}_scenario{scenario_idx}"
                )

                if hour == 0:
                    pass
                    # model.addConstr(
                    #     power_vars_chp[hour, name, scenario_idx]
                    #     - generators_chp[name]["Pmin"]
                    #     <= generators_chp[name]["ramp_up"],
                    #     name=f"CHP_ramp_up_hour{hour}_generator{name}_scenario{scenario_idx}"
                    # )
                    # model.addConstr(
                    #     generators_chp[name]["Pmin"]
                    #     - power_vars_chp[hour, name, scenario_idx]
                    #     <= generators_chp[name]["ramp_down"],
                    #     name=f"CHP_ramp_down_hour{hour}_generator{name}_scenario{scenario_idx}"
                    # )
                else:
                    model.addConstr(
                        power_vars_chp[hour, name, scenario_idx]
                        - power_vars_chp[hour - 1, name, scenario_idx]
                        <= generators_chp[name]["ramp_up"],
                        name=f"CHP_ramp_up_hour{hour}_generator{name}_scenario{scenario_idx}"
                    )
                    model.addConstr(
                        power_vars_chp[hour - 1, name, scenario_idx]
                        - power_vars_chp[hour, name, scenario_idx]
                        <= generators_chp[name]["ramp_down"],
                        name=f"CHP_ramp_down_hour{hour}_generator{name}_scenario{scenario_idx}"
                    )

            for name in generators_condensing.keys():
                model.addConstr(
                    power_vars_condensing[hour, name, scenario_idx]
                    <= generators_condensing[name]["Pmax"],
                    name=f"Condensing_max_power_hour{hour}_generator{name}_scenario{scenario_idx}"
                )
                model.addConstr(
                    power_vars_condensing[hour, name, scenario_idx]
                    >= generators_condensing[name]["Pmin"],
                    name=f"Condensing_min_power_hour{hour}_generator{name}_scenario{scenario_idx}"
                )
                if hour == 0:
                    pass
                    # model.addConstr(
                    #     power_vars_condensing[hour, name, scenario_idx]
                    #     - generators_condensing[name]["Pmin"]
                    #     <= generators_condensing[name]["ramp_up"],
                    #     name=f"Condensing_ramp_up_hour{hour}_generator{name}_scenario{scenario_idx}"
                    # )
                    # model.addConstr(
                    #     generators_condensing[name]["Pmin"]
                    #     - power_vars_condensing[hour, name, scenario_idx]
                    #     <= generators_condensing[name]["ramp_down"],
                    #     name=f"Condensing_ramp_down_hour{hour}_generator{name}_scenario{scenario_idx}"
                    # )
                else:
                    model.addConstr(
                        power_vars_condensing[hour, name, scenario_idx]
                        - power_vars_condensing[hour - 1, name, scenario_idx]
                        <= generators_condensing[name]["ramp_up"],
                        name=f"Condensing_ramp_up_hour{hour}_generator{name}_scenario{scenario_idx}"
                    )
                    model.addConstr(
                        power_vars_condensing[hour - 1, name, scenario_idx]
                        - power_vars_condensing[hour, name, scenario_idx]
                        <= generators_condensing[name]["ramp_down"],
                        name=f"Condensing_ramp_down_hour{hour}_generator{name}_scenario{scenario_idx}"
                    )

            # 储热罐的约束
            for name in thermal_storage.keys():
                model.addConstr(
                    storage_heat_vars[hour + 1, name, scenario_idx] - storage_heat_vars[hour, name, scenario_idx]
                    == heat_charge_vars[hour, name, scenario_idx] - heat_discharge_vars[hour, name, scenario_idx],
                    name=f"StorageEnergy_hour{hour}_scenario{scenario_idx}_HST{name}")
                # 储热罐的容量约束和充放热约束
                model.addConstr(storage_heat_vars[hour, name, scenario_idx] <= thermal_storage[name]["capacity"],
                                name=f"StorageEnergy_max_hour{hour}_scenario{scenario_idx}_HST{name}")
                model.addConstr(storage_heat_vars[hour, name, scenario_idx] >= 0,
                                name=f"StorageEnergy_min_hour{hour}_scenario{scenario_idx}_HST{name}")
                model.addConstr(heat_charge_vars[hour, name, scenario_idx] <= thermal_storage[name]["charge_capacity"],
                                name=f"StorageCharge_max_hour{hour}_scenario{scenario_idx}_HST{name}")
                model.addConstr(
                    heat_discharge_vars[hour, name, scenario_idx] <= thermal_storage[name]["discharge_capacity"],
                    name=f"StorageDischarge_max_hour{hour}_scenario{scenario_idx}_HST{name}")
                # 不允许同时充放电
                model.addConstr(heat_charge_vars[hour, name, scenario_idx] <= thermal_storage[name]["charge_capacity"] *
                                charge_indicator_vars[hour, name, scenario_idx],
                                name=f"StorageCharge_max_hour{hour}_scenario{scenario_idx}_HST{name}")
                model.addConstr(
                    heat_discharge_vars[hour, name, scenario_idx] <= thermal_storage[name]["discharge_capacity"] *
                    discharge_indicator_vars[hour, name, scenario_idx],
                    name=f"StorageDischarge_max_hour{hour}_scenario{scenario_idx}_HST{name}")
                model.addConstr(charge_indicator_vars[hour, name, scenario_idx] + discharge_indicator_vars[
                    hour, name, scenario_idx] <= 1,
                                name=f"StorageChargeDischarge_{hour}_scenario{scenario_idx}_HST{name}")

            # 风电的约束
            model.addConstr(wind_power_vars[hour, scenario_idx] <= wind_scenario[hour],
                            name=f"WindPowerConstraint_hour{hour}_scenario{scenario_idx}")

        # 储热罐的始末状态约束
        for name in thermal_storage.keys():
            if scenario_idx == 0:
                model.addConstr(storage_heat_vars[0, name, scenario_idx] == thermal_storage[name]["initial_heat"],
                                name=f"StorageEnergy_initial_scenario{scenario_idx}_HST{name}")
                # model.addConstr(storage_heat_vars[len(hourly_demand), name, scenario_idx] == thermal_storage[name]["initial_heat"],
                #                 name=f"StorageEnergy_final_scenario{scenario_idx}_HST{name}")

            elif scenario_idx == 2:
                model.addConstr(
                    storage_heat_vars[len(hourly_demand), name, scenario_idx] == thermal_storage[name]["initial_heat"],
                    name=f"StorageEnergy_final_scenario{scenario_idx}_HST{name}")
                model.addConstr(
                    storage_heat_vars[0, name, scenario_idx] == storage_heat_vars[
                        len(hourly_demand), name, scenario_idx - 1],
                    name=f"StorageEnergy_initial_scenario{scenario_idx}_HST{name}")

            else:
                model.addConstr(
                    storage_heat_vars[0, name, scenario_idx] == storage_heat_vars[
                        len(hourly_demand), name, scenario_idx - 1],
                    name=f"StorageEnergy_initial_scenario{scenario_idx}_HST{name}")

    # 求解模型
    model.optimize()

    # 检查模型是否找到了最优解
    if model.Status == gp.GRB.Status.OPTIMAL:
        print("找到最优解")

        # 提取解决方案
        solution_chp = {(hour, name, idx): power_vars_chp[hour, name, idx].x for hour in range(len(hourly_demand))
                        for name in generators_chp.keys() for idx in range(len(wind_scenarios))}
        solution_condensing = {(hour, name, idx): power_vars_condensing[hour, name, idx].x for hour in
                               range(len(hourly_demand))
                               for name in generators_condensing.keys() for idx in range(len(wind_scenarios))}
        solution_heat_chp = {(hour, name, idx): heat_vars_chp[hour, name, idx].x for hour in range(len(hourly_demand))
                             for name in generators_chp.keys() for idx in range(len(wind_scenarios))}
        solution_storage_charge = {(hour, name, idx): heat_charge_vars[hour, name, idx].x for hour in
                                   range(len(hourly_demand))
                                   for name in thermal_storage.keys() for idx in range(len(wind_scenarios))}
        solution_storage_discharge = {(hour, name, idx): heat_discharge_vars[hour, name, idx].x for hour in
                                      range(len(hourly_demand))
                                      for name in thermal_storage.keys() for idx in range(len(wind_scenarios))}
        solution_storage_energy = {(hour, name, idx): storage_heat_vars[hour, name, idx].x for hour in
                                   range(len(hourly_demand) + 1)
                                   for name in thermal_storage.keys() for idx in range(len(wind_scenarios))}
        solution_wind_power = {(hour, idx): wind_power_vars[hour, idx].x for hour in range(len(hourly_demand))
                               for idx in range(len(wind_scenarios))}

        # 将解决方案以字典形式返回
        results = {
            "CHP": solution_chp,
            "Condensing": solution_condensing,
            "Heat_CHP": solution_heat_chp,
            "Storage_Charge": solution_storage_charge,
            "Storage_Discharge": solution_storage_discharge,
            "Storage_Energy": solution_storage_energy,
            "Wind_Power": solution_wind_power,
        }

    else:
        print("未找到最优解")
        results = None
    # 返回最优值和解决方案
    return model.objVal, results


def economic_dispatch_continuous_gurobi(hourly_demand, hourly_heat_demand, wind_scenarios, probabilities):
    """
    不对储热罐的始末状态进行优化 强制约束等于初始状态（即只需要确定一个初始值）
    """

    model = gp.Model("EconomicDispatchcontinuous")
    model.setParam('OutputFlag', 0)

    # 创建变量
    power_vars_chp = model.addVars(range(len(hourly_demand)), generators_chp.keys(), range(len(wind_scenarios)), lb=0,
                                   name="Power_chp")
    heat_vars_chp = model.addVars(range(len(hourly_demand)), generators_chp.keys(), range(len(wind_scenarios)), lb=0,
                                  name="Heat_chp")
    power_vars_condensing = model.addVars(range(len(hourly_demand)), generators_condensing.keys(),
                                          range(len(wind_scenarios)), lb=0,
                                          name="Power_condensing")
    heat_charge_vars = model.addVars(range(len(hourly_demand)), thermal_storage.keys(), range(len(wind_scenarios)),
                                     lb=0,
                                     name="HeatCharge")
    heat_discharge_vars = model.addVars(range(len(hourly_demand)), thermal_storage.keys(), range(len(wind_scenarios)),
                                        lb=0,
                                        name="HeatDischarge")
    storage_heat_vars = model.addVars(range(len(hourly_demand) + 1), thermal_storage.keys(), range(len(wind_scenarios)),
                                      lb=0,
                                      name="StorageHeat")
    charge_indicator_vars = model.addVars(range(len(hourly_demand)), thermal_storage.keys(), range(len(wind_scenarios)),
                                          vtype=gp.GRB.BINARY,
                                          name="ChargeIndicator")
    discharge_indicator_vars = model.addVars(range(len(hourly_demand)), thermal_storage.keys(),
                                             range(len(wind_scenarios)), vtype=gp.GRB.BINARY,
                                             name="DischargeIndicator")
    wind_power_vars = model.addVars(range(len(hourly_demand)), range(len(wind_scenarios)), lb=0, name="WindPower")

    # 设置目标函数
    expected_total_cost = 0
    # 遍历所有场景
    for scenario_idx, (wind_scenario, scenario_probability) in enumerate(zip(wind_scenarios, probabilities)):
        scenario_cost = 0
        for hour in range(len(hourly_demand)):
            for name in generators_chp.keys():
                scenario_cost += (heat_vars_chp[hour, name, scenario_idx] + power_vars_chp[
                    hour, name, scenario_idx]) * (heat_vars_chp[hour, name, scenario_idx] + power_vars_chp[
                    hour, name, scenario_idx]) * generators_chp[name]["a"] + \
                                 (heat_vars_chp[hour, name, scenario_idx] + power_vars_chp[hour, name, scenario_idx]) * \
                                 generators_chp[name]["b"] + \
                                 generators_chp[name]["c"]
            for name in generators_condensing.keys():
                scenario_cost += power_vars_condensing[hour, name, scenario_idx] * power_vars_condensing[
                    hour, name, scenario_idx] * generators_condensing[name]["a"] + \
                                 power_vars_condensing[hour, name, scenario_idx] * generators_condensing[name]["b"] + \
                                 generators_condensing[name]["c"]
            # scenario_cost += wind_power_vars[hour, scenario_idx] * 0.33  # 风电的成本 (0.33的值会产生弃风现象)

            # scenario_cost += storage_charge_vars[hour, scenario_idx] * thermal_storage["charge_cost"]
            # scenario_cost += storage_discharge_vars[hour, scenario_idx] * thermal_storage["discharge_cost"]

            # 将每个场景的目标函数乘以相应的概率，并将结果累加到期望总成本中
        # expected_total_cost += scenario_probability * scenario_cost
        expected_total_cost += 1 * scenario_cost

    model.setObjective(expected_total_cost, gp.GRB.MINIMIZE)

    # 添加约束条件
    for scenario_idx, wind_scenario in enumerate(wind_scenarios):

        for hour in range(len(hourly_demand)):
            # 电力平衡
            model.addConstr(
                gp.quicksum(power_vars_chp[hour, name, scenario_idx] for name in generators_chp.keys())
                + gp.quicksum(power_vars_condensing[hour, name, scenario_idx] for name in generators_condensing.keys())
                + wind_power_vars[hour, scenario_idx]
                == hourly_demand[hour],
                name=f"PowerBalance_hour{hour}_scenario{scenario_idx}"
            )

            # 热力平衡

            model.addConstr(
                gp.quicksum(heat_vars_chp[hour, name, scenario_idx] for name in ['G1', 'G2', 'G3'])
                + heat_discharge_vars[hour, 'a', scenario_idx]
                == hourly_heat_demand[hour] + heat_charge_vars[hour, 'a', scenario_idx],
                name=f"HeatBalance_hour{hour}_scenario{scenario_idx}"
            )

            model.addConstr(
                gp.quicksum(heat_vars_chp[hour, name, scenario_idx] for name in ['G4', 'G5', 'G6'])
                + heat_discharge_vars[hour, 'b', scenario_idx]
                == hourly_heat_demand[hour] + heat_charge_vars[hour, 'b', scenario_idx],
                name=f"HeatBalance_hour{hour}_scenario{scenario_idx}"
            )

            # 热电联产发电机的热电输出关系
            for name in generators_chp.keys():
                # # 选择两个中的较小值
                # aux_var = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY,
                #                        name=f"aux_var_{hour}_{name}_{scenario_idx}")
                # model.addConstr(
                #     aux_var <= generators_chp[name]["Pmin"] - generators_chp[name]["cv1"] * heat_vars_chp[
                #         hour, name, scenario_idx])
                # model.addConstr(aux_var <= generators_chp[name]["cm"] * heat_vars_chp[hour, name, scenario_idx])

                model.addConstr(power_vars_chp[hour, name, scenario_idx] >= generators_chp[name]["cm"] * heat_vars_chp[
                    hour, name, scenario_idx])

                # if generators_chp[name]["Pmin"] - generators_chp[name]["cv1"] * heat_vars_chp[hour, name, scenario_idx] > heat_vars_chp[hour, name, scenario_idx] * generators_chp[name]["cm"]:
                #     model.addConstr(power_vars_chp[hour, name, scenario_idx] >= heat_vars_chp[hour, name, scenario_idx] * generators_chp[name]["cm"])
                # else:
                #     model.addConstr(power_vars_chp[hour, name, scenario_idx] >= generators_chp[name]["Pmin"] - generators_chp[name]["cv1"] * heat_vars_chp[hour, name, scenario_idx])
                model.addConstr(
                    power_vars_chp[hour, name, scenario_idx] <= generators_chp[name]["Pmax"] - generators_chp[name][
                        "cv1"] * heat_vars_chp[hour, name, scenario_idx])

            # 发电机组与热电机组的出力约束
            for name in generators_chp.keys():
                model.addConstr(
                    power_vars_chp[hour, name, scenario_idx]
                    <= generators_chp[name]["Pmax"],
                    name=f"CHP_max_power_hour{hour}_generator{name}_scenario{scenario_idx}"
                )
                model.addConstr(
                    power_vars_chp[hour, name, scenario_idx]
                    >= generators_chp[name]["Pmin"],
                    name=f"CHP_min_power_hour{hour}_generator{name}_scenario{scenario_idx}"
                )
                model.addConstr(
                    heat_vars_chp[hour, name, scenario_idx]
                    <= generators_chp[name]["Hmax"],
                    name=f"CHP_max_heat_hour{hour}_generator{name}_scenario{scenario_idx}"
                )

                if hour == 0:
                    pass
                    # model.addConstr(
                    #     power_vars_chp[hour, name, scenario_idx]
                    #     - generators_chp[name]["Pmin"]
                    #     <= generators_chp[name]["ramp_up"],
                    #     name=f"CHP_ramp_up_hour{hour}_generator{name}_scenario{scenario_idx}"
                    # )
                    # model.addConstr(
                    #     generators_chp[name]["Pmin"]
                    #     - power_vars_chp[hour, name, scenario_idx]
                    #     <= generators_chp[name]["ramp_down"],
                    #     name=f"CHP_ramp_down_hour{hour}_generator{name}_scenario{scenario_idx}"
                    # )
                else:
                    model.addConstr(
                        power_vars_chp[hour, name, scenario_idx]
                        - power_vars_chp[hour - 1, name, scenario_idx]
                        <= generators_chp[name]["ramp_up"],
                        name=f"CHP_ramp_up_hour{hour}_generator{name}_scenario{scenario_idx}"
                    )
                    model.addConstr(
                        power_vars_chp[hour - 1, name, scenario_idx]
                        - power_vars_chp[hour, name, scenario_idx]
                        <= generators_chp[name]["ramp_down"],
                        name=f"CHP_ramp_down_hour{hour}_generator{name}_scenario{scenario_idx}"
                    )

            for name in generators_condensing.keys():
                model.addConstr(
                    power_vars_condensing[hour, name, scenario_idx]
                    <= generators_condensing[name]["Pmax"],
                    name=f"Condensing_max_power_hour{hour}_generator{name}_scenario{scenario_idx}"
                )
                model.addConstr(
                    power_vars_condensing[hour, name, scenario_idx]
                    >= generators_condensing[name]["Pmin"],
                    name=f"Condensing_min_power_hour{hour}_generator{name}_scenario{scenario_idx}"
                )
                if hour == 0:
                    pass
                    # model.addConstr(
                    #     power_vars_condensing[hour, name, scenario_idx]
                    #     - generators_condensing[name]["Pmin"]
                    #     <= generators_condensing[name]["ramp_up"],
                    #     name=f"Condensing_ramp_up_hour{hour}_generator{name}_scenario{scenario_idx}"
                    # )
                    # model.addConstr(
                    #     generators_condensing[name]["Pmin"]
                    #     - power_vars_condensing[hour, name, scenario_idx]
                    #     <= generators_condensing[name]["ramp_down"],
                    #     name=f"Condensing_ramp_down_hour{hour}_generator{name}_scenario{scenario_idx}"
                    # )
                else:
                    model.addConstr(
                        power_vars_condensing[hour, name, scenario_idx]
                        - power_vars_condensing[hour - 1, name, scenario_idx]
                        <= generators_condensing[name]["ramp_up"],
                        name=f"Condensing_ramp_up_hour{hour}_generator{name}_scenario{scenario_idx}"
                    )
                    model.addConstr(
                        power_vars_condensing[hour - 1, name, scenario_idx]
                        - power_vars_condensing[hour, name, scenario_idx]
                        <= generators_condensing[name]["ramp_down"],
                        name=f"Condensing_ramp_down_hour{hour}_generator{name}_scenario{scenario_idx}"
                    )

            # 储热罐的约束
            for name in thermal_storage.keys():
                model.addConstr(
                    storage_heat_vars[hour + 1, name, scenario_idx] - storage_heat_vars[hour, name, scenario_idx]
                    == heat_charge_vars[hour, name, scenario_idx] - heat_discharge_vars[hour, name, scenario_idx],
                    name=f"StorageEnergy_hour{hour}_scenario{scenario_idx}_HST{name}")
                # 储热罐的容量约束和充放热约束
                model.addConstr(storage_heat_vars[hour, name, scenario_idx] <= thermal_storage[name]["capacity"],
                                name=f"StorageEnergy_max_hour{hour}_scenario{scenario_idx}_HST{name}")
                model.addConstr(storage_heat_vars[hour, name, scenario_idx] >= 0,
                                name=f"StorageEnergy_min_hour{hour}_scenario{scenario_idx}_HST{name}")
                model.addConstr(heat_charge_vars[hour, name, scenario_idx] <= thermal_storage[name]["charge_capacity"],
                                name=f"StorageCharge_max_hour{hour}_scenario{scenario_idx}_HST{name}")
                model.addConstr(
                    heat_discharge_vars[hour, name, scenario_idx] <= thermal_storage[name]["discharge_capacity"],
                    name=f"StorageDischarge_max_hour{hour}_scenario{scenario_idx}_HST{name}")
                # 不允许同时充放电
                model.addConstr(heat_charge_vars[hour, name, scenario_idx] <= thermal_storage[name]["charge_capacity"] *
                                charge_indicator_vars[hour, name, scenario_idx],
                                name=f"StorageCharge_max_hour{hour}_scenario{scenario_idx}_HST{name}")
                model.addConstr(
                    heat_discharge_vars[hour, name, scenario_idx] <= thermal_storage[name]["discharge_capacity"] *
                    discharge_indicator_vars[hour, name, scenario_idx],
                    name=f"StorageDischarge_max_hour{hour}_scenario{scenario_idx}_HST{name}")
                model.addConstr(charge_indicator_vars[hour, name, scenario_idx] + discharge_indicator_vars[
                    hour, name, scenario_idx] <= 1,
                                name=f"StorageChargeDischarge_{hour}_scenario{scenario_idx}_HST{name}")

            # 风电的约束
            model.addConstr(wind_power_vars[hour, scenario_idx] <= wind_scenario[hour],
                            name=f"WindPowerConstraint_hour{hour}_scenario{scenario_idx}")

        # 储热罐的始末状态约束
        for name in thermal_storage.keys():
            if scenario_idx == 0:
                model.addConstr(storage_heat_vars[0, name, scenario_idx] == thermal_storage[name]["initial_heat"],
                                name=f"StorageEnergy_initial_scenario{scenario_idx}_HST{name}")
                model.addConstr(
                    storage_heat_vars[len(hourly_demand), name, scenario_idx] == thermal_storage[name]["initial_heat"],
                    name=f"StorageEnergy_final_scenario{scenario_idx}_HST{name}")

            else:
                model.addConstr(
                    storage_heat_vars[0, name, scenario_idx] == storage_heat_vars[
                        len(hourly_demand), name, scenario_idx - 1],
                    name=f"StorageEnergy_initial_scenario{scenario_idx}_HST{name}")
            model.addConstr(
                storage_heat_vars[len(hourly_demand), name, scenario_idx] == storage_heat_vars[0, name, scenario_idx],
                name=f"StorageEnergy_final_scenario{scenario_idx}_HST{name}")

    # 求解模型
    model.optimize()

    # 检查模型是否找到了最优解
    if model.Status == gp.GRB.Status.OPTIMAL:
        # print("找到最优解")

        # 提取解决方案
        solution_chp = {(hour, name, idx): power_vars_chp[hour, name, idx].x for hour in range(len(hourly_demand))
                        for name in generators_chp.keys() for idx in range(len(wind_scenarios))}
        solution_condensing = {(hour, name, idx): power_vars_condensing[hour, name, idx].x for hour in
                               range(len(hourly_demand))
                               for name in generators_condensing.keys() for idx in range(len(wind_scenarios))}
        solution_heat_chp = {(hour, name, idx): heat_vars_chp[hour, name, idx].x for hour in range(len(hourly_demand))
                             for name in generators_chp.keys() for idx in range(len(wind_scenarios))}
        solution_storage_charge = {(hour, name, idx): heat_charge_vars[hour, name, idx].x for hour in
                                   range(len(hourly_demand))
                                   for name in thermal_storage.keys() for idx in range(len(wind_scenarios))}
        solution_storage_discharge = {(hour, name, idx): heat_discharge_vars[hour, name, idx].x for hour in
                                      range(len(hourly_demand))
                                      for name in thermal_storage.keys() for idx in range(len(wind_scenarios))}
        solution_storage_energy = {(hour, name, idx): storage_heat_vars[hour, name, idx].x for hour in
                                   range(len(hourly_demand) + 1)
                                   for name in thermal_storage.keys() for idx in range(len(wind_scenarios))}
        solution_wind_power = {(hour, idx): wind_power_vars[hour, idx].x for hour in range(len(hourly_demand))
                               for idx in range(len(wind_scenarios))}

        # 将解决方案以字典形式返回
        results = {
            "CHP": solution_chp,
            "Condensing": solution_condensing,
            "Heat_CHP": solution_heat_chp,
            "Storage_Charge": solution_storage_charge,
            "Storage_Discharge": solution_storage_discharge,
            "Storage_Energy": solution_storage_energy,
            "Wind_Power": solution_wind_power,
        }

    else:
        print("未找到最优解")
        results = None
    # 返回最优值和解决方案
    return model.objVal, results


def economic_dispatch_continuous_reward(hourly_demand, hourly_heat_demand, wind_scenarios, probabilities,
                                        initial_storage_heat):
    """
    通过强化学习给出日间储热罐的区间约束
    """
    model = gp.Model("EconomicDispatchcontinuous")
    model.setParam('OutputFlag', 0)

    # 创建变量
    power_vars_chp = model.addVars(range(len(hourly_demand)), generators_chp.keys(), range(len(wind_scenarios)), lb=0,
                                   name="Power_chp")
    heat_vars_chp = model.addVars(range(len(hourly_demand)), generators_chp.keys(), range(len(wind_scenarios)), lb=0,
                                  name="Heat_chp")
    power_vars_condensing = model.addVars(range(len(hourly_demand)), generators_condensing.keys(),
                                          range(len(wind_scenarios)), lb=0,
                                          name="Power_condensing")
    heat_charge_vars = model.addVars(range(len(hourly_demand)), thermal_storage.keys(), range(len(wind_scenarios)),
                                     lb=0,
                                     name="HeatCharge")
    heat_discharge_vars = model.addVars(range(len(hourly_demand)), thermal_storage.keys(), range(len(wind_scenarios)),
                                        lb=0,
                                        name="HeatDischarge")
    storage_heat_vars = model.addVars(range(len(hourly_demand) + 1), thermal_storage.keys(), range(len(wind_scenarios)),
                                      lb=0,
                                      name="StorageHeat")
    charge_indicator_vars = model.addVars(range(len(hourly_demand)), thermal_storage.keys(), range(len(wind_scenarios)),
                                          vtype=gp.GRB.BINARY,
                                          name="ChargeIndicator")
    discharge_indicator_vars = model.addVars(range(len(hourly_demand)), thermal_storage.keys(),
                                             range(len(wind_scenarios)), vtype=gp.GRB.BINARY,
                                             name="DischargeIndicator")
    wind_power_vars = model.addVars(range(len(hourly_demand)), range(len(wind_scenarios)), lb=0, name="WindPower")

    # 设置目标函数
    expected_total_cost = 0
    # 遍历所有场景
    for scenario_idx, (wind_scenario, scenario_probability) in enumerate(zip(wind_scenarios, probabilities)):
        scenario_cost = 0
        for hour in range(len(hourly_demand)):
            for name in generators_chp.keys():
                scenario_cost += (heat_vars_chp[hour, name, scenario_idx] + power_vars_chp[
                    hour, name, scenario_idx]) * (heat_vars_chp[hour, name, scenario_idx] + power_vars_chp[
                    hour, name, scenario_idx]) * generators_chp[name]["a"] + \
                                 (heat_vars_chp[hour, name, scenario_idx] + power_vars_chp[hour, name, scenario_idx]) * \
                                 generators_chp[name]["b"] + \
                                 generators_chp[name]["c"]
            for name in generators_condensing.keys():
                scenario_cost += power_vars_condensing[hour, name, scenario_idx] * power_vars_condensing[
                    hour, name, scenario_idx] * generators_condensing[name]["a"] + \
                                 power_vars_condensing[hour, name, scenario_idx] * generators_condensing[name]["b"] + \
                                 generators_condensing[name]["c"]
            # scenario_cost += wind_power_vars[hour, scenario_idx] * 0.33  # 风电的成本 (0.33的值会产生弃风现象)

            # scenario_cost += storage_charge_vars[hour, scenario_idx] * thermal_storage["charge_cost"]
            # scenario_cost += storage_discharge_vars[hour, scenario_idx] * thermal_storage["discharge_cost"]

            # 将每个场景的目标函数乘以相应的概率，并将结果累加到期望总成本中
        # expected_total_cost += scenario_probability * scenario_cost
        expected_total_cost += scenario_cost

    model.setObjective(expected_total_cost, gp.GRB.MINIMIZE)

    # 添加约束条件
    for scenario_idx, wind_scenario in enumerate(wind_scenarios):

        for hour in range(len(hourly_demand)):
            # 电力平衡
            model.addConstr(
                gp.quicksum(power_vars_chp[hour, name, scenario_idx] for name in generators_chp.keys())
                + gp.quicksum(power_vars_condensing[hour, name, scenario_idx] for name in generators_condensing.keys())
                + wind_power_vars[hour, scenario_idx]
                == hourly_demand[hour],
                name=f"PowerBalance_hour{hour}_scenario{scenario_idx}"
            )

            # 热力平衡

            model.addConstr(
                gp.quicksum(heat_vars_chp[hour, name, scenario_idx] for name in ['G1', 'G2', 'G3'])
                + heat_discharge_vars[hour, 'a', scenario_idx]
                == hourly_heat_demand[hour] + heat_charge_vars[hour, 'a', scenario_idx],
                name=f"HeatBalance_hour{hour}_scenario{scenario_idx}"
            )

            model.addConstr(
                gp.quicksum(heat_vars_chp[hour, name, scenario_idx] for name in ['G4', 'G5', 'G6'])
                + heat_discharge_vars[hour, 'b', scenario_idx]
                == hourly_heat_demand[hour] + heat_charge_vars[hour, 'b', scenario_idx],
                name=f"HeatBalance_hour{hour}_scenario{scenario_idx}"
            )

            # 热电联产发电机的热电输出关系
            for name in generators_chp.keys():
                # # 选择两个中的较小值
                # aux_var = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY,
                #                        name=f"aux_var_{hour}_{name}_{scenario_idx}")
                # model.addConstr(
                #     aux_var <= generators_chp[name]["Pmin"] - generators_chp[name]["cv1"] * heat_vars_chp[
                #         hour, name, scenario_idx])
                # model.addConstr(aux_var <= generators_chp[name]["cm"] * heat_vars_chp[hour, name, scenario_idx])

                model.addConstr(power_vars_chp[hour, name, scenario_idx] >= generators_chp[name]["cm"] * heat_vars_chp[
                    hour, name, scenario_idx])

                # if generators_chp[name]["Pmin"] - generators_chp[name]["cv1"] * heat_vars_chp[hour, name, scenario_idx] > heat_vars_chp[hour, name, scenario_idx] * generators_chp[name]["cm"]:
                #     model.addConstr(power_vars_chp[hour, name, scenario_idx] >= heat_vars_chp[hour, name, scenario_idx] * generators_chp[name]["cm"])
                # else:
                #     model.addConstr(power_vars_chp[hour, name, scenario_idx] >= generators_chp[name]["Pmin"] - generators_chp[name]["cv1"] * heat_vars_chp[hour, name, scenario_idx])
                model.addConstr(
                    power_vars_chp[hour, name, scenario_idx] <= generators_chp[name]["Pmax"] - generators_chp[name][
                        "cv1"] * heat_vars_chp[hour, name, scenario_idx])

            # 发电机组与热电机组的出力约束
            for name in generators_chp.keys():
                model.addConstr(
                    power_vars_chp[hour, name, scenario_idx]
                    <= generators_chp[name]["Pmax"],
                    name=f"CHP_max_power_hour{hour}_generator{name}_scenario{scenario_idx}"
                )
                model.addConstr(
                    power_vars_chp[hour, name, scenario_idx]
                    >= generators_chp[name]["Pmin"],
                    name=f"CHP_min_power_hour{hour}_generator{name}_scenario{scenario_idx}"
                )
                model.addConstr(
                    heat_vars_chp[hour, name, scenario_idx]
                    <= generators_chp[name]["Hmax"],
                    name=f"CHP_max_heat_hour{hour}_generator{name}_scenario{scenario_idx}"
                )

                if hour == 0:
                    pass
                    # model.addConstr(
                    #     power_vars_chp[hour, name, scenario_idx]
                    #     - generators_chp[name]["Pmin"]
                    #     <= generators_chp[name]["ramp_up"],
                    #     name=f"CHP_ramp_up_hour{hour}_generator{name}_scenario{scenario_idx}"
                    # )
                    # model.addConstr(
                    #     generators_chp[name]["Pmin"]
                    #     - power_vars_chp[hour, name, scenario_idx]
                    #     <= generators_chp[name]["ramp_down"],
                    #     name=f"CHP_ramp_down_hour{hour}_generator{name}_scenario{scenario_idx}"
                    # )
                else:
                    model.addConstr(
                        power_vars_chp[hour, name, scenario_idx]
                        - power_vars_chp[hour - 1, name, scenario_idx]
                        <= generators_chp[name]["ramp_up"],
                        name=f"CHP_ramp_up_hour{hour}_generator{name}_scenario{scenario_idx}"
                    )
                    model.addConstr(
                        power_vars_chp[hour - 1, name, scenario_idx]
                        - power_vars_chp[hour, name, scenario_idx]
                        <= generators_chp[name]["ramp_down"],
                        name=f"CHP_ramp_down_hour{hour}_generator{name}_scenario{scenario_idx}"
                    )

            for name in generators_condensing.keys():
                model.addConstr(
                    power_vars_condensing[hour, name, scenario_idx]
                    <= generators_condensing[name]["Pmax"],
                    name=f"Condensing_max_power_hour{hour}_generator{name}_scenario{scenario_idx}"
                )
                model.addConstr(
                    power_vars_condensing[hour, name, scenario_idx]
                    >= generators_condensing[name]["Pmin"],
                    name=f"Condensing_min_power_hour{hour}_generator{name}_scenario{scenario_idx}"
                )
                if hour == 0:
                    pass
                    # model.addConstr(
                    #     power_vars_condensing[hour, name, scenario_idx]
                    #     - generators_condensing[name]["Pmin"]
                    #     <= generators_condensing[name]["ramp_up"],
                    #     name=f"Condensing_ramp_up_hour{hour}_generator{name}_scenario{scenario_idx}"
                    # )
                    # model.addConstr(
                    #     generators_condensing[name]["Pmin"]
                    #     - power_vars_condensing[hour, name, scenario_idx]
                    #     <= generators_condensing[name]["ramp_down"],
                    #     name=f"Condensing_ramp_down_hour{hour}_generator{name}_scenario{scenario_idx}"
                    # )
                else:
                    model.addConstr(
                        power_vars_condensing[hour, name, scenario_idx]
                        - power_vars_condensing[hour - 1, name, scenario_idx]
                        <= generators_condensing[name]["ramp_up"],
                        name=f"Condensing_ramp_up_hour{hour}_generator{name}_scenario{scenario_idx}"
                    )
                    model.addConstr(
                        power_vars_condensing[hour - 1, name, scenario_idx]
                        - power_vars_condensing[hour, name, scenario_idx]
                        <= generators_condensing[name]["ramp_down"],
                        name=f"Condensing_ramp_down_hour{hour}_generator{name}_scenario{scenario_idx}"
                    )

            # 储热罐的约束
            for name in thermal_storage.keys():
                model.addConstr(
                    storage_heat_vars[hour + 1, name, scenario_idx] - storage_heat_vars[hour, name, scenario_idx]
                    == heat_charge_vars[hour, name, scenario_idx] - heat_discharge_vars[hour, name, scenario_idx],
                    name=f"StorageEnergy_hour{hour}_scenario{scenario_idx}_HST{name}")
                # 储热罐的容量约束和充放热约束
                model.addConstr(storage_heat_vars[hour, name, scenario_idx] <= thermal_storage[name]["capacity"],
                                name=f"StorageEnergy_max_hour{hour}_scenario{scenario_idx}_HST{name}")
                model.addConstr(storage_heat_vars[hour, name, scenario_idx] >= 0,
                                name=f"StorageEnergy_min_hour{hour}_scenario{scenario_idx}_HST{name}")
                model.addConstr(heat_charge_vars[hour, name, scenario_idx] <= thermal_storage[name]["charge_capacity"],
                                name=f"StorageCharge_max_hour{hour}_scenario{scenario_idx}_HST{name}")
                model.addConstr(
                    heat_discharge_vars[hour, name, scenario_idx] <= thermal_storage[name]["discharge_capacity"],
                    name=f"StorageDischarge_max_hour{hour}_scenario{scenario_idx}_HST{name}")
                # 不允许同时充放电
                model.addConstr(heat_charge_vars[hour, name, scenario_idx] <= thermal_storage[name]["charge_capacity"] *
                                charge_indicator_vars[hour, name, scenario_idx],
                                name=f"StorageCharge_max_hour{hour}_scenario{scenario_idx}_HST{name}")
                model.addConstr(
                    heat_discharge_vars[hour, name, scenario_idx] <= thermal_storage[name]["discharge_capacity"] *
                    discharge_indicator_vars[hour, name, scenario_idx],
                    name=f"StorageDischarge_max_hour{hour}_scenario{scenario_idx}_HST{name}")
                model.addConstr(charge_indicator_vars[hour, name, scenario_idx] + discharge_indicator_vars[
                    hour, name, scenario_idx] <= 1,
                                name=f"StorageChargeDischarge_{hour}_scenario{scenario_idx}_HST{name}")

            # 风电的约束
            model.addConstr(wind_power_vars[hour, scenario_idx] <= wind_scenario[hour],
                            name=f"WindPowerConstraint_hour{hour}_scenario{scenario_idx}")

        # 储热罐的始末状态约束
        for name in thermal_storage.keys():
            if scenario_idx == 0:
                model.addConstr(storage_heat_vars[0, name, scenario_idx] == thermal_storage[name]["initial_heat"],
                                name=f"StorageEnergy_initial_scenario{scenario_idx}_HST{name}")
                # model.addConstr(storage_heat_vars[len(hourly_demand), name, scenario_idx] == initial_storage_heat[0],
                #                 name=f"StorageEnergy_final_scenario{scenario_idx}_HST{name}")
                model.addRange(storage_heat_vars[len(hourly_demand), name, scenario_idx], initial_storage_heat[0],
                               initial_storage_heat[1],
                               name=f"StorageEnergy_final_scenario{scenario_idx}_HST{name}")

            # elif scenario_idx == 1:
            #     model.addConstr(
            #         storage_heat_vars[0, name, scenario_idx] == storage_heat_vars[
            #             len(hourly_demand), name, scenario_idx - 1],
            #         name=f"StorageEnergy_initial_scenario{scenario_idx}_HST{name}")
            #     model.addRange(storage_heat_vars[len(hourly_demand), name, scenario_idx], initial_storage_heat[1][0],
            #                    initial_storage_heat[1][1],
            #                    name=f"StorageEnergy_final_scenario{scenario_idx}_HST{name}")
            else:
                model.addConstr(
                    storage_heat_vars[0, name, scenario_idx] == storage_heat_vars[
                        len(hourly_demand), name, scenario_idx - 1],
                    name=f"StorageEnergy_initial_scenario{scenario_idx}_HST{name}")
                model.addConstr(
                    storage_heat_vars[len(hourly_demand), name, scenario_idx] == thermal_storage[name]["initial_heat"],
                    name=f"StorageEnergy_final_scenario{scenario_idx}_HST{name}")

    # 求解模型
    model.optimize()

    # 检查模型是否找到了最优解
    if model.Status == gp.GRB.Status.OPTIMAL:
        # print("找到最优解")

        # 提取解决方案
        solution_chp = {(hour, name, idx): power_vars_chp[hour, name, idx].x for hour in range(len(hourly_demand))
                        for name in generators_chp.keys() for idx in range(len(wind_scenarios))}
        solution_condensing = {(hour, name, idx): power_vars_condensing[hour, name, idx].x for hour in
                               range(len(hourly_demand))
                               for name in generators_condensing.keys() for idx in range(len(wind_scenarios))}
        solution_heat_chp = {(hour, name, idx): heat_vars_chp[hour, name, idx].x for hour in range(len(hourly_demand))
                             for name in generators_chp.keys() for idx in range(len(wind_scenarios))}
        solution_storage_charge = {(hour, name, idx): heat_charge_vars[hour, name, idx].x for hour in
                                   range(len(hourly_demand))
                                   for name in thermal_storage.keys() for idx in range(len(wind_scenarios))}
        solution_storage_discharge = {(hour, name, idx): heat_discharge_vars[hour, name, idx].x for hour in
                                      range(len(hourly_demand))
                                      for name in thermal_storage.keys() for idx in range(len(wind_scenarios))}
        solution_storage_energy = {(hour, name, idx): storage_heat_vars[hour, name, idx].x for hour in
                                   range(len(hourly_demand) + 1)
                                   for name in thermal_storage.keys() for idx in range(len(wind_scenarios))}
        solution_wind_power = {(hour, idx): wind_power_vars[hour, idx].x for hour in range(len(hourly_demand))
                               for idx in range(len(wind_scenarios))}

        # 将解决方案以字典形式返回
        results = {
            "CHP": solution_chp,
            "Condensing": solution_condensing,
            "Heat_CHP": solution_heat_chp,
            "Storage_Charge": solution_storage_charge,
            "Storage_Discharge": solution_storage_discharge,
            "Storage_Energy": solution_storage_energy,
            "Wind_Power": solution_wind_power,
        }

    else:
        print("未找到最优解")
        results = None
    # 返回最优值和解决方案
    return model.objVal, results
