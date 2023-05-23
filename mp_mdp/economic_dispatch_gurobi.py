import pandas as pd
from mp_mdp.drawing import *
from gurobi import economic_dispatch_continuous_gurobi
from gurobi import economic_dispatch_continuous_reward
from gurobi import economic_dispatch_continuous_optimal


def generate_wind_scenarios(hourly_wind_power_available, hourly_wind_power_available1, hourly_wind_power_available2,
                            num_scenarios):
    scenarios = []
    probabilities = []

    for _ in range(num_scenarios):
        scenario = np.random.normal(hourly_wind_power_available, scale=np.array(hourly_wind_power_available) * 0.1)
        scenario = np.maximum(scenario, 0)  # 保证不会出现负值
        scenarios.append(scenario)
        scenario1 = np.random.normal(hourly_wind_power_available1, scale=np.array(hourly_wind_power_available1) * 0.1)
        scenario1 = np.maximum(scenario1, 0)  # 保证不会出现负值
        scenarios.append(scenario1)
        scenario2 = np.random.normal(hourly_wind_power_available2, scale=np.array(hourly_wind_power_available2) * 0.1)
        scenario2 = np.maximum(scenario2, 0)  # 保证不会出现负值
        scenarios.append(scenario2)
        probabilities.append(1 / (3 * num_scenarios))

    return scenarios, probabilities


def save_results_to_csv(results, filename):
    chp_power_data = results["CHP_Power"]
    condensing_power_data = results["Condensing_Power"]
    chp_heat_data = results["Heat_CHP"]
    storage_charge_data = results["Storage_Charge"]
    storage_discharge_data = results["Storage_Discharge"]
    storage_energy_data = results["Storage_Energy"]
    wind_power_data = results["Wind_Power"]

    data = []
    for hour in range(len(hourly_demand)):
        row = {"hour": hour}
        for name in generators_chp.keys():
            row[name] = chp_power_data[hour, name]
            row[name + "_heat"] = chp_heat_data[hour, name]
        for name in generators_condensing.keys():
            row[name] = condensing_power_data[hour, name]
        for name in thermal_storage.keys():
            row[name] = storage_energy_data[hour, name]
        for name in thermal_storage.keys():
            row[name + "_charge"] = storage_charge_data[hour, name]
        for name in thermal_storage.keys():
            row[name + "_discharge"] = storage_discharge_data[hour, name]
        row["Wind_Power"] = wind_power_data[hour]
        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)


def results_process(wind_scenarios, results):
    scenario_results = []
    for scenario_idx in range(len(wind_scenarios)):
        scenario_data = {
            "CHP_Power": {(hour, name): results["CHP"][(hour, name, scenario_idx)] for hour in range(len(hourly_demand))
                          for name in generators_chp.keys()},
            "Condensing_Power": {(hour, name): results["Condensing"][(hour, name, scenario_idx)] for hour in
                                 range(len(hourly_demand)) for name in generators_condensing.keys()},
            "Heat_CHP": {(hour, name): results["Heat_CHP"][(hour, name, scenario_idx)] for hour in
                         range(len(hourly_demand)) for name in generators_chp.keys()},
            "Storage_Charge": {(hour, name): results["Storage_Charge"][(hour, name, scenario_idx)] for hour in
                               range(len(hourly_demand)) for name in thermal_storage.keys()},
            "Storage_Discharge": {(hour, name): results["Storage_Discharge"][(hour, name, scenario_idx)] for hour in
                                  range(len(hourly_demand)) for name in thermal_storage.keys()},
            "Storage_Energy": {(hour, name): results["Storage_Energy"][(hour, name, scenario_idx)] for hour in
                               range(len(hourly_demand) + 1) for name in thermal_storage.keys()},
            "Wind_Power": {(hour): results["Wind_Power"][(hour, scenario_idx)] for hour in range(len(hourly_demand))},
        }
        filename = f"results/scenario_{scenario_idx + 1}.csv"
        # save_results_to_csv(scenario_data, filename)
        scenario_results.append(scenario_data)
    return scenario_results


if __name__ == "__main__":
    # 生成风电场景
    # num_scenarios = 10
    # wind_scenarios, scenario_probabilities = generate_wind_scenarios(hourly_wind_power_available, num_scenarios)

    wind_scenarios = [hourly_wind_power_available_low, hourly_wind_power_available, hourly_wind_power_available_high]
    wind_scenarios1 = [[i * 1.5 for i in hourly_wind_power_available], [i * 1 for i in hourly_wind_power_available_low], [i * 1.7 for i in hourly_wind_power_available_high]]
    wind_scenarios2 = [[i * 5 for i in hourly_wind_power_available], [i * 5 for i in hourly_wind_power_available], [i * 5 for i in hourly_wind_power_available]]

    scenario_probabilities = [1, 1, 1]

    # obj_day1, results_day1 = economic_dispatch_continuous_gurobi(hourly_demand, hourly_heat_demand,
    #                                                              [wind_scenarios[0]], [1])
    #
    # obj_day1_optimal, results_day1_optimal = economic_dispatch_continuous_optimal(hourly_demand, hourly_heat_demand,
    #                                                                               [wind_scenarios[0]], [1])
    #
    # scenario_results_day1 = results_process([wind_scenarios[0]], results_day1)
    # scenario_results_day1_optimal = results_process([wind_scenarios[0]], results_day1_optimal)
    #
    # print('obj_day1 {}; obj_day1_optimal {}'.format(obj_day1, obj_day1_optimal))
    #
    # plot_wind_power(scenario_results_day1, [wind_scenarios[0]], day=1)
    # plot_area_b_multi(scenario_results_day1, day=1)

    obj_1, results_1 = economic_dispatch_continuous_gurobi(hourly_demand, hourly_heat_demand, wind_scenarios,
                                                           scenario_probabilities)
    obj_optimal, results_optimal = economic_dispatch_continuous_optimal(hourly_demand, hourly_heat_demand,
                                                                        wind_scenarios, scenario_probabilities)
    scenario_results_1 = results_process(wind_scenarios, results_1)
    scenario_results_optimal = results_process(wind_scenarios, results_optimal)
    obj_day = []
    results_day = []
    for day in range(3):
        # 存在每天重置热量 偷热量的漏洞
        obj_temp, results_temp = economic_dispatch_continuous_gurobi(hourly_demand, hourly_heat_demand,
                                                                     [wind_scenarios[day]], [1])
        obj_day.append(obj_temp)
        results_day.append(results_temp)

    # print('obj_day is {} and the sum is {}'.format(obj_day, sum(obj_day)))
    # print('++++++++++++++++++++++')
    print('obj_1 is {}'.format(obj_1))
    print('++++++++++++++++++++++')
    print('obj_optimal is {}'.format(obj_optimal))

    # plot_wind_power(scenario_results_1, wind_scenarios, day=3)
    # plot_wind_power(scenario_results_optimal, wind_scenarios, day=3)
    # plot_area_b_multi(scenario_results_1, day=3)
    # plot_area_b_multi(scenario_results_optimal, day=3)

    obj, results = economic_dispatch_continuous_reward(hourly_demand, hourly_heat_demand, wind_scenarios,
                                                       scenario_probabilities, hst_status_combinations[12])
    obj_base, results_base = economic_dispatch_continuous_gurobi(hourly_demand, hourly_heat_demand, wind_scenarios,
                                                                    scenario_probabilities)
    obj1, results1 = economic_dispatch_continuous_reward(hourly_demand, hourly_heat_demand, wind_scenarios1,
                                                         scenario_probabilities, hst_status_combinations[24])
    obj1_base, results1_base = economic_dispatch_continuous_gurobi(hourly_demand, hourly_heat_demand, wind_scenarios1,
                                                                     scenario_probabilities)
    obj2, results2 = economic_dispatch_continuous_reward(hourly_demand, hourly_heat_demand, wind_scenarios2,
                                                         scenario_probabilities, hst_status_combinations[67])
    obj2_base, results2_base = economic_dispatch_continuous_gurobi(hourly_demand, hourly_heat_demand, wind_scenarios2,
                                                                     scenario_probabilities)

    print(obj, obj1, obj2)

    scenario_results = results_process(wind_scenarios, results)
    scenario_results_base = results_process(wind_scenarios, results_base)
    scenario_results1 = results_process(wind_scenarios1, results1)
    scenario_results1_base = results_process(wind_scenarios1, results1_base)
    scenario_results2 = results_process(wind_scenarios2, results2)
    scenario_results2_base = results_process(wind_scenarios2, results2_base)

    plot_wind_power(scenario_results, wind_scenarios, day=3)
    plot_wind_power(scenario_results_base, wind_scenarios, day=3)
    plot_wind_power(scenario_results1, wind_scenarios1, day=3)
    plot_wind_power(scenario_results1_base, wind_scenarios1, day=3)
    plot_wind_power(scenario_results2, wind_scenarios2, day=3)
    plot_wind_power(scenario_results2_base, wind_scenarios2, day=3)

    # plot_heat(scenario_results[0])
    # plot_storage(scenario_results[0])

    # plot_area_b_multi(scenario_results, day=3)
    # plot_area_b_multi(scenario_results1, day=3)
    # plot_area_b_multi(scenario_results2, day=3)
