import pandas as pd
from mp_mdp.drawing import *
from gurobi import economic_dispatch_continuous_gurobi



def generate_wind_scenarios(hourly_wind_power_available, hourly_wind_power_available1, hourly_wind_power_available2, num_scenarios):
    scenarios = []
    probabilities = []

    for _ in range(num_scenarios):
        scenario = np.random.normal(hourly_wind_power_available, scale=np.array(hourly_wind_power_available) * 0.1)
        scenario = np.maximum(scenario, 0) # 保证不会出现负值
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
    num_scenarios = 10
    wind_scenarios, scenario_probabilities = generate_wind_scenarios(hourly_wind_power_available, num_scenarios)

    wind_scenarios = [hourly_wind_power_available, hourly_wind_power_available1, hourly_wind_power_available2]
    scenario_probabilities = [1, 1, 1]

    # for day in range(1):
    #     wind = [wind_scenarios[day]]
    #     results = economic_dispatch_continuous_gurobi(hourly_demand, hourly_heat_demand, wind, scenario_probabilities)

    obj, results = economic_dispatch_continuous_gurobi(hourly_demand, hourly_heat_demand, wind_scenarios, scenario_probabilities)

    scenario_results = results_process(wind_scenarios, results)

    # plot_power(scenario_results, day=3)
    # plot_wind_power(scenario_results, wind_scenarios, day=3)
    # plot_heat(scenario_results[0])
    # plot_storage(scenario_results[0])
    # plot_area_b_multi(scenario_results, day=3)

