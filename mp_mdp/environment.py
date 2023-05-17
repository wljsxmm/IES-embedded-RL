import numpy as np
import pandas as pd
from mp_mdp.economic_dispatch_gurobi import *
from mp_mdp.parameters import *
from mp_mdp.gurobi import *
from mp_mdp.drawing import *


def calculate_wind_discard(results, wind_scenarios, day):
    wind_power = []
    for i in range(day):
        wind_power.append(results[i]["Wind_Power"])
    wind_power = [wind_power[i][hour] for i in range(day) for hour in range(24)]

    # wind_power = results["Wind_Power"]
    wind_power_forecast = [wind_scenarios[day][hour] for day in range(day) for hour in range(24)]
    hours = range(24 * day)
    # 计算出总的弃风量
    wind_power_discard = sum([wind_power_forecast[hour] - wind_power[hour] for hour in hours])
    wind_power_accommodation = sum(wind_power)
    # print("===========Wind Power Discard===========: ", wind_power_discard)

    return wind_power_discard, wind_power_accommodation


class HST_Env(object):
    def __init__(self, mode):
        self.wind_scenarios = None
        self.mode = mode
        self.done = False

    def step(self, action, day):
        if self.mode == 1:
            self.done = True if day >= 7 else False
            s = [sum(self.wind_scenarios[day - 1]), sum(self.wind_scenarios[day]), sum(self.wind_scenarios[day + 1])]
            s_ = [sum(self.wind_scenarios[day]), sum(self.wind_scenarios[day + 1]), sum(self.wind_scenarios[day + 2])]
            r, results = economic_dispatch_continuous_reward(hourly_demand, hourly_heat_demand, self.wind_scenarios,
                                                             [1, 1, 1], action)
            scenario_results = results_process(s, results)
            wind_curtailed, wind_accommodation = calculate_wind_discard(scenario_results, self.wind_scenarios[day-1:day+2], len(s))
            reward = wind_accommodation
            return s_, reward, self.done

        elif self.mode == 0:
            self.done = True if day >= 1 else False
            s = [sum(self.wind_scenarios[day-1]), sum(self.wind_scenarios[day]), sum(self.wind_scenarios[day+1])]
            r, results = economic_dispatch_continuous_reward(hourly_demand, hourly_heat_demand, self.wind_scenarios,
                                                             [1, 1, 1], action)
            scenario_results = results_process(s, results)
            plot_wind_power(scenario_results, self.wind_scenarios, day=3)
            plot_area_b_multi(scenario_results, day=3)
            wind_curtailed, wind_accommodation = calculate_wind_discard(scenario_results, self.wind_scenarios, len(s))
            reward = wind_accommodation
            return 0, reward, self.done

    def reset(self, episode_steps):
        if self.mode == 1:
            self.wind_scenarios, _ = generate_wind_scenarios(hourly_wind_power_available, hourly_wind_power_available1, hourly_wind_power_available2, 10)
            # 在self.wind_scenarios中随机选择10个场景组成一个episode
            self.wind_scenarios = [self.wind_scenarios[i] for i in np.random.choice(10, 10, replace=False)]
            state = [sum(self.wind_scenarios[episode_steps]), sum(self.wind_scenarios[episode_steps + 1]), sum(self.wind_scenarios[episode_steps + 2])]

        elif self.mode == 0:
            self.wind_scenarios = [hourly_wind_power_available, hourly_wind_power_available1, hourly_wind_power_available2]
            state = [sum(self.wind_scenarios[episode_steps]), sum(self.wind_scenarios[episode_steps + 1]), sum(self.wind_scenarios[episode_steps + 2])]
        else:
            assert False
        return state


def env_make(mode):
    '''确定性的不用历史数据进行优化 仅需验证随机生成场景的可行性'''
    # file_path = '/Users/xmm/PycharmProjects/pythonProject/rl4uc-master/DRL-code-pytorch/4.PPO-discrete/data/trainY2022.csv'
    # 读取forecast列
    # wind_df = pd.read_csv(file_path, index_col=[0,1], parse_dates=True)
    # wind_df = pd.read_csv(file_path)
    env = HST_Env(mode)
    return env
