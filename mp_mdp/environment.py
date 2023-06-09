import numpy as np
import pandas as pd
from mp_mdp.economic_dispatch_gurobi import *
from mp_mdp.parameters import *
from mp_mdp.gurobi import *
from mp_mdp.drawing import *


# from mp_mdp.wind_scenarios_generator import *


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


class HstEnv(object):
    def __init__(self, mode):
        self.state_actual = None
        self.state_obs = None
        self.data = None
        self.wind_scenarios = None
        self.mode = mode
        self.done = False

    def load_data(self):

        # Load the Excel files
        if self.mode == 1:
            df = pd.read_csv('data/WindForecast_20220711-20230531_train.csv')
        else:
            df = pd.read_csv('data/WindForecast_20220701-20220710_test.csv')
        df = df.set_index(df.columns[0])
        df.index = pd.to_datetime(df.index, format="%Y-%m-%d %H:%M:%S")
        self.data = df

        return df

    def step_uncertainty(self, action, day, max_day , draw=False):
        if self.mode == 1:
            self.done = True if day >= max_day else False

            s = [self.state_actual[24 * (day - 1):24 * day].values, self.state_actual[24 * day:24 * (day + 1)].values,
                 self.state_actual[24 * (day + 1):24 * (day + 2)].values]

            s_ = np.concatenate([self.state_obs[24 * day:24 * (day + 1)].values, self.state_obs[24 * (day + 1):24 * (day + 2)].values,
                  self.state_obs[24 * (day + 2):24 * (day + 3)].values], axis=0)
            r, results = economic_dispatch_continuous_reward(hourly_demand, hourly_heat_demand, s, [1, 1, 1], action)
            r_base, results_base = economic_dispatch_continuous_gurobi(hourly_demand, hourly_heat_demand, s, [1, 1, 1])
            scenario_results = results_process(s, results)
            scenario_results_base = results_process(s, results_base)
            wind_curtailed, wind_accommodation = calculate_wind_discard(scenario_results, s, 3)
            wind_curtailed_base, wind_accommodation_base = calculate_wind_discard(scenario_results_base, s, 3)
            reward = wind_accommodation - wind_accommodation_base

        elif self.mode == 0:
            self.done = True if day >= max_day else False
            s = [self.state_actual[24 * (day - 1):24 * day].values, self.state_actual[24 * day:24 * (day + 1)].values,
                 self.state_actual[24 * (day + 1):24 * (day + 2)].values]

            s_ = np.concatenate([self.state_obs[24 * day:24 * (day + 1)].values, self.state_obs[24 * (day + 1):24 * (day + 2)].values,
                  self.state_obs[24 * (day + 2):24 * (day + 3)].values], axis=0)
            r, results = economic_dispatch_continuous_reward(hourly_demand, hourly_heat_demand, s, [1, 1, 1], action)
            r_base, results_base = economic_dispatch_continuous_gurobi(hourly_demand, hourly_heat_demand, s, [1, 1, 1])
            scenario_results = results_process(s, results)
            scenario_results_base = results_process(s, results_base)
            if draw:
                plot_wind_power(scenario_results, s, day=3)
                plot_wind_power(scenario_results_base, s, day=3)
                # plot_area_b_multi(scenario_results, day=3)
                # plot_area_b_multi(scenario_results_base, day=3)
            wind_curtailed, wind_accommodation = calculate_wind_discard(scenario_results, s, 3)
            wind_curtailed_base, wind_accommodation_base = calculate_wind_discard(scenario_results_base, s, 3)
            reward = wind_accommodation - wind_accommodation_base


        else:
            assert False, "mode error"
        return s_, reward, self.done

    def step(self, action, day):
        if self.mode == 1:
            self.done = True if day >= 7 else False
            # s = [sum(self.wind_scenarios[day - 1]), sum(self.wind_scenarios[day]), sum(self.wind_scenarios[day + 1])]
            s = [self.wind_scenarios[day - 1], self.wind_scenarios[day], self.wind_scenarios[day + 1]]
            # s_ = [sum(self.wind_scenarios[day]), sum(self.wind_scenarios[day + 1]), sum(self.wind_scenarios[day + 2])]
            # s_ = [self.wind_scenarios[day], self.wind_scenarios[day+1], self.wind_scenarios[day+2]]
            # s_ = self.wind_scenarios[day]+self.wind_scenarios[day+1]+self.wind_scenarios[day+2]
            s_ = np.concatenate((self.wind_scenarios[day], self.wind_scenarios[day + 1], self.wind_scenarios[day + 2]),
                                axis=0)
            # TODO 确定性这边self.wind_scenarios整个场景传进去是有问题的
            r, results = economic_dispatch_continuous_reward(hourly_demand, hourly_heat_demand, self.wind_scenarios,
                                                             [1, 1, 1], action)
            r_base, results_base = economic_dispatch_continuous_gurobi(hourly_demand, hourly_heat_demand, s, [1, 1, 1])
            scenario_results = results_process(s, results)
            scenario_results_base = results_process(s, results_base)
            wind_curtailed, wind_accommodation = calculate_wind_discard(scenario_results,
                                                                        self.wind_scenarios[day - 1:day + 2], 3)
            wind_curtailed_base, wind_accommodation_base = calculate_wind_discard(scenario_results_base,
                                                                                  self.wind_scenarios[day - 1:day + 2],
                                                                                  3)
            reward = wind_accommodation - wind_accommodation_base
            return s_, reward, self.done

        elif self.mode == 0:
            self.done = True if day >= 1 else False
            # s = [sum(self.wind_scenarios[day-1]), sum(self.wind_scenarios[day]), sum(self.wind_scenarios[day+1])]
            s = [self.wind_scenarios[day - 1], self.wind_scenarios[day], self.wind_scenarios[day + 1]]
            r, results = economic_dispatch_continuous_reward(hourly_demand, hourly_heat_demand, self.wind_scenarios,
                                                             [1, 1, 1], action)
            scenario_results = results_process(s, results)
            plot_area_b_multi(scenario_results, day=3)
            plot_wind_power(scenario_results, s, day=3)
            wind_curtailed, wind_accommodation = calculate_wind_discard(scenario_results, self.wind_scenarios, 3)
            reward = wind_accommodation
            return 0, reward, self.done

    def reset_uncertainty(self, day, fixed):
        if self.mode == 1:

            # Randomly select 10 consecutive days of data in self.data as an episode
            # Randomly select a starting point
            # Create a new DataFrame excluding the last 10 days
            data_excluding_last_10_days = self.data.iloc[:-10 * 24]  # assuming your data is hourly

            # Randomly select a starting point from the new DataFrame
            if fixed:
                start = pd.to_datetime('2022-08-01 00:00:00')
            else:
                start = data_excluding_last_10_days.sample(1).index[0]

            # Normalize the start datetime to the start of the day
            start = start.normalize()

            # Get the end point 10 days later
            # TODO: 这边的10天是怎么确定的？？？ 一个episode直接设定为整个供暖季节，参考n-step那篇文章 这个其实更实际一点
            end = start + pd.DateOffset(days=20)
            # Subtract 1 hour from the end time
            end = end - pd.DateOffset(hours=1)

            # Slice the dataframe to get the 10 consecutive days
            self.wind_scenarios = self.data.loc[start:end]

            # choose the first 3 days as the initial state
            # state = self.wind_scenarios[24*day:24*(day+3)]

            state = np.concatenate(
                (self.wind_scenarios[24 * day:24 * (day + 1)], self.wind_scenarios[24 * (day + 1):24 * (day + 2)],
                 self.wind_scenarios[24 * (day + 2):24 * (day + 3)]), axis=0)

            self.state_obs = self.wind_scenarios.Forecast
            self.state_actual = self.wind_scenarios.Measured

        elif self.mode == 0:
            self.wind_scenarios = self.data
            state = np.concatenate(
                (self.wind_scenarios[24 * day:24 * (day + 1)], self.wind_scenarios[24 * (day + 1):24 * (day + 2)],
                 self.wind_scenarios[24 * (day + 2):24 * (day + 3)]), axis=0)
            self.state_obs = self.wind_scenarios.Forecast
            self.state_actual = self.wind_scenarios.Measured
        else:
            raise ValueError('Invalid mode')

        return state[:, 0]

    def reset(self, episode_steps):
        if self.mode == 1:
            self.wind_scenarios, _ = generate_wind_scenarios(hourly_wind_power_available,
                                                             hourly_wind_power_available_low,
                                                             hourly_wind_power_available_high, 10)
            # 在self.wind_scenarios中随机选择10个场景组成一个episode
            self.wind_scenarios = [self.wind_scenarios[i] for i in np.random.choice(10, 10, replace=False)]
            # state = [sum(self.wind_scenarios[episode_steps]), sum(self.wind_scenarios[episode_steps + 1]), sum(self.wind_scenarios[episode_steps + 2])]
            state = [self.wind_scenarios[episode_steps], self.wind_scenarios[episode_steps + 1],
                     self.wind_scenarios[episode_steps + 2]]
            # state = self.wind_scenarios[episode_steps]+self.wind_scenarios[episode_steps + 1]+self.wind_scenarios[episode_steps + 2]
            state = np.concatenate((self.wind_scenarios[episode_steps], self.wind_scenarios[episode_steps + 1],
                                    self.wind_scenarios[episode_steps + 2]), axis=0)

            # state = self.wind_scenarios[episode_steps]+self.wind_scenarios[episode_steps + 1]+self.wind_scenarios[episode_steps + 2]
        elif self.mode == 0:
            self.wind_scenarios = [hourly_wind_power_available_low, hourly_wind_power_available,
                                   hourly_wind_power_available_high]
            # state = [sum(self.wind_scenarios[episode_steps]), sum(self.wind_scenarios[episode_steps + 1]), sum(self.wind_scenarios[episode_steps + 2])]
            state = [self.wind_scenarios[episode_steps], self.wind_scenarios[episode_steps + 1],
                     self.wind_scenarios[episode_steps + 2]]
            state = self.wind_scenarios[episode_steps] + self.wind_scenarios[episode_steps + 1] + self.wind_scenarios[
                episode_steps + 2]
        else:
            assert False
        return state


def env_make(mode):
    '''确定性的不用历史数据进行优化 仅需验证随机生成场景的可行性'''

    env = HstEnv(mode)
    return env
