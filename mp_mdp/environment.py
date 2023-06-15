import pandas as pd
from mp_mdp.economic_dispatch_gurobi import results_process, generate_wind_scenarios
from mp_mdp.parameters import *
from mp_mdp.gurobi import *
from mp_mdp.drawing import *


def calculate_wind_discard(results, wind_scenarios, day):
    wind_power = [results[i]["Wind_Power"] for i in range(day)]
    wind_power = [power[hour] for power in wind_power for hour in range(24)]

    wind_power_forecast = [wind_scenarios[day][hour] for day in range(day) for hour in range(24)]

    # Calculate total wind power discard
    wind_power_curtailed = sum(forecast - actual for forecast, actual in zip(wind_power_forecast, wind_power))
    wind_power_accommodation = sum(wind_power)

    return wind_power_curtailed, wind_power_accommodation


class HstEnv(object):
    def __init__(self, mode, day_cycle):
        self.state_actual = None
        self.state_obs = None
        self.data = None
        self.wind_scenarios = None
        self.mode = mode
        self.done = False
        self.day_cycle = day_cycle

    def load_data(self):

        # Load the Excel files
        if self.mode == 1:
            df = pd.read_csv('data/WindForecast_20220711-20230531_train.csv')
        elif self.mode == 0:
            df = pd.read_csv('data/WindForecast_20220701-20220710_test.csv')
        else:
            raise ValueError("Mode Error!")
        df = df.set_index(df.columns[0])
        df.index = pd.to_datetime(df.index, format="%Y-%m-%d %H:%M:%S")

        df = df.resample('H').asfreq()
        df = df.interpolate()

        self.data = df

        return df

    def step_uncertainty(self, action, day, max_day, draw=False):
        s = []
        for i in range(5):
            s.append(self.state_obs[24 * (day + i -1):24 * (day + i)].values.ravel())
        # s = [self.state_obs[24 * (day - 1):24 * day].values, self.state_obs[24 * day:24 * (day + 1)].values, self.state_obs[24 * (day + 1):24 * (day + 2)].values]

        s_ = self.state_obs.iloc[24 * day: 24 * (day + 5)].values.ravel()
        # s_2 = np.concatenate([self.state_obs[24 * day:24 * (day + 1)].values, self.state_obs[24 * (day + 1):24 * (day + 2)].values, self.state_obs[24 * (day + 2):24 * (day + 3)].values], axis=0)

        r, results = economic_dispatch_continuous_reward(hourly_demand, hourly_heat_demand, s, [1, 1, 1, 1,1], action)
        r_base, results_base = economic_dispatch_continuous_gurobi(hourly_demand, hourly_heat_demand, s, [1, 1, 1 , 1, 1])

        scenario_results = results_process(s, results)
        scenario_results_base = results_process(s, results_base)

        wind_curtailed, wind_accommodation = calculate_wind_discard(scenario_results, s, self.day_cycle)
        wind_curtailed_base, wind_accommodation_base = calculate_wind_discard(scenario_results_base, s, self.day_cycle)

        optimal = True
        if optimal:
            r_optimal, results_optimal = economic_dispatch_continuous_optimal(hourly_demand, hourly_heat_demand, s,
                                                                              [1, 1, 1,1,1])
            scenario_results_optimal = results_process(s, results_optimal)
            wind_curtailed_optimal, wind_accommodation_optimal = calculate_wind_discard(scenario_results_optimal, s,
                                                                                        self.day_cycle)
            reward_optimal = wind_accommodation_optimal - wind_accommodation_base
            reward_optimal_ratio = wind_accommodation_optimal / (
                    wind_accommodation_optimal + wind_curtailed_optimal) - wind_accommodation_base / (
                                           wind_accommodation_base + wind_curtailed_base)

        if draw:
            for scenario in [scenario_results, scenario_results_base, scenario_results_optimal]:
                plot_wind_power(scenario, s, day=self.day_cycle)
                plot_area_b_multi(scenario, day=self.day_cycle)

        reward = wind_accommodation - wind_accommodation_base
        reward_ratio = wind_accommodation / (wind_accommodation + wind_curtailed) - wind_accommodation_base / (
                wind_accommodation_base + wind_curtailed_base)

        print("Day: %d, Reward: %.2f, Reward Optimal: %.2f, Reward Ratio: %.2f, Reward Optimal Ratio: %.2f" % (day, reward, reward_optimal, reward_ratio, reward_optimal_ratio))

        self.done = day >= max_day or (self.mode == 1 and reward < 0)

        if -1e-4 <= reward_ratio <= 1e-4:
            reward_ratio = -1e4
        elif reward_ratio < -1e-4:
            reward_ratio *= 1e6
        elif reward_ratio > 1e-4:
            reward_ratio *= 1e6

        return s_, reward, self.done

    def step(self, action, day, max_day, draw=False):

        s = [self.wind_scenarios[day - 1], self.wind_scenarios[day], self.wind_scenarios[day + 1]]
        # s = [self.wind_scenarios[day - 1], self.wind_scenarios[day]]

        s_ = np.concatenate(self.wind_scenarios[day:day + self.day_cycle], axis=0)
        if len(s_) == 48:
            print("Day: %d, s_ length: %d" % (day, len(s_)))

        r, results = economic_dispatch_test(hourly_demand, hourly_heat_demand, s, action)  # TODO  probabilities [1, 1, 1] need to be changed
        r_base, results_base = economic_dispatch_test_base(hourly_demand, hourly_heat_demand, s)

        scenario_results = results_process(s, results)
        scenario_results_base = results_process(s, results_base)

        wind_curtailment, wind_accommodation = calculate_wind_discard(scenario_results, s, self.day_cycle)
        wind_curtailed_base, wind_accommodation_base = calculate_wind_discard(scenario_results_base, s, self.day_cycle)

        reward = wind_accommodation - wind_accommodation_base
        reward_ratio = wind_accommodation / (wind_accommodation + wind_curtailment) - wind_accommodation_base / (
                wind_accommodation_base + wind_curtailed_base)

        optimal = 0

        if optimal:
            r_optimal, results_optimal = economic_dispatch_test_optimal(hourly_demand, hourly_heat_demand, s)

            scenario_results_optimal = results_process(s, results_optimal)
            wind_curtailment_optimal, wind_accommodation_optimal = calculate_wind_discard(scenario_results_optimal, s,
                                                                                          self.day_cycle)
            reward_optimal = wind_accommodation_optimal - wind_accommodation_base
            reward_optimal_ratio = wind_accommodation_optimal / (
                        wind_accommodation_optimal + wind_curtailment_optimal) - wind_accommodation_base / (
                                               wind_accommodation_base + wind_curtailed_base)

        if draw and self.mode == 0:
            plot_area_test(scenario_results, day=self.day_cycle)
            plot_wind_power(scenario_results, s, day=self.day_cycle)

            plot_area_test(scenario_results_base, day=self.day_cycle)
            plot_area_test(scenario_results_optimal, day=self.day_cycle)
            plot_wind_power(scenario_results_base, s, day=self.day_cycle)
            plot_wind_power(scenario_results_optimal, s, day=self.day_cycle)

        self.done = day >= max_day or (self.mode == 1 and reward_ratio < 0)
        # print("Day: %d, Reward: %.2f, Reward Optimal: %.2f, Reward Ratio: %.2f, Reward Optimal Ratio: %.2f" % (day, reward, reward_optimal, reward_ratio, reward_optimal_ratio))
        if -1e-4 <= reward_ratio <= 1e-4:
            reward_ratio = -1e4
        elif reward_ratio < -1e-4:
            reward_ratio *= 1e6
        elif reward_ratio > 1e-4:
            reward_ratio *= 1e6
        return s_, reward, self.done

    def reset_uncertainty(self, day, fixed):
        if self.mode == 1:
            data_excluding_last_10_days = self.data.iloc[:-10 * 24]

            start = pd.to_datetime('2022-08-05 00:00:00') if fixed else data_excluding_last_10_days.sample().index[0]

            start = start.normalize()

            end = start + pd.DateOffset(days=12) - pd.DateOffset(hours=1)

            self.wind_scenarios = self.data.loc[start:end]
            # print('mode 1 {}'.format((len(self.wind_scenarios))))

        elif self.mode == 0:
            self.wind_scenarios = self.data
            # print('mode 0 {}'.format((len(self.wind_scenarios))))
        else:
            raise ValueError('Invalid mode')

        # Set the state observations and actual state
        self.state_obs = self.wind_scenarios.Forecast
        self.state_actual = self.wind_scenarios.Measured

        # Choose initial state
        state = self.state_obs.iloc[0: 120].values.ravel()
        # state = np.concatenate(
        #     (self.wind_scenarios[24 * day:24 * (day + 1)], self.wind_scenarios[24 * (day + 1):24 * (day + 2)], self.wind_scenarios[24 * (day + 2):24 * (day + 3)].values), axis=0)

        return state

    def reset(self, episode_steps):
        if self.mode == 1:
            self.wind_scenarios, _ = generate_wind_scenarios(hourly_wind_power_available_low,
                                                             hourly_wind_power_available,
                                                             hourly_wind_power_available_high, 10)
            # Select 10 random scenarios from self.wind_scenarios to form an episode
            self.wind_scenarios = [self.wind_scenarios[i] for i in np.random.choice(10, 7, replace=False)]

        elif self.mode == 0:
            self.wind_scenarios = [hourly_wind_power_available_low, hourly_wind_power_available, hourly_wind_power_available_high,
                                   hourly_wind_power_available, hourly_wind_power_available_low, hourly_wind_power_available_high,
                                   hourly_wind_power_available_high]
        else:
            raise ValueError('Invalid mode')

        # Choose the initial state
        state = np.concatenate(self.wind_scenarios[episode_steps:episode_steps + self.day_cycle], axis=0)

        return state


def env_make(mode, day_cycle):
    """
    确定性的不用历史数据进行优化 仅需验证随机生成场景的可行性
    """
    env = HstEnv(mode, day_cycle)
    env.load_data()
    return env
