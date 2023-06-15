import pandas as pd
import numpy as np
import math
import random
from sklearn.mixture import GaussianMixture
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# TODO 异常值处理

# L_order = 4  # number of lines above the main diagonal to consider
N_states = 30  # number of states


class WindGenerator(object):
    def __init__(self):
        self.state = N_states
        self.L_order = None
        self.data = None
        self.transition_matrix_cluster_0 = None
        self.transition_matrix_cluster_1 = None

    def load_data(self):
        # Load the Excel files
        df = pd.read_csv('./data/WindForecast_20220701-20230531.csv')
        # use the first column as the row index
        df = df.set_index(df.columns[0])
        # read the Day-ahead forecast [MW] and the Measured & upscaled [MW] column
        df = df[['Day-ahead forecast [MW]', 'Measured & upscaled [MW]']]

        # fill the missing values using linear interpolation
        df = df.interpolate()

        # Computing Wind Power Forecast Errors
        df['Wind_Prediction_Error'] = df['Day-ahead forecast [MW]'] - df['Measured & upscaled [MW]']

        df.index = pd.to_datetime(df.index, dayfirst=True)

        df_hourly = df.resample('H').mean()
        df_hourly = df_hourly.interpolate()

        return df_hourly

    def cluster(self):

        data = self.data['Day-ahead forecast [MW]'].values.reshape(-1, 1)

        self.scaler = StandardScaler()
        data = self.scaler.fit_transform(data)

        num_clusters = 2

        self.kmeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)

        self.kmeans.fit(data)

        self.data['Cluster'] = self.kmeans.labels_
        clusters = {}
        for i in set(self.kmeans.labels_):
            clusters[i] = self.data[self.data['Cluster'] == i]

        # 分别取出两个聚类的数据
        cluster_0 = clusters[0]
        cluster_1 = clusters[1]

        return cluster_0, cluster_1

    def predict_cluster(self, new_data_point):
        # Reshape and scale the new data point
        new_data_point = new_data_point.reshape(-1, 1)
        new_data_point = self.scaler.transform(new_data_point)

        # Predict the cluster of the new data point
        cluster = self.kmeans.predict(new_data_point)

        return cluster[0]

    def correlation_matrix(self, plot, correlation_threshold):
        self.data.index = pd.to_datetime(self.data.index, dayfirst=True)
        self.data['date'] = self.data.index.date
        self.data['time'] = self.data.index.time

        # pivot_df = df.pivot(index='date', columns='time', values='Measured & upscaled [MW]')
        # TODO 为什么这里要用mean，而不是直接用pivot，因为有些时间点有多个值，需要取平均
        pivot_df = self.data.pivot_table(index='date', columns='time', values='Wind_Prediction_Error', aggfunc='mean')
        # drop any column that contains at least one NaN value
        pivot_df = pivot_df.dropna(axis=0)

        correlation_matrix = pivot_df.corr()

        # 返回矩阵中某一列大于0.6的相关系数的行数
        L_order = (correlation_matrix.iloc[:, 0] > correlation_threshold).sum()

        if plot:
            plt.figure(figsize=(10, 10))
            sns.heatmap(correlation_matrix, square=True, cmap='RdBu_r')
            plt.title('Correlation matrix of wind power output on different time')
            plt.show()

        return correlation_matrix, pivot_df, L_order

    def get_rho_matrix(self, correlation_matrix):
        diag_indices = np.diag_indices_from(correlation_matrix)
        num_lines = self.L_order

        mean_values = []
        for i in range(1, num_lines + 1):
            indices = diag_indices[0][:-i], diag_indices[1][i:]
            mean_values.append(correlation_matrix.values[indices].mean())

        # normalize mean_values
        mean_values = mean_values / np.sum(mean_values)

        return mean_values

    def transition_matrix_cluster(self, cluster):
        df = cluster.copy()
        n_intervals = self.state

        # TODO 0610 可以直接对wind forecast error进行聚类  替代这里的简单的cut分状态

        # df['state'] = pd.qcut(df['Wind_Prediction_Error'], n_intervals)
        df['state'] = pd.cut(df['Wind_Prediction_Error'], n_intervals)

        # TODO interval_list有点问题  或者是聚类距离确认的问题
        intervals = df['state'].cat.categories
        interval_list = intervals.to_list()

        # Create a new column 'next_state' that is the 'state' column shifted up by one
        df['next_state'] = df['state'].shift(-1)

        df.index = pd.to_datetime(df.index, format="%d/%m/%Y %H:%M")
        df['time_diff'] = df.index.to_series().diff().dt.total_seconds()

        # Define a threshold for what constitutes a break (in seconds)
        break_threshold = 3650  # slightly more than 15 minutes

        # Create a new column 'is_break' that is True if 'time_diff' is greater than the break threshold
        df['is_break'] = df['time_diff'] > break_threshold

        # Drop the last row of the dataframe, which now has a NaN 'next_state' and 'time_diff'
        df = df[:-1]

        # Create a transition matrix as a dataframe, initialized with zeros
        transition_matrix = pd.DataFrame(0, index=interval_list, columns=interval_list)

        # Count the transitions from each state to each other state, ignoring transitions across breaks
        for i in range(len(df) - 1):
            if not df.iloc[i]['is_break']:
                current_state = df.iloc[i]['state']
                next_state = df.iloc[i + 1]['next_state']
                transition_matrix.loc[current_state, next_state] += 1

        # Normalize the transition matrix to get probabilities
        transition_matrix = transition_matrix.div(transition_matrix.sum(axis=1), axis=0)

        # fill the NaN value with 0
        transition_matrix = transition_matrix.fillna(0)

        return transition_matrix, interval_list

    def transition_matrix(self):
        df = self.data.copy()
        n_intervals = self.state  # number of intervals
        # df['state'] = pd.qcut(df['Wind_Prediction_Error'], n_intervals)
        df['state'] = pd.cut(df['Wind_Prediction_Error'], n_intervals)

        intervals = df['state'].cat.categories
        interval_list = intervals.to_list()

        transition_probabilities = []
        for i in range(1, L_order + 1):  # Calculate for 1st, 2nd, and 3rd order transitions
            # Calculate the state at the next time step
            df[f'next_state_{i}nd_order'] = df['state'].shift(-i)
            # TODO: Remove the last three rows, which do not have a next state
            df = df.iloc[:-1]  # Remove the last three rows, which do not have a next state

            # Calculate the transition counts and convert to numpy array
            transition_counts = pd.crosstab(df['state'], df[f'next_state_{i}nd_order']).values

            # Calculate the transition probabilities
            transition_probabilities.append(transition_counts / transition_counts.sum(axis=1)[:, None])

        return transition_probabilities, interval_list

    def mean_and_variance(self, pivot_df):
        pivot_df_copy = pivot_df.copy()

        # Convert interval_list into a list of boundaries
        boundaries = [interval.left for interval in interval_list]
        boundaries.append(interval_list[-1].right)

        mean_df = pd.DataFrame(index=interval_list)
        var_df = pd.DataFrame(index=interval_list)

        for col in pivot_df_copy.columns:
            # Cut the data into intervals
            pivot_df_copy['group'] = pd.cut(pivot_df_copy[col], bins=boundaries)

            # Calculate mean and variance for each group
            mean = pivot_df_copy.groupby('group')[col].mean()
            var = pivot_df_copy.groupby('group')[col].var()

            # Store results in dataframes
            mean_df[col] = mean
            var_df[col] = var

        # Drop the group column from pivot_df
        pivot_df_copy.drop('group', axis=1, inplace=True)

        return mean_df, var_df

    def lambda_matrix(self, rho_matrix, transition_probabilities):
        lambda_matrices = []
        for i in range(self.L_order):  # Loop over the 1st, 2nd, and 3rd orders
            lambda_matrix = pd.DataFrame(rho_matrix[i] * transition_probabilities[i])
            lambda_matrices.append(lambda_matrix)
        return tuple(lambda_matrices)

    # Gaussian Mixture Model
    # TODO: fit the historical data to a Gaussian Mixture Model
    def gaussian_mixture_model(self, t, state, lambda_matrix, mean_df, var_df):
        gmm = GaussianMixture(n_components=N_states, covariance_type='full')

        N_orders = len(lambda_matrix)
        weights_ = np.zeros(N_states)
        for i in range(N_orders):
            weights_ += lambda_matrix[i].iloc[state[i]]
        # weights_ = np.array([0.0085, 0.0271, 0.0203, 0.0175, 0.0337, 0.0590, 0.5336, 0.3003])  # special
        gmm.weights_ = np.array(weights_)

        means_ = mean_df.reset_index(drop=True)
        means_.index.name = None
        means_.columns = range(mean_df.shape[1])
        means_ = np.array(means_[t]).reshape(-1, 1)
        # means_ = np.array([31.5434, 161.5254, 247.1334, 355.5078, 450.4639, 554.2496, 661.1297, 736.2757]).reshape(-1,1)  # special
        gmm.means_ = means_

        covariances_ = var_df.reset_index(drop=True)
        covariances_.index.name = None
        covariances_.columns = range(var_df.shape[1])
        # covariances_ = covariances_.applymap(lambda x: np.sqrt(x))
        covariances_ = np.array(covariances_[t]).reshape(-1, 1, 1)
        # covariances_ = np.array([1045.8920, 805.4263, 772.6691, 801.8539, 761.7209, 949.5520, 688.3084, 600.9669]).reshape(-1, 1, 1)  # special
        # covariances_ = np.sqrt(covariances_).reshape(-1, 1, 1)
        gmm.covariances_ = covariances_

        gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(gmm.covariances_))

        return gmm

    # detemine the state brlongs to which interval
    def find_interval(self, state, interval_list):
        for i, interval in enumerate(interval_list):
            if interval.left <= state <= interval.right:
                return i
        return None  # state is not within any interval

    def rolling_prediction_cluster(self, current_state, transition_matrix):
        # Get the probabilities of transitioning from the current state to each other state
        transition_probabilities = transition_matrix.loc[current_state].iloc[0]

        # Sample the next state interval according to these probabilities
        next_state_interval = np.random.choice(transition_probabilities.index, p=transition_probabilities.values)

        # choose the most likely state
        # next_state_interval = transition_probabilities.idxmax()

        # Sample an exact value within the selected interval
        next_state = np.random.uniform(next_state_interval.left, next_state_interval.right)
        # next_state = (next_state_interval.left + next_state_interval.right) / 2

        return next_state

    # day-head scenario generation
    def rolling_prediction(self, wind, timestep, lambda_matrix, mean_df, var_df, plot=False, sample=True):
        # wind = [175.75, 221.2, 302.76, 387.41]
        state = []
        for i in range(len(wind)):
            state.append(self.find_interval(wind[i], interval_list))
            state_reverse = state[::-1]

        gmm = self.gaussian_mixture_model(timestep, state_reverse, lambda_matrix, mean_df, var_df)

        # sample or get the point corresponding to the maximum probability density
        if sample:
            next_predict_error = gmm.sample(1)[0]
        else:
            # Get the component with the highest weight
            max_weight_index = gmm.weights_.argmax()

            # Get the mean of this component
            next_predict_error = [gmm.means_[max_weight_index]]
            # next_predict_error = gmm.means_[np.argmax(gmm.weights_)]

        if plot:
            # Create a grid of points where the PDF will be evaluated

            # get the lower and upper boundary of the intervallist
            boundaries = [interval.left for interval in interval_list]
            boundaries.append(interval_list[-1].right)
            boundaries[0] = math.floor(boundaries[0])
            boundaries[-1] = math.ceil(boundaries[-1])
            x = np.linspace(boundaries[0], boundaries[-1], 1000).reshape(-1, 1)

            # Compute the PDF of the GMM at each point on the grid
            pdf = np.exp(gmm.score_samples(x))

            # Plot the PDF
            plt.plot(x, pdf, label='GMM')

            plt.legend()
            plt.show()

        # Generate some samples and plot them
        # samples, labels = gmm.sample(n_samples=100)
        # print(samples, labels)
        # plt.hist(samples, bins=50, density=True, alpha=0.5, label='Samples')

        return next_predict_error

    def get_day_ahead_scenario(self, seed, sample, cluster):

        # Get unique dates
        unique_dates = self.data.index.normalize().unique()

        # Randomly select one date from the unique dates, expect the last 8 day
        # random.seed(seed)
        # random_date = random.choice(unique_dates[:-8])
        random_date = pd.to_datetime('2022-7-5 00:00:00')

        # Select the corresponding hour of this date to be selected
        # Calculate the number of hours and minutes that L_order corresponds to
        # hours = int((self.L_order * 15) // 60)
        # minutes = int((self.L_order * 15) % 60)
        #
        # wind_error = self.data['Wind_Prediction_Error'].loc[
        #              random_date + pd.DateOffset(hours=24 - hours, minutes=-minutes):random_date + pd.DateOffset(
        #                  hours=23, minutes=45)]
        # wind_error = wind_error.tolist()
        #
        # day_ahead_wind_forecast = self.data['Day-ahead forecast [MW]'].loc[
        #                           random_date + pd.DateOffset(days=1):random_date + pd.DateOffset(days=2,
        #                                                                                           minutes=-15)]
        # wind_measured = self.data['Measured & upscaled [MW]'].loc[
        #                 random_date + pd.DateOffset(days=1):random_date + pd.DateOffset(days=2,
        #                                                                                 minutes=-15)]

        hours = self.L_order

        wind_error = self.data['Wind_Prediction_Error'].loc[
                     random_date + pd.DateOffset(hours=24 - hours):random_date + pd.DateOffset(hours=23)]
        wind_error = wind_error.tolist()

        # TODO 这边是25h 需要改一下
        day_ahead_wind_forecast = self.data['Day-ahead forecast [MW]'].loc[
                                  random_date + pd.DateOffset(days=1):random_date + pd.DateOffset(days=2)]
        wind_measured = self.data['Measured & upscaled [MW]'].loc[
                        random_date + pd.DateOffset(days=1):random_date + pd.DateOffset(days=2)]
        day_ahead_wind_error = day_ahead_wind_forecast - wind_measured

        day_ahead_predict_error = []

        for i in range(len(day_ahead_wind_forecast)):

            if cluster:
                if self.predict_cluster(day_ahead_wind_forecast.values[i]) == 0:
                    next_predict_error = self.rolling_prediction_cluster(wind_error, self.transition_matrix_cluster_0)
                    day_ahead_predict_error.append(next_predict_error)
                    # wind_error.append(next_predict_error)
                    wind_error.append(day_ahead_wind_error.values[i])
                    wind_error.pop(0)

                elif self.predict_cluster(day_ahead_wind_forecast.values[i]) == 1:
                    next_predict_error = self.rolling_prediction_cluster(wind_error, self.transition_matrix_cluster_1)
                    day_ahead_predict_error.append(next_predict_error)
                    # wind_error.append(next_predict_error)
                    wind_error.append(day_ahead_wind_error.values[i])
                    wind_error.pop(0)
                else:
                    assert 'unseen wind forecast value'

            else:

                wind_error = wind_error.tolist()

                next_predict_error = self.rolling_prediction(wind_error, i, lambda_matrix, mean_df, var_df, plot=True,
                                                             sample=sample)

                day_ahead_predict_error.append(next_predict_error[0][0])

                wind_error.append(next_predict_error[0][0])
                wind_error.pop(0)

        day_ahead_error = pd.Series(day_ahead_predict_error, index=day_ahead_wind_forecast.index)

        day_ahead_wind = day_ahead_wind_forecast - 1 * day_ahead_error

        day_ahead_wind_hourly_df = day_ahead_wind.resample('H').sum()
        wind_measured_hourly = wind_measured.resample('H').sum()
        wind_forecast_hourly = day_ahead_wind_forecast.resample('H').sum()

        return day_ahead_wind_hourly_df, wind_measured_hourly, wind_forecast_hourly

    def reset(self):
        self.data = self.load_data()
        return self.state


if __name__ == '__main__':

    wind_generator = WindGenerator()
    wind_generator.reset()

    cluster_0, cluster_1 = wind_generator.cluster()
    wind_generator.L_order = 1

    wind_generator.transition_matrix_cluster_0, interval_list_0 = wind_generator.transition_matrix_cluster(cluster_0)
    wind_generator.transition_matrix_cluster_1, interval_list_1 = wind_generator.transition_matrix_cluster(cluster_1)

    wind_scenarios_set = []
    for i in range(5):
        day_ahead_wind, wind_measured, wind_forecast_hourly = wind_generator.get_day_ahead_scenario(seed=1, sample=True,
                                                                                                    cluster=True)
        wind_scenarios_set.append(day_ahead_wind)

    wind_forecast_hourly.plot(label='day-ahead wind forecast')
    wind_measured.plot(label='measured wind')

    # Plot each line with a different color and transparency
    for i in range(len(wind_scenarios_set)):
        wind_scenarios_set[i].plot(color='gray', linestyle=':', linewidth=0.8)

    # day_ahead_wind.plot(label='day-ahead wind gengerated')
    plt.xlabel('Time', labelpad=-2)
    plt.legend()
    plt.show()

    correlation_threshold = 0.8
    correlation_matrix, pivot_df, L_order = wind_generator.correlation_matrix(plot=False,
                                                                              correlation_threshold=correlation_threshold)
    wind_generator.L_order = 1
    rho_matrix = wind_generator.get_rho_matrix(correlation_matrix)
    transition_probabilities, interval_list = wind_generator.transition_matrix()
    lambda_matrix = wind_generator.lambda_matrix(rho_matrix, transition_probabilities)
    mean_df, var_df = wind_generator.mean_and_variance(pivot_df)

    wind_scenarios_set = []
    for i in range(5):
        day_ahead_wind, wind_measured, wind_forecast_hourly = wind_generator.get_day_ahead_scenario(seed=1,
                                                                                                    sample=False)
        wind_scenarios_set.append(day_ahead_wind)

    # 场景生成效果检验
    # TODO 1. 换其他的预测误差模型，比如简单的ARMA模型  2. 现有的模型改进（增加历史数据值；数据聚类进行分类；对measured值直接继续HMP建模） 3.试下复现 hourly + 一阶 + 聚类

    # 计算误差并打印
    # error1 = np.sum(wind_measured - wind_forecast_hourly)
    # error2 = np.sum(wind_measured - day_ahead_wind)
    #
    # print('error1: ', error1)
    # print('error2: ', error2)

    # plot the day-ahead wind forecast and the measured wind
    wind_forecast_hourly.plot(label='day-ahead wind forecast')
    wind_measured.plot(label='measured wind')
    for i in range(len(wind_scenarios_set)):
        wind_scenarios_set[i].plot()
    # day_ahead_wind.plot(label='day-ahead wind gengerated')
    plt.xlabel('Time', labelpad=-2)
    plt.legend()
    plt.show()
