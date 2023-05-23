import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
import seaborn as sns
import matplotlib.pyplot as plt

# TODO 异常值处理

# Load the Excel files
df = pd.read_csv('./data/WindForecast_20210101-20210331.csv')

# use the first column as the row index
df = df.set_index(df.columns[0])

# read the Day-ahead forecast [MW] and the Measured & upscaled [MW] column
df = df[['Day-ahead forecast [MW]', 'Measured & upscaled [MW]']]

# 计算风电预测误差
# df['Wind_Prediction_Error'] = df['Day-ahead forecast [MW]'] - df['Measured & upscaled [MW]']

L_order = 4  # number of lines above the main diagonal to consider
N_states = 8  # number of states


def correlation_matrix(df):
    df.index = pd.to_datetime(df.index, dayfirst=True)
    df['date'] = df.index.date
    df['time'] = df.index.time

    pivot_df = df.pivot(index='date', columns='time', values='Measured & upscaled [MW]')
    # drop any column that contains at least one NaN value
    pivot_df = pivot_df.dropna(axis=0)

    correlation_matrix = pivot_df.corr()

    # plt.figure(figsize=(10, 10))
    # sns.heatmap(correlation_matrix, square=True, cmap='RdBu_r')
    # plt.title('Correlation matrix of wind power output on different days')
    # plt.show()

    return correlation_matrix, pivot_df


def get_rho_matrix(correlation_matrix):
    diag_indices = np.diag_indices_from(correlation_matrix)
    num_lines = L_order  # number of lines above the main diagonal to consider

    mean_values = []
    for i in range(1, num_lines + 1):
        indices = diag_indices[0][:-i], diag_indices[1][i:]
        mean_values.append(correlation_matrix.values[indices].mean())

    # normalize mean_values
    mean_values = mean_values / np.sum(mean_values)

    return mean_values


def transition_matrix(df):
    df = df.copy()
    n_intervals = N_states  # number of intervals
    df['state'] = pd.qcut(df['Measured & upscaled [MW]'], n_intervals)

    # bin_width = 100  # change this to the desired bin width
    # bins = np.arange(df['Measured & upscaled [MW]'].min(), df['Measured & upscaled [MW]'].max(), bin_width)
    # df['state'] = pd.cut(df['Measured & upscaled [MW]'], bins, labels=False)

    # print the intervals
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


def mean_and_variance(pivot_df):
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


def lambda_matrix(rho_matrix, transition_probabilities):
    lambda_matrices = []
    for i in range(L_order):  # Loop over the 1st, 2nd, and 3rd orders
        lambda_matrix = pd.DataFrame(rho_matrix[i] * transition_probabilities[i])
        lambda_matrices.append(lambda_matrix)
    return tuple(lambda_matrices)


# Gaussian Mixture Model
# TODO: fit the historical data to a Gaussian Mixture Model
def gaussian_mixture_model(t, state, lambda_matrix, mean_df, var_df):
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
def find_interval(state, interval_list):
    for i, interval in enumerate(interval_list):
        if interval.left <= state <= interval.right:
            return i
    return None  # state is not within any interval


correlation_matrix, pivot_df = correlation_matrix(df)

rho_matrix = get_rho_matrix(correlation_matrix)

transition_probabilities, interval_list = transition_matrix(df)

lambda_matrix = lambda_matrix(rho_matrix, transition_probabilities)

mean_df, var_df = mean_and_variance(pivot_df)

wind = [175.75, 221.2, 302.76, 387.41]
state = []
for i in range(len(wind)):
    state.append(find_interval(wind[i], interval_list))
state_reverse = state[::-1]

gmm = gaussian_mixture_model(5, state_reverse, lambda_matrix, mean_df, var_df)

# Create a grid of points where the PDF will be evaluated
x = np.linspace(-20, 1000, 1000).reshape(-1, 1)

# Compute the PDF of the GMM at each point on the grid
pdf = np.exp(gmm.score_samples(x))

# Plot the PDF
plt.plot(x, pdf, label='GMM')

# get the point corresponding to the maximum probability density
x_max = x[np.argmax(pdf)]
print(x_max)

# Generate some samples and plot them
samples, labels = gmm.sample(n_samples=100)
print(samples, labels)
plt.hist(samples, bins=50, density=True, alpha=0.5, label='Samples')

plt.legend()
plt.show()
