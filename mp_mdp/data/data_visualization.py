import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('WebAgg')



# import scienceplots
plt.style.use(['ieee', 'grid', 'no-latex'])
plt.rcParams['figure.figsize'] = (10, 6)
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.size'] = 8
# plt.rcParams['axes.labelsize'] = 8
# plt.rcParams['axes.titlesize'] = 12
# plt.rcParams['xtick.labelsize'] = 8
# plt.rcParams['ytick.labelsize'] = 8
# plt.rcParams['legend.fontsize'] = 8
# plt.rcParams['figure.titlesize'] = 12


def convert_to_hourly(df):
    df['DateTime'] = pd.to_datetime(df['DateTime'], format="%d/%m/%Y %H:%M")
    df = df.set_index('DateTime')
    numeric_df = df.select_dtypes(include=['number'])
    hourly_df = numeric_df.resample('H').sum()
    return hourly_df


wind_data = pd.read_csv("WindForecast_20210101-20210331.csv")

wind_aggregate_hour = convert_to_hourly(wind_data)
wind_aggregate_day = wind_aggregate_hour.resample('D').sum()

# 取第四列
wind_data_measured = wind_aggregate_day.iloc[:, 3:4]
wind_data_forecast = wind_aggregate_day.iloc[:, 1:2]
# wind_data_measured = wind_data_measured.rename(columns={'Measured & upscaled [MW]': 'Measured'})
# wind_data_forecast = wind_data_forecast.rename(columns={'Day-ahead forecast [MW]': 'Forecast'})
# 同时取第二列和第四列
wind_data_two = wind_aggregate_day.iloc[:, [1, 3]]


wind_data_measured_hour = wind_aggregate_hour.iloc[:, 3:4]
wind_data_forecast_hour = wind_aggregate_hour.iloc[:, 1:2]
wind_data_hour_two = wind_aggregate_hour.iloc[:, [1, 3]]
# 在wind_data_hour_two中随机选取一周
wind_data_hour_two = wind_data_hour_two.loc['2021-01-01':'2021-01-03']

wind_data_hour_two.plot()
plt.xlabel('Date')
plt.ylabel('Wind power output [MW]')
plt.title('Wind power output in Belgium')
plt.show()

# plot the data
wind_data_two.plot()
plt.xlabel('Date')
plt.ylabel('Wind power output [MW]', labelpad=-2)
plt.title('Wind power output in Belgium')
plt.show()



