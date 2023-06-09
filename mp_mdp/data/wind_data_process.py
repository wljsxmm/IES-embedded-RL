import pandas as pd
import os
import numpy as np
# get current path of this file
path = os.path.abspath(os.path.dirname(__file__))


"""scale the data from 2022-2023 to the test system"""
df = pd.read_csv(path + '/WindForecast_20220701-20230531.csv')
# use the first column as the row index
df = df.set_index(df.columns[0])

# Convert index to DateTimeIndex with day first
df.index = pd.to_datetime(df.index, dayfirst=True)

# read the Day-ahead forecast [MW] and the Measured & upscaled [MW] column
df = df[['Day-ahead forecast [MW]', 'Measured & upscaled [MW]']]
# rename the columns
df.columns = ['Forecast', 'Measured']

# Aggregate by day, take the average
df = df.resample('H').mean()

# Get boolean mask where True indicates the presence of at least one NaN in a day
# 这边分预测值和实际值的缺失值
mask = df['Measured'].resample('D').apply(lambda x: x.isnull().any().any())
# Filter out days where mask is True
df = df[~mask.reindex(df.index, method='ffill')]

# check the nan value
# print(df.isnull())

max_wind = 1000
min_wind = 200

df.Forecast = (df.Forecast - min(df.Forecast)) / np.ptp(df.Forecast)
df.Forecast = df.Forecast * (max_wind - min_wind) + min_wind
df.Measured = (df.Measured - min(df.Measured)) / np.ptp(df.Measured)
df.Measured = df.Measured * (max_wind - min_wind) + min_wind

# TODO 测试数据选取的trick
# 选择从2022年7月1日开始的10天的数据作为测试数据
df_test = df['2022-07-01':'2022-07-10']
# 选择df中除了测试数据外的数据作为训练数据
df_train = df.drop(df_test.index)

df_train.to_csv(path + '/WindForecast_20220711-20230531_train.csv', index=True)
df_test.to_csv(path + '/WindForecast_20220701-20220710_test.csv', index=True)
df.to_csv('/Users/xmm/Desktop/IES-embedded-RL/Rainbow_DQN/data' + '/WindForecast_20220701-20230531.csv', index=True)

"""process the excel file"""
# 从第四行开始读取读取xls文件
df2022 = pd.read_excel('WindForecast_20220701-20221231.xls', header=5)
df2023 = pd.read_excel('WindForecast_20230101-20230531.xls', header=5)

# 把2022年和2023年的数据合并，存储为csv文件
df = pd.concat([df2022, df2023], axis=0)
df.to_csv('WindForecast_20220701-20230531.csv', index=False)

print(df2022.head())