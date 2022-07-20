import linecache
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import math
from sklearn.metrics import mean_squared_error

data = pd.read_csv(r'C:\\Users\\nsavvin\Desktop\\test forcasting\\de.csv')
data_de = data.drop(['end'],axis=1)

train_set = data_de.loc[data_de['start'] <= '2019-05-31 00:00']
#print(f"train_set - {train_set}")
train_set = train_set.set_index("start")
#print(f'train_set - {train_set}')
train_set.index = pd.to_datetime(train_set.index)
#print(f'train_set.index - {train_set.index}')
train_set = train_set.groupby(pd.Grouper(freq="h")).sum()
#print(f'train_set - {train_set}')

test_set  = data_de.loc[data_de['start'] > '2019-05-31 00:00']
#print(f'test_set - {test_set}')
test_set = test_set.set_index("start")
#print(f'test_set - {test_set}')
test_set.index = pd.to_datetime(test_set.index)
#print(f'test_set - {test_set.index}')
test_set = test_set.groupby(pd.Grouper(freq="h")).sum()
#print(f'test_set - {test_set}')

data_de = data[['start', 'load']]
#print(f'data_de - {data_de[:3]}')
data_de['start'] = pd.to_datetime(data_de['start'])
#print(f'data_de['"start"'] - {e}'.format(e = data_de[:3]))
data_de = data_de.set_index("start")
#print(f'data_de- {data_de[:3]}')
data_de = data_de.groupby(pd.Grouper(freq="h")).sum()
#print(f'data_de- {data_de[:3]}')

data_de['start'] = pd.to_datetime(data_de.index)
#print(f'data_de['"start"'] - {e}'.format(e = data_de[:3]))

plt.figure(figsize=(16,10))

#-----EXAMPLE-------
r = '-'*50
'''
print(r)
print('Данные по строке - plt.plot(data_de['"start"'], data_de['"load"'])')
print(r)
print(f'Тип данных в массиве data_de['"start"'] - {t}'.format(t=type(data_de['start'])))
print(f'Длинна массива data_de['"start"'] - {l}'.format(l=len(data_de['start'])))
print("Выборка 3 значения для data_de['start']")
print(data_de['start'][:3])
print(r)
print(f'Тип данных в массиве data_de['"load"'] - {t}'.format(t=type(data_de['load'])))
print(f'Длинна массива data_de['"load"'] - {l}'.format(l=len(data_de['load'])))
print("Выборка 3 значения для data_de['load']")
print(data_de['load'][:3])
print(r)
'''
#-----EXAMPLE-------
