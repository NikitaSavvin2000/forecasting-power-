import random
import pandas as pd
import numpy as np
import math
from parse import parse_hour_in_min, parse_min_in_hour

def processed_data (d, t_step_min):
    a = int(((24*60)/t_step_min) + 1)
    A = [0.0]*a
    t_turn_on = np.array(d['t_turn_on'])
    t_turn_of = np.array(d['t_turn_of'])
    P = np.array(d['P'])
    # проход по массиву длинной P 
    for i in range(len(P)):
        # минуты включения к примеру включен в 1:00 выключен в 2:00 ->
        # -> включен в 60 мин выключен в 120 мин от начала суток
        min_on = parse_hour_in_min(t_turn_on[i])
        min_of = parse_hour_in_min(t_turn_of[i])
        # определяет в какую ячейку списка вписывать начало включения
        number_in_list = math.trunc(min_on/t_step_min)
        # "Y" - число обозначающее сколько промежутков времени задевает включение
        # если интервал равен 30 мин а время включения 13:30 выключения 14:35
        # то задевает 3 промежутка времени по 30 минут
        Y = math.ceil((min_of -min_on)/t_step_min)
        # проверяет краевой случай, если время включения больше времени выключения
        # к примеру включение 16:20 2:00 выключение
        if min_on>=min_of:
            for j in range(0, len(A) - 1):
                A[j] += P[i]
            on = math.ceil((min_on)/t_step_min)
            of = math.ceil((min_of)/t_step_min)
            for k in range(of, on):
                A[k] -= P[i]
            # записывает данные при стандартном включении выключении, когда min_on < min_of
        for j in range(number_in_list, number_in_list+Y, 1):
            A[j] += P[i]
    r = {
        'T' : [x*t_step_min for x in range(len(A))],
        'P' : [p for p in A],
        'Time' : [parse_min_in_hour(x*t_step_min) for x in range(len(A))]
    }
    result = pd.DataFrame(data = r)
    result.to_excel(r'C:\\Users\\nsavvin\Desktop\\programm\\Result.xlsx')
    Data = pd.read_excel(r'C:\\Users\\nsavvin\\Desktop\\programm\\Result.xlsx' )
    return Data

def data_poisson(D, medium_values, count_values, N):
    for i in range(N):
        Q = []
        s = np.random.poisson(medium_values, count_values)
        for i in range(len(s)):
            P = np.array(D['P'])
            P_poission = [(P[w]*s[i])/1000 for w in range(len(P))]
            Q.append(P_poission)
    return Q

def data_poisson(D, medium_values, count_values, N):
        Q = []
        interval = []
        for k in range(len(D['Time'])):
            for i in range(N):
                poisson = np.random.poisson(medium_values, N)
                tow = random(0,1) #доработать это список чисел длинной N сумма сгенерированных чисел должна равнятся 1 и шаг значений до стотых к примеру 0.01 + 0.07 + 0.02 = 1
                interval_tow = k*100*tow[i]
                interval.append(interval_tow)
                
                
                


def min_and_max_poisson(Q, Data):
    Data_min = []
    Data_max = []
    R = list(map(list, zip(*Q))) # транспонирование списка списков
    for i in range(len(R)):
        P_min = min(R[i])
        P_max = max(R[i])
        Data_min.append(P_min)
        Data_max.append(P_max)
    r = {
        'Time' : [time for time in Data['Time']],
        'Min count consumer' : [(Data_min[i]/Data['P'][i])*1000 for i in range(len(Data['P']))],
        'P_min_poisson_kW' : [min for min in Data_min],
        'Max count consumer' : [(Data_max[i]/Data['P'][i])*1000 for i in range(len(Data['P']))],
        'P_max_poisson_kW' : [max for max in Data_max]
    }
    result = pd.DataFrame(data = r)
    result.to_excel(r'C:\\Users\\nsavvin\Desktop\\programm2\\Intermediate_data\\max_min_poisson.xlsx')
    
    return Data_min, Data_max
