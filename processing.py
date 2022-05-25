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

def data_poisson(D, medium_values, count_values):
    Q = []
    s = np.random.poisson(medium_values, count_values)
    #path_result_poisson = 'C:\\Users\\nsavvin\\Desktop\\programm\\result_poisson\\result_poisson.xlsx'
    for i in range(len(s)):
        P = np.array(D['P'])
        P_poission = [(P[w]*s[i])/1000 for w in range(len(P))]
        Q.append(P_poission)
        #Data_poisson = pd.DataFrame(P_poission)
        #Data_poisson.to_excel(path_result_poisson)
        #print(P_poission)
    return Q

def min_and_max_poisson(Q):
    Data_min = []
    Data_max = []
    R = list(map(list, zip(*Q))) # транспонирование списка списков
    for i in range(len(R)):
        P_min = min(R[i])
        P_max = max(R[i])
        Data_min.append(P_min)
        Data_max.append(P_max)
    return Data_min, Data_max
        

            
            

        