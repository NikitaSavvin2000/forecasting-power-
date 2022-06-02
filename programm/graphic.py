import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


def graf_gistogram_medium(D, t_step_min, count_consumer):
    df1 = D
    x1=np.array(df1['T'], dtype=int)
    y=np.array(df1['P'], dtype=float)
    # находит среднее значение P в заданном промежутке в зависимости от количества файлов
    y1 = [y[i]/count_consumer for i in range(len(y))]
    Time_0 = np.array(df1['Time'], dtype=str)
    step = int(60/t_step_min)
    Time_01 = [Time_0[i] for i in range(0 ,len(Time_0), step)]
    # размер гафика
    fig = plt.figure(figsize=(20,15))
    ax = fig.add_subplot()
    # сам график
    ax.step(x1, y1, where='post',linewidth = 3)
    ax.set_xlabel('Время', size = 24)
    ax.set_ylabel('Средняя потребляема мощность 1 потребителя Вт', size = 15)
    ax.tick_params(labelsize = 14)
    #ax.set_ylim(-50, max(y1)+1000)
    ax.set_xlim(-20, 1500)
    ax.set_ylim(-50, 5000)
    #ax.legend(fontsize = 14)
    ax.set_xticks(np.arange(0, 1500, 60))
    ax.set_xticklabels(Time_01, fontsize = 8)
    ax.grid(which='major',
        color = 'k')
    ax.minorticks_on()
    ax.grid(which='minor',
        color = 'gray',
        linestyle = ':')
# шаг вспомогательной сетки в зависимости от выбранного шага
    ax.xaxis.set_minor_locator(MultipleLocator(t_step_min))
    #ax.legend()

    
def graf_gistogram_poison(D, t_step_min, y, x):
    x1=x
    # находит среднее значение P в заданном промежутке в зависимости от количества файлов
    # размер графика
    fig = plt.figure(figsize=(20,15))
    ax = fig.add_subplot()
    # сам график
    # размер графика
    fig = plt.figure(figsize=(20,15))
    ax = fig.add_subplot()
    Time_0 = np.array(D['Time'], dtype=str)
    step = int(60/t_step_min)
    Time_01 = [Time_0[i] for i in range(0 ,len(Time_0), step)]
    # сам график
    #ax.set_xticklabels(Time_01, fontsize = 8)
    df1 = D
    Time_0 = np.array(df1['Time'], dtype=str)
    step = int(60/t_step_min)
    Time_01 = [Time_0[i] for i in range(0 ,len(Time_0), step)]
    # размер графика
    ax.tick_params(labelsize = 14)
    #ax.set_xlim(-20, 15000)
    e = 2400 * int(60/t_step_min)
    step = e / 24
    end =  e + step
    ax.set_xlim(-200, end)
    ax.set_xticks(np.arange(0, end, step))
    ax.set_xticklabels(Time_01, fontsize = 8)
    for i in range(len(y)):
        y1 = y[i]
        ax.step(x1, y1, color = 'r', linewidth = 1)
    ax.grid(which='major',
        color = 'k')
    ax.minorticks_on()
    ax.grid(which='minor',
        color = 'gray',
        linestyle = ':')
# шаг вспомогательной сетки в зависимости от выбранного шага
    ax.xaxis.set_minor_locator(MultipleLocator(100))


def graf_gistogram_min_max_medium_poison(D, Data_min, Data_max, Data_medium, x, t_step_min):
    x1=x
    print(x1)
    # размер графика
    fig = plt.figure(figsize=(20,15))
    ax = fig.add_subplot()
    Time_0 = np.array(D['Time'], dtype=str)
    step = int(60/t_step_min)
    Time_01 = [Time_0[i] for i in range(0 ,len(Time_0), step)]
    # сам график
    y1 = Data_min
    y2 = Data_max
    y3 = Data_medium
    #ax.set_xticklabels(Time_01, fontsize = 8)
    df1 = D
    Time_0 = np.array(df1['Time'], dtype=str)
    step = int(60/t_step_min)
    Time_01 = [Time_0[i] for i in range(0 ,len(Time_0), step)]
    # размер графика
    ax.set_xlim(-200, 15000)
    ax.tick_params(labelsize = 14)
    #ax.set_xlim(-20, 15000)
    e = 2400 * int(60/t_step_min)
    step = e / 24
    end =  e + step
    ax.set_xlim(-200, end)
    ax.set_xticks(np.arange(0, end, step))
    # сам график
    ax.step(x1, y1, linestyle='--', color = 'b', linewidth = 1 )
    ax.step(x1, y2, linestyle='--', color = 'r', linewidth = 1)
    ax.step(x1, y3, color = 'g', linewidth = 2)
    ax.set_xticklabels(Time_01, fontsize = 8)
    ax.grid(which='major',
        color = 'k')
    ax.minorticks_on()
    ax.grid(which='minor',
        color = 'gray',
        linestyle = ':')
# шаг вспомогательной сетки в зависимости от выбранного шага
    ax.xaxis.set_minor_locator(MultipleLocator(t_step_min*10))
