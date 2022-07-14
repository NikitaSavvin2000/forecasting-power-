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
    ax.step(x1, y1, where='post',linewidth = 2, label = 'Medium consumer')
    ax.legend()
    plt.title(
        'Medium consumption per consumer', 
        size = 15,
    )
    ax.set_xlabel('Time format: hh:mm', size = 12)
    ax.set_ylabel('Electricity consumption W',  size = 12)
    ax.tick_params(labelsize = 14)
    #ax.set_ylim(-50, max(y1)+1000)
    ax.set_xlim(-20, 1500)
    ax.set_ylim(-50, 2500)
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




def graf_gistogram_min_max_medium_poison(D, Data_min, Data_max, Data_medium, x, t_step_min):
    x1=x
    print(D['Time'])
    print(len(Data_max))
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
    # размер графика
    ax.set_xlim(-200, 15000)
    ax.tick_params(labelsize = 10)
    #ax.set_xlim(-20, 15000)
    e = 2400 * int(60/t_step_min)
    step = e / 24
    end =  e + step
    ax.set_xlim(-200, end)
    ax.set_xticks(np.arange(0, end, step))
    # сам график
    ax.step(x1, y1, color = 'b', linewidth = 1, label = 'min')
    ax.step(x1, y2, color = 'r', linewidth = 1, label = 'max')
    ax.step(x1, y3, color = 'g', linewidth = 1, label = 'average')
    ax.legend()
    plt.title(
        "The model day electricity consumption using algorithm destribution Poisson", 
        size = 15,
    )
    ax.set_xticklabels(Time_01, fontsize = 8)
    ax.set_xlabel('Time format: hh:mm', size = 12)
    ax.set_ylabel('Electricity consumption kW',  size = 12)
    ax.grid(which='major',
        color = 'k')
    ax.minorticks_on()
    ax.grid(which='minor',
        color = 'gray',
        linestyle = ':')
# шаг вспомогательной сетки в зависимости от выбранного шага
    ax.xaxis.set_minor_locator(MultipleLocator(t_step_min*10))
