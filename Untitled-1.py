def graf_gistogram_poison(D, data_poisson, time_index_interval, t_step_min):
    df1 = D
    x1=time_index_interval
    # находит среднее значение P в заданном промежутке в зависимости от количества файлов
    Time_0 = np.array(df1['Time'], dtype=str)
    step = int(60/t_step_min)
    Time_01 = [Time_0[i] for i in range(0 ,len(Time_0), step)]

    # размер графика
    fig = plt.figure(figsize=(20,15))
    ax = fig.add_subplot()

    # сам график
    for i in range(len(data_poisson)):
        y1 = data_poisson[i]
        ax.step(x1, y1, color = 'r', linewidth = 1)
    
    ax.set_xlabel('Время', size = 24)
    ax.set_ylabel('Мощность при различном одновременном включении (распределение Пуассона) кВт', size = 8)
    ax.tick_params(labelsize = 14)
    #ax.set_ylim(-50, max(y1)+1000)
    ax.set_xlim(-20, 1500)
    #ax.set_ylim(-50, 1500)
    ax.set_ylim(-50, 2000)
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
    
    
    
    
    
    
    
    
    
    
def poisson_all(N,I):
    p_expected_list = []
    Data_consumer = pd.read_excel(r'C:\\Users\\nsavvin\Desktop\\programm2\\Count result.xlsx')
    Data_list_interval = pd.read_excel(r'C:\\Users\\nsavvin\Desktop\\programm2\\random_list_interval.xlsx')
    index_time_normal = Data_list_interval['index_time_normal']
    format_time = Data_list_interval['format_time']
    p_list = Data_list_interval['p_list']
    index_on_list = Data_consumer['index_on']
    index_off_list = Data_consumer['index_off']
    count_consumer_list = Data_consumer['Count']
    poisson_list_all =[]
    poisson_list =[]
    p_expected_poisson_list_all = []
    for i in range(len(count_consumer_list)):
        index_on = index_on_list[i]*N
        index_off = index_off_list[i]*N
        count_consumer_medium = count_consumer_list[i]
        count_index = index_off - index_on
        s = np.random.poisson(count_consumer_medium, count_index).tolist()
        poisson_list.extend(s)
    for q in range(len(poisson_list_all)):
        p_expected = (p_list[q])*(poisson_list[q])
        poisson_list_all.append(p_expected)
    p_expected_poisson_list_all.append(p_expected_list)
    return index_time_normal, poisson_list_all