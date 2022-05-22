import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import json
import copy
from smac.env import StarCraft2Env


plot_len = 204
tick_spacing = 500*1000
info_name = "info.json"
config_name = "config.json"
seed_num = 3

config_path_ls = [
    
    # 3s4z
    # "pymarl-master/results/sacred/204/"+config_name, 
    # "pymarl-master/results/sacred/208/"+config_name,       
    # "pymarl-master/results/sacred/212/"+config_name,
    # "pymarl-master/results/sacred/366/"+config_name,
    # "pymarl-master/results/sacred/367/"+config_name, 
    # "pymarl-master/results/sacred/368/"+config_name,       
    # "pymarl-master/results/sacred/795/"+config_name,
    # "pymarl-master/results/sacred/796/"+config_name,
    # "pymarl-master/results/sacred/797/"+config_name,

    # 2c
    "pymarl-master/results/sacred/222/"+config_name, 
    "pymarl-master/results/sacred/223/"+config_name,       
    "pymarl-master/results/sacred/224/"+config_name,
    "pymarl-master/results/sacred/388/"+config_name,
    "pymarl-master/results/sacred/387/"+config_name, 
    "pymarl-master/results/sacred/386/"+config_name,       
    "pymarl-master/results/sacred/798/"+config_name,
    "pymarl-master/results/sacred/799/"+config_name,
    "pymarl-master/results/sacred/800/"+config_name,

    
	# 2c
    # "pymarl-master/results/sacred/222/"+config_name, 
    # "pymarl-master/results/sacred/388/"+config_name,       
    # "pymarl-master/results/sacred/785/"+config_name,
    
    # # 8m
    # "pymarl-master/results/sacred/8m9m01/"+config_name,
    # "pymarl-master/results/sacred/8m9m02/"+config_name, 
    # "pymarl-master/results/sacred/8m9m03/"+config_name,  
    # "pymarl-master/results/sacred/360/"+config_name,
    # "pymarl-master/results/sacred/361/"+config_name,
    # "pymarl-master/results/sacred/362/"+config_name,       
    # "pymarl-master/results/sacred/792/"+config_name,
    # "pymarl-master/results/sacred/793/"+config_name,
    # "pymarl-master/results/sacred/794/"+config_name,

]


info_path_ls = [
    
    # 3s4z
    # "pymarl-master/results/sacred/204/"+info_name, 
    # "pymarl-master/results/sacred/208/"+info_name,       
    # "pymarl-master/results/sacred/212/"+info_name,
    # "pymarl-master/results/sacred/366/"+info_name,
    # "pymarl-master/results/sacred/367/"+info_name, 
    # "pymarl-master/results/sacred/368/"+info_name,       
    # "pymarl-master/results/sacred/795/"+info_name,
    # "pymarl-master/results/sacred/796/"+info_name,
    # "pymarl-master/results/sacred/797/"+info_name,
    
    
    # 2c
    "pymarl-master/results/sacred/222/"+info_name, 
    "pymarl-master/results/sacred/223/"+info_name,       
    "pymarl-master/results/sacred/224/"+info_name,
    "pymarl-master/results/sacred/388/"+info_name,
    "pymarl-master/results/sacred/387/"+info_name, 
    "pymarl-master/results/sacred/386/"+info_name,       
    "pymarl-master/results/sacred/798/"+info_name,
    "pymarl-master/results/sacred/799/"+info_name,
    "pymarl-master/results/sacred/800/"+info_name,


    # stgat 3s4z
    # "pymarl-master/results/sacred/204/"+info_name,        
    # "pymarl-master/results/sacred/366/"+info_name, 
    # "pymarl-master/results/sacred/782/"+info_name,
    # "pymarl-master/results/sacred/783/"+info_name,

	# stgat 2c
    # "pymarl-master/results/sacred/222/"+info_name,        
    # "pymarl-master/results/sacred/388/"+info_name, 
    # "pymarl-master/results/sacred/785/"+info_name,

	# # 8m
    # "pymarl-master/results/sacred/8m9m01/"+info_name,
    # "pymarl-master/results/sacred/8m9m02/"+info_name, 
    # "pymarl-master/results/sacred/8m9m03/"+info_name,  
    # "pymarl-master/results/sacred/360/"+info_name,
    # "pymarl-master/results/sacred/361/"+info_name,
    # "pymarl-master/results/sacred/362/"+info_name,       
    # "pymarl-master/results/sacred/792/"+info_name,
    # "pymarl-master/results/sacred/793/"+info_name,
    # "pymarl-master/results/sacred/794/"+info_name,

                       
]


def getdata():
    data = []
    alg_name = []
    scen_name = []
    for idx, config_path in enumerate(config_path_ls):
        with open(config_path, 'r') as f:
            config = json.load(f)
            if config["env_args"]["map_name"] not in scen_name:
                scen_name.append(config["env_args"]["map_name"])
            if config["name"] not in alg_name:
                alg_name.append(config["name"]) 

    info_count = 0
    for scen_i in range(len(scen_name)):
        data.append([])
        for alg_i in range(len(alg_name)):
            data[scen_i].append([])
            for seed_i in range(seed_num):
                with open(info_path_ls[info_count], 'r') as f:
                    info = json.load(f)
                    data[scen_i][alg_i].append(info["test_battle_won_mean"][:plot_len])
                info_count += 1
    return data, alg_name, scen_name

def plot_figure():
    data, alg_name, scen_name = getdata()
    xdata = [10000 * i for i in range(plot_len)]
    linestyle = ['-', '--', ':', '-.']
    color = ['r', 'g', 'b', 'k']
    sns.set_style(style = "whitegrid")
    for scen_i in range(len(scen_name)):
        for alg_i in range(len(alg_name)):
            for seed_i in range(len(data[scen_i][alg_i])):
                for idx in range(len(data[scen_i][alg_i][seed_i])):
                    if idx < 3:
                        data[scen_i][alg_i][seed_i][idx] = np.mean(data[scen_i][alg_i][seed_i][:idx+1])
                    else:
                        data[scen_i][alg_i][seed_i][idx] = np.mean(data[scen_i][alg_i][seed_i][idx-3:idx+1])

    for scen_i in range(len(scen_name)):
        fig = plt.figure()
        for alg_i in range(len(alg_name)):
            ax = sns.tsplot(time=xdata, data=data[scen_i][alg_i], color=color[alg_i], linestyle=linestyle[alg_i], condition=alg_name[alg_i])

        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        ax.axes.xaxis.set_ticks([500*1000, 1000*1000 , 1500*1000, 2000*1000]) 
        ax.axes.set_xticklabels(['500.0k', '1.0m', '1.5m', '2.0m'])
        plt.xlabel("T", fontsize=15)
        plt.ylabel("Test Win Rate %", fontsize=15)
        plt.title('{}'.format(scen_name[scen_i]), fontsize=15)
        plt.savefig('pymarl-master/results/fig/{}.jpg'.format(scen_name[scen_i]))


if __name__ == "__main__":
    plot_figure()
    