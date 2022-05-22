import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import json
import copy
from smac.env import StarCraft2Env


plot_len = 201
tick_spacing = 500*1000
info_name = "info.json"
config_name = "config.json"
seed_num = 3

config_path_ls = [
    # "pymarl-master/results/sacred/01/"+config_name,        
    # "pymarl-master/results/sacred/02/"+config_name,
    # "pymarl-master/results/sacred/03/"+config_name,
    # "pymarl-master/results/sacred/95/"+config_name,        
    # "pymarl-master/results/sacred/96/"+config_name,
    # "pymarl-master/results/sacred/97/"+config_name,
    # "pymarl-master/results/sacred/04/"+config_name,        
    # "pymarl-master/results/sacred/05/"+config_name,
    # "pymarl-master/results/sacred/06/"+config_name,
    # "pymarl-master/results/sacred/98/"+config_name,        
    # "pymarl-master/results/sacred/99/"+config_name,
    # "pymarl-master/results/sacred/100/"+config_name, 
    # "pymarl-master/results/sacred/7/"+config_name,        
    # "pymarl-master/results/sacred/8/"+config_name,
    # "pymarl-master/results/sacred/15/"+config_name,
    # "pymarl-master/results/sacred/23/"+config_name,        
    # "pymarl-master/results/sacred/24/"+config_name,
    # "pymarl-master/results/sacred/25/"+config_name,
    # "pymarl-master/results/sacred/094/"+config_name,        
    # "pymarl-master/results/sacred/095/"+config_name,
    # "pymarl-master/results/sacred/096/"+config_name,         
    # "pymarl-master/results/sacred/11/"+config_name,       
    # "pymarl-master/results/sacred/12/"+config_name,
    # "pymarl-master/results/sacred/17/"+config_name,
    # "pymarl-master/results/sacred/20/"+config_name,        
    # "pymarl-master/results/sacred/21/"+config_name,
    # "pymarl-master/results/sacred/091/"+config_name,        
    # "pymarl-master/results/sacred/092/"+config_name,
    # "pymarl-master/results/sacred/093/"+config_name,
    # "pymarl-master/results/sacred/22/"+config_name,         
    # "pymarl-master/results/sacred/13/"+config_name,         
    # "pymarl-master/results/sacred/14/"+config_name,
    # "pymarl-master/results/sacred/16/"+config_name,
    # "pymarl-master/results/sacred/26/"+config_name,        
    # "pymarl-master/results/sacred/27/"+config_name,
    # "pymarl-master/results/sacred/28/"+config_name,

    # 0510 MMM2
    "pymarl-master/results/sacred/7/"+config_name,          # MMM2 seed 01 QMIX
    "pymarl-master/results/sacred/8/"+config_name,          # MMM2 seed 02 QMIX
    "pymarl-master/results/sacred/15/"+config_name,         # MMM2 seed 03 QMIX
    # "pymarl-master/results/sacred/167/"+config_name,         # MMM2 seed 01 Graph_QMIX
    # "pymarl-master/results/sacred/168/"+config_name,         # MMM2 seed 02 Graph_QMIX
    # "pymarl-master/results/sacred/169/"+config_name,         # MMM2 seed 03 Graph_QMIX
    # "pymarl-master/results/sacred/201/"+config_name,        
    # "pymarl-master/results/sacred/202/"+config_name,
    # "pymarl-master/results/sacred/203/"+config_name,
    # "pymarl-master/results/sacred/0199/"+config_name,
    # "pymarl-master/results/sacred/0200/"+config_name,
    # "pymarl-master/results/sacred/0201/"+config_name,
    "pymarl-master/results/sacred/363/"+config_name,          # MMM2 seed 01 QMIX
    "pymarl-master/results/sacred/364/"+config_name,          # MMM2 seed 02 QMIX
    "pymarl-master/results/sacred/365/"+config_name,    

    # "pymarl-master/results/sacred/222/"+config_name,         # MMM2 seed 03 Graph_QMIX
    # "pymarl-master/results/sacred/223/"+config_name,
    # "pymarl-master/results/sacred/224/"+config_name,
    # "pymarl-master/results/sacred/225/"+config_name,
    # "pymarl-master/results/sacred/228/"+config_name,
    # "pymarl-master/results/sacred/229/"+config_name,
    # "pymarl-master/results/sacred/230/"+config_name,
    # "pymarl-master/results/sacred/231/"+config_name,
    # "pymarl-master/results/sacred/232/"+config_name,
    # "pymarl-master/results/sacred/233/"+config_name,
    # "pymarl-master/results/sacred/234/"+config_name,
    # "pymarl-master/results/sacred/235/"+config_name,

    # 5m6m
    # "pymarl-master/results/sacred/260/"+config_name,        
    # "pymarl-master/results/sacred/261/"+config_name,
    # "pymarl-master/results/sacred/262/"+config_name,
                       
    # stgat mmm2
    # "pymarl-master/results/sacred/770/"+config_name,        
    # "pymarl-master/results/sacred/771/"+config_name,
    # "pymarl-master/results/sacred/776/"+config_name,
    
    # "pymarl-master/results/sacred/773/"+config_name,        
    # "pymarl-master/results/sacred/774/"+config_name,
    # "pymarl-master/results/sacred/775/"+config_name,
    
    "pymarl-master/results/sacred/789/"+config_name,        
    "pymarl-master/results/sacred/790/"+config_name,
    "pymarl-master/results/sacred/791/"+config_name,
]


info_path_ls = [
    # "pymarl-master/results/sacred/01/"+info_name,        # 1c3s5z QMIX
    # "pymarl-master/results/sacred/02/"+info_name,
    # "pymarl-master/results/sacred/03/"+info_name,
    # "pymarl-master/results/sacred/95/"+info_name,        #1c3s5z Graph_QMIX
    # "pymarl-master/results/sacred/96/"+info_name,
    # "pymarl-master/results/sacred/97/"+info_name,
    # "pymarl-master/results/sacred/04/"+info_name,        # 3s5z_vs_3s6z QMIX
    # "pymarl-master/results/sacred/05/"+info_name,       
    # "pymarl-master/results/sacred/06/"+info_name,
    # "pymarl-master/results/sacred/98/"+info_name,        # 3s5z_vs_3s6z Graph_QMIX
    # "pymarl-master/results/sacred/99/"+info_name,
    # "pymarl-master/results/sacred/100/"+info_name, 
    # "pymarl-master/results/sacred/7/"+info_name,          # MMM2 seed 01 QMIX
    # "pymarl-master/results/sacred/8/"+info_name,          # MMM2 seed 02 QMIX
    # "pymarl-master/results/sacred/15/"+info_name,         # MMM2 seed 03 QMIX
    # "pymarl-master/results/sacred/89/"+info_name,         # MMM2 seed 01 Graph_QMIX
    # "pymarl-master/results/sacred/90/"+info_name,         # MMM2 seed 02 Graph_QMIX
    # "pymarl-master/results/sacred/91/"+info_name,         # MMM2 seed 03 Graph_QMIX
    # "pymarl-master/results/sacred/094/"+info_name,         # multihead 
    # "pymarl-master/results/sacred/095/"+info_name,       
    # "pymarl-master/results/sacred/096/"+info_name,
    # "pymarl-master/results/sacred/11/"+info_name,         # 2s3z seed 01 QMIX
    # "pymarl-master/results/sacred/12/"+info_name,         # 2s3z seed 02 QMIX
    # "pymarl-master/results/sacred/17/"+info_name,         # 2s3z seed 02 QMIX
    # "pymarl-master/results/sacred/86/"+info_name,         # 2s3z seed 01 Graph_QMIX
    # "pymarl-master/results/sacred/87/"+info_name,         # 2s3z seed 02 Graph_QMIX
    # "pymarl-master/results/sacred/88/"+info_name,         # 2s3z seed 02 Graph_QMIX
    # "pymarl-master/results/sacred/091/"+info_name,        # multihead
    # "pymarl-master/results/sacred/092/"+info_name,
    # "pymarl-master/results/sacred/093/"+info_name,
    # "pymarl-master/results/sacred/13/"+info_name,         # 8m   seed 01 QMIX
    # "pymarl-master/results/sacred/14/"+info_name,         # 8m   seed 02 QMIX
    # "pymarl-master/results/sacred/16/"+info_name,         # 8m   seed 03 QMIX
    # "pymarl-master/results/sacred/92/"+info_name,         # 8m   seed 01 Graph_QMIX
    # "pymarl-master/results/sacred/93/"+info_name,         # 8m   seed 02 Graph_QMIX
    # "pymarl-master/results/sacred/94/"+info_name,         # 8m   seed 03 Graph_QMIX
    
    # 0510 MMM2
    "pymarl-master/results/sacred/7/"+info_name,          # MMM2 seed 01 QMIX
    "pymarl-master/results/sacred/8/"+info_name,          # MMM2 seed 02 QMIX
    "pymarl-master/results/sacred/15/"+info_name,         # MMM2 seed 03 QMIX
    # "pymarl-master/results/sacred/167/"+info_name,         # MMM2 seed 01 Graph_QMIX
    # "pymarl-master/results/sacred/168/"+info_name,         # MMM2 seed 02 Graph_QMIX
    # "pymarl-master/results/sacred/169/"+info_name,         # MMM2 seed 03 Graph_QMIX
    # "pymarl-master/results/sacred/201/"+info_name,        
    # "pymarl-master/results/sacred/202/"+info_name,
    # "pymarl-master/results/sacred/203/"+info_name,
    # "pymarl-master/results/sacred/0199/"+info_name,
    # "pymarl-master/results/sacred/0200/"+info_name,
    # "pymarl-master/results/sacred/0201/"+info_name, 
    "pymarl-master/results/sacred/363/"+info_name,          # MMM2 seed 01 QMIX
    "pymarl-master/results/sacred/364/"+info_name,          # MMM2 seed 02 QMIX
    "pymarl-master/results/sacred/365/"+info_name,   


    # # 2c vs 64zg 0524
    # "pymarl-master/results/sacred/222/"+info_name,         # MMM2 seed 03 Graph_QMIX
    # "pymarl-master/results/sacred/223/"+info_name,
    # "pymarl-master/results/sacred/224/"+info_name,
    # "pymarl-master/results/sacred/225/"+info_name,
    # "pymarl-master/results/sacred/228/"+info_name,
    # "pymarl-master/results/sacred/229/"+info_name,
    # "pymarl-master/results/sacred/230/"+info_name,
    # "pymarl-master/results/sacred/231/"+info_name,
    # "pymarl-master/results/sacred/232/"+info_name,
    # "pymarl-master/results/sacred/233/"+info_name,
    # "pymarl-master/results/sacred/234/"+info_name,
    # "pymarl-master/results/sacred/235/"+info_name,

    # stgat mmm2
    # "pymarl-master/results/sacred/770/"+info_name,        
    # "pymarl-master/results/sacred/771/"+info_name,
    # "pymarl-master/results/sacred/776/"+info_name,
    
    "pymarl-master/results/sacred/789/"+info_name,        
    "pymarl-master/results/sacred/790/"+info_name,
    "pymarl-master/results/sacred/791/"+info_name,
    
    # "pymarl-master/results/sacred/773/"+info_name,        
    # "pymarl-master/results/sacred/774/"+info_name,
    # "pymarl-master/results/sacred/775/"+info_name,

                       
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
    