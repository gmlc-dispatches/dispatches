import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
import itertools
matplotlib.rc('font', size=32)
plt.rc('axes', titlesize=32)

# p_max_vector = [175,275,350,450]
# mrg_vector = [5,10,15,25]
# test_inputs = list(itertools.product(p_max_vector,mrg_vector))

p_max_vector = [175,275,350,450]
mrg_vector = [16, 19, 22, 24]

test_inputs = list(itertools.product(p_max_vector,mrg_vector))


parm_dicts = []
for i in range(16):
    result_dir = 'pmax_grid_search_3/price_results/rts_results_grid_search_{}.npy'.format(i)


    with open(result_dir, 'rb') as f:

        pmax = test_inputs[i][0]
        dispatch = np.load(f)
        prices = np.load(f)
        n_startups = int(np.load(f))
        cap_factor = sum(dispatch)/(pmax*len(dispatch))
        revenue = (np.dot(dispatch,prices)/1e6)

        parm_dict = {}
        parm_dict['cap_factor'] = cap_factor
        parm_dict['revenue'] = revenue
        parm_dict['n_startups'] = n_startups
        parm_dict['mrg_cst'] = test_inputs[i][1]
        parm_dict['pmax'] = test_inputs[i][0]
        parm_dicts.append(parm_dict)


colors = matplotlib.cm.tab10(range(8))
#markers = matplotlib.lines.Line2D.markers
markers = ['o','v','^','<','>','D','s','*']


colors_list = ['blue','purple','orange','red']
colors = dict(zip(mrg_vector,colors_list))
# colors = {5:'blue',10:'purple',15:'orange',25:'red'}


#revenue
fig,axs = plt.subplots(figsize = (8,8))
axs.set_xlabel("$P_{max}$ [MW]")
axs.set_ylabel("Revenue [MM$]")

for i in range(16):
    revenue = parm_dicts[i]['revenue']
    mrg_cst = parm_dicts[i]['mrg_cst']
    axs.scatter([parm_dicts[i]['pmax']],[revenue],color=colors[mrg_cst],alpha = 0.8,s = 120)

from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], lw=4, color='w', marker='o',label = 'mrg_cst = {}'.format(mrg_vector[0]), markerfacecolor='blue', markersize=15),
                Line2D([0], [0], lw=4, color='w',marker='o',label = 'mrg_cst = {}'.format(mrg_vector[1]), markerfacecolor='purple', markersize=15),
                Line2D([0], [0],  lw=4, color='w',marker='o',label = 'mrg_cst = {}'.format(mrg_vector[2]), markerfacecolor='orange', markersize=15),
                Line2D([0], [0],  lw=4, color='w',marker='o',label = 'mrg_cst = {}'.format(mrg_vector[3]), markerfacecolor='red', markersize=15)
                ]
axs.legend(handles=legend_elements, loc='upper left',fontsize = 12)
fig.subplots_adjust(left=0.2, bottom=0.15, right=0.95, top=0.95)  
plt.savefig("pmax_grid_search_3/revenue_search.png")
#plt.savefig("pmax_sweep_runs/revenue_sweeps.pdf")

#capacity factor
fig,axs = plt.subplots(figsize = (8,8))
axs.set_xlabel("$P_{max}$ [MW]")
axs.set_ylabel("Capacity Factor")

for i in range(0,16):
    cap_factor = parm_dicts[i]['cap_factor']
    mrg_cst = parm_dicts[i]['mrg_cst']
    axs.scatter([parm_dicts[i]['pmax']],[cap_factor],color=colors[mrg_cst],alpha = 0.8,s = 120)

from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], lw=4, color='w', marker='o',label = 'mrg_cst = {}'.format(mrg_vector[0]), markerfacecolor='blue', markersize=15),
                Line2D([0], [0], lw=4, color='w',marker='o',label = 'mrg_cst = {}'.format(mrg_vector[1]), markerfacecolor='purple', markersize=15),
                Line2D([0], [0],  lw=4, color='w',marker='o',label = 'mrg_cst = {}'.format(mrg_vector[2]), markerfacecolor='orange', markersize=15),
                Line2D([0], [0],  lw=4, color='w',marker='o',label = 'mrg_cst = {}'.format(mrg_vector[3]), markerfacecolor='red', markersize=15)
                ]
axs.legend(handles=legend_elements, loc='upper left',fontsize = 12)
fig.subplots_adjust(left=0.2, bottom=0.15, right=0.95, top=0.95)  
plt.savefig("pmax_grid_search_3/cap_factor_search.png")


# fig,axs = plt.subplots(figsize = (8,8))
# axs.set_xlabel("$P_{max}$ [MW]")
# axs.set_ylabel("Capacity Factor")

# for p in range(len(parm_dicts)):
#     cap_fac = parm_dicts[p]['cap_factor']
#     axs.scatter(p_max_vector,cap_fac, label="Scenario {}".format(p+1),color=colors[p],marker=markers[p],alpha = 0.8,s = 120)
# #axs.legend(loc='upper left',fontsize=12)
# fig.subplots_adjust(left=0.2, bottom=0.15, right=0.95, top=0.95)  
# plt.savefig("pmax_sweep_runs/cap_fac_sweeps.png")
# plt.savefig("pmax_sweep_runs/cap_fac_sweeps.pdf")

#nstartups
# fig,axs = plt.subplots(figsize = (8,8))
# axs.set_xlabel("$P_{max}$ [MW]")
# axs.set_ylabel("Number of Startups")

# for p in range(len(parm_dicts)):
#     n_startups = parm_dicts[p]['n_startups']
#     axs.scatter(p_max_vector,n_startups, label="Scenario {}".format(p+1),color=colors[p],marker=markers[p],alpha = 0.8,s = 120)
# #axs.legend(loc='upper left',fontsize=12)
# fig.subplots_adjust(left=0.2, bottom=0.15, right=0.95, top=0.95)      
# plt.savefig("pmax_sweep_runs/nstartups_sweeps.png")
# plt.savefig("pmax_sweep_runs/nstartups_sweeps.pdf")