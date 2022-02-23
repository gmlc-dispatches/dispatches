#this script plots 3 summary metrics for each set of Prescient simulations. Namely:
#- Annual Revenue: MM$
#- Annual Capacity factor: MWh (delivered) / MWh (if at 100% all time)
#- Annual number of startups

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rc('font', size=32)
plt.rc('axes', titlesize=32)

#using these runs:
parameter_indices = [1,2,10,11,12,13,14,15]
p_max_vector = np.arange(175,450,25)
parm_dicts = []
for i in range(len(parameter_indices)):
    parm_idx = parameter_indices[i]
    base_dir = 'pmax_sweep_runs/sensitivity_sweep_over_pmax_{}'.format(parm_idx)
    result_dir = base_dir+'/pmax_sweep_prices_{}'.format(parm_idx)

    pmax_revenue = []
    pmax_cap_factor = []
    pmax_nstartups = []
    for pmax in p_max_vector:
        with open(result_dir +'/rts_results_all_prices_pmax_{}.npy'.format(pmax), 'rb') as f:
            dispatch = np.load(f)
            prices = np.load(f)
            n_startups = int(np.load(f))
        pmax_cap_factor.append(sum(dispatch)/(pmax*len(dispatch)))
        pmax_revenue.append(np.dot(dispatch,prices)/1e6)
        pmax_nstartups.append(n_startups)

    parm_dict = {}
    parm_dict['cap_factor'] = pmax_cap_factor
    parm_dict['revenue'] = pmax_revenue
    parm_dict['n_startups'] = pmax_nstartups
    parm_dicts.append(parm_dict)


colors = matplotlib.cm.tab10(range(8))
#markers = matplotlib.lines.Line2D.markers
markers = ['o','v','^','<','>','D','s','*']


#revenue
fig,axs = plt.subplots(figsize = (8,8))
axs.set_xlabel("$P_{max}$ [MW]")
axs.set_ylabel("Revenue [MM$]")

for p in range(len(parm_dicts)):
    revenue = parm_dicts[p]['revenue']
    axs.scatter(p_max_vector,revenue, label="Scenario {}".format(p+1),color=colors[p],marker=markers[p],alpha = 0.8,s = 120)
    # for x, y in zip(p_max_vector, revenue):
    #     axs.text(x, y, str(p+1), color=colors[p], fontsize=12)
axs.legend(loc='upper left',fontsize=24)
fig.subplots_adjust(left=0.2, bottom=0.15, right=0.95, top=0.95)  
plt.savefig("pmax_sweep_runs/revenue_sweeps.png")
plt.savefig("pmax_sweep_runs/revenue_sweeps.pdf")

#capacity factor
fig,axs = plt.subplots(figsize = (8,8))
axs.set_xlabel("$P_{max}$ [MW]")
axs.set_ylabel("Capacity Factor")

for p in range(len(parm_dicts)):
    cap_fac = parm_dicts[p]['cap_factor']
    axs.scatter(p_max_vector,cap_fac, label="Scenario {}".format(p+1),color=colors[p],marker=markers[p],alpha = 0.8,s = 120)
#axs.legend(loc='upper left',fontsize=12)
fig.subplots_adjust(left=0.2, bottom=0.15, right=0.95, top=0.95)  
plt.savefig("pmax_sweep_runs/cap_fac_sweeps.png")
plt.savefig("pmax_sweep_runs/cap_fac_sweeps.pdf")

#nstartups
fig,axs = plt.subplots(figsize = (8,8))
axs.set_xlabel("$P_{max}$ [MW]")
axs.set_ylabel("Number of Startups")

for p in range(len(parm_dicts)):
    n_startups = parm_dicts[p]['n_startups']
    axs.scatter(p_max_vector,n_startups, label="Scenario {}".format(p+1),color=colors[p],marker=markers[p],alpha = 0.8,s = 120)
#axs.legend(loc='upper left',fontsize=12)
fig.subplots_adjust(left=0.2, bottom=0.15, right=0.95, top=0.95)      
plt.savefig("pmax_sweep_runs/nstartups_sweeps.png")
plt.savefig("pmax_sweep_runs/nstartups_sweeps.pdf")