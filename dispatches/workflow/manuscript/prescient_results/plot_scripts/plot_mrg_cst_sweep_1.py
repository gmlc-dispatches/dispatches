import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
import itertools
import json
matplotlib.rc('font', size=32)
plt.rc('axes', titlesize=32)

p_max_vector = [175,275,350,450]
mrg_vector1 = [16,19,22,24]
mrg_vector2 = [5,10,15,25]

test_inputs = list(itertools.product(p_max_vector,mrg_vector2))

mrg_csts = []
revenues = [] 
capacity_factors = []
for i in range(16):
    # result_dir = 'pmax_grid_search_3/price_results/rts_results_grid_search_{}.npy'.format(i)
    result_dir = 'pmax_grid_search/price_results/rts_results_grid_serach_{}.npy'.format(i)

    with open(result_dir, 'rb') as f:

        pmax = test_inputs[i][0]
        if pmax == 175:
            dispatch = np.load(f)
            prices = np.load(f)
            #n_startups = int(np.load(f))
            cap_factor = sum(dispatch)/(pmax*len(dispatch))
            revenue = (np.dot(dispatch,prices)/1e6)

            revenues.append(revenue)
            capacity_factors.append(cap_factor)
            mrg_csts.append(test_inputs[i][1])

#load surrogate results
with open('../rankine_results/scikit_surrogate/mrg_cst_sweep.json', 'r') as infile:
    data = json.load(infile)

mrg_csts_surrogate = data['mrg_csts']
rev_surrogate = data['revenue']
cap_surrogate = data['capacity_factor']

fig,axs = plt.subplots(figsize = (8,8))
axs.set_xlabel("Marginal Cost [$/MWh]")
axs.set_ylabel("Revenue [$MM]")
plt.scatter(mrg_csts,revenues,label = "Prescient", s = 48)
plt.scatter(mrg_csts_surrogate,rev_surrogate,label = "Surrogate", s=48)
plt.legend()
plt.tight_layout()
plt.savefig("revenue_mrg_cst_sweep.png")


fig,axs = plt.subplots(figsize = (8,8))
axs.set_xlabel("Marginal Cost [$/MWh]")
axs.set_ylabel("Capacity Factor")
plt.scatter(mrg_csts,capacity_factors,label = "Prescient", s = 48)
plt.scatter(mrg_csts_surrogate,cap_surrogate,label = "Surrogate", s = 48)
plt.legend()
plt.tight_layout()
plt.savefig("capfactor_mrg_cst_sweep.png")
