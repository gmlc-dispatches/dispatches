import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
import json
import pickle

matplotlib.rc('font', size=24)
plt.rc('axes', titlesize=24)


mrg_csts = [5,10,15,20,25,30]
revenues = [] 
capacity_factors = []
avg_lmps = []
lmps = []
pmax = 177.5
for i in range(6):
    result_dir = 'sensitivity_sweep_over_mrg_cst_0/price_results/mrg_cst_sweep_{}.npy'.format(i)

    with open(result_dir, 'rb') as f:
        dispatch = np.load(f)
        prices = np.load(f)

        cap_factor = sum(dispatch)/(pmax*len(dispatch))
        revenue = (np.dot(dispatch,prices)/1e6)
        avg_lmp = np.mean(prices)

        revenues.append(revenue)
        capacity_factors.append(cap_factor)
        avg_lmps.append(avg_lmp)
        lmps.append(prices)

df_lmps = pd.DataFrame(np.transpose(lmps),columns=mrg_csts)
#df_dispatch = pd.DataFrame(np.transpose(pmax_dispatch),columns=p_max_vector)

#plot lmps
plt.figure(figsize=(8,8))
lmp_plt = df_lmps.boxplot(showmeans=True)
lmp_plt.set_ylim(0,50)
lmp_plt.set_xlabel("Marginal Cost [$/MWh]")
lmp_plt.set_ylabel("LMP [$/MWh]")
plt.tight_layout()
plt.xticks(rotation = 45)
plt.savefig("mrg_cst_lmp_box.png")
# plt.savefig(figure_dir+"/pmax_lmp_box_{}.pdf".format(parm_idx))


#load price taker (vanilla) results 
df = pd.read_pickle('price_taker_revenue_vs_marginal_cost_fixed_pmax.pkl')

mrg_csts_price_taker = df["marginal_cost"].to_numpy()
rev_price_taker = df["annual_revenue"].to_numpy()
cap_price_taker = df["avg_capacity_factor"].to_numpy()/100


#load surrogate results
with open('../rankine_results/scikit_surrogate/mrg_cst_sweep.json', 'r') as infile:
    data = json.load(infile)

mrg_csts_surrogate = data['mrg_csts']
rev_surrogate = data['revenue']
cap_surrogate = data['capacity_factor']

fig,axs = plt.subplots(figsize = (8,8))
axs.set_xlabel("Marginal Cost [$/MWh]")
axs.set_ylabel("Revenue [$MM]")
plt.scatter(mrg_csts,revenues,label = "Prescient", s = 100, alpha = 0.75)
plt.scatter(mrg_csts_surrogate,rev_surrogate,label = "Surrogate", s = 100, alpha = 0.75)
plt.scatter(mrg_csts_price_taker,rev_price_taker,label = "Price Taker", s = 100, alpha = 0.75)
plt.legend()
plt.tight_layout()
plt.savefig("revenue_mrg_cst_sweep.png")


fig,axs = plt.subplots(figsize = (8,8))
axs.set_xlabel("Marginal Cost [$/MWh]")
axs.set_ylabel("Capacity Factor")
plt.scatter(mrg_csts,capacity_factors,label = "Prescient", s = 100, alpha = 0.75)
plt.scatter(mrg_csts_surrogate,cap_surrogate,label = "Surrogate", s = 100, alpha = 0.75)
plt.scatter(mrg_csts_price_taker,cap_price_taker,label = "Price Taker", s = 100, alpha = 0.75)
plt.legend()
plt.tight_layout()
plt.savefig("capfactor_mrg_cst_sweep.png")
