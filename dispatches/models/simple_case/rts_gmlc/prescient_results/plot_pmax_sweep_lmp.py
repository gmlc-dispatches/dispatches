import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

p_max_vector = np.arange(175,450,25)
parm_idx = 2

base_dir = 'pmax_sweep_runs/sensitivity_sweep_over_pmax_{}'.format(parm_idx)
result_dir = base_dir+'/pmax_sweep_prices_{}'.format(parm_idx)

pmax_lmps = []
pmax_dispatch = []
for pmax in p_max_vector:
	with open(result_dir +'/rts_results_all_prices_pmax_{}.npy'.format(pmax), 'rb') as f:
	    dispatch = np.load(f)
	    prices = np.load(f)
	pmax_dispatch.append(dispatch)
	pmax_lmps.append(prices)

df_lmps = pd.DataFrame(np.transpose(pmax_lmps),columns=p_max_vector)
df_dispatch = pd.DataFrame(np.transpose(pmax_dispatch),columns=p_max_vector)

#plot lmps
plt.figure()
lmp_plt = df_lmps.boxplot(showmeans=True)
lmp_plt.set_ylim(0,50)
lmp_plt.set_xlabel("Pmax")
lmp_plt.set_ylabel("LMP [$/MWh]")
plt.savefig(result_dir+"/pmax_lmp_box.png")

plt.figure()
axs = df_lmps.hist(bins=100)
for ax in axs.flatten():
	ax.set_xlim((0,50))
plt.tight_layout()
plt.savefig(result_dir+"/pmax_lmp_histograms.png")

#plot dispatch
plt.figure()
dispatch_box = df_dispatch.boxplot()
dispatch_box.set_xlabel("Pmax")
dispatch_box.set_ylabel("Dispatch [MWh]")
plt.savefig(result_dir+"/pmax_dispatch_box.png")

plt.figure()
axs = df_dispatch.hist(bins=100,figsize = (12,12))
plt.tight_layout()
plt.savefig(result_dir+"/pmax_dispatch_histograms.png")

#plot revenue
