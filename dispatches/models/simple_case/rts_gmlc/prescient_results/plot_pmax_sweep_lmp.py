import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rc('font', size=32)
plt.rc('axes', titlesize=32)

#using runs:
#1,2,10,11,12,13,14,15
parameter_indices = [1,2,10,11,12,13,14,15]
#parameter_indices = [1]
p_max_vector = np.arange(175,450,25)

for parm_idx in parameter_indices:
#parm_idx = 15

	base_dir = 'pmax_sweep_runs/sensitivity_sweep_over_pmax_{}'.format(parm_idx)
	result_dir = base_dir+'/pmax_sweep_prices_{}'.format(parm_idx)
	figure_dir = 'pmax_sweep_runs/sweep_figures'

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
	plt.figure(figsize=(8,8))
	lmp_plt = df_lmps.boxplot(showmeans=True)
	lmp_plt.set_ylim(0,50)
	lmp_plt.set_xlabel("$P_{max}$ [MW]")
	lmp_plt.set_ylabel("LMP [$/MWh]")
	plt.tight_layout()
	plt.xticks(rotation = 45)
	plt.savefig(figure_dir+"/pmax_lmp_box_{}.png".format(parm_idx))
	plt.savefig(figure_dir+"/pmax_lmp_box_{}.pdf".format(parm_idx))

	#plot dispatch
	plt.figure(figsize=(8,8))
	dispatch_box = df_dispatch.boxplot(showmeans=True)
	dispatch_box.set_ylim(0,450)
	dispatch_box.set_xlabel("$P_{max}$ [MW]")
	dispatch_box.set_ylabel("Dispatch [MWh]")
	plt.tight_layout()
	plt.xticks(rotation = 45)
	plt.savefig(figure_dir+"/pmax_dispatch_box_{}.png".format(parm_idx))
	plt.savefig(figure_dir+"/pmax_dispatch_box_{}.pdf".format(parm_idx))
