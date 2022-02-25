import pandas as pd
import numpy as np
from pyomo.common.fileutils import this_file_dir
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', size=24)
plt.rc('axes', titlesize=18)
import seaborn as sns
import os


base_dir = os.path.join(this_file_dir(),"_prescient_sweep_manuscript_figure/prescient_sweep_subset/pmax_433")

pmax_high = [[52219, 52939, 53659, 54379, 55099, 55819],
 [39259, 39979, 40699, 41419, 42139, 42859],
 [26299, 27019, 27739, 28459, 29179, 29899],
 [13339, 14059, 14779, 15499, 16219, 16939],
 [379, 1099, 1819, 2539, 3259, 3979]]

start_profiles = [0,1,2,3,4]
mrg_csts = [5,10,15,20,25,30]

#append everything into one dataframe
df = pd.DataFrame(columns = ["Hour","Startup Profile","Marginal Cost","LMP"])
for (start_profile,sim_vector) in enumerate(pmax_high):
    for (i,sim_index) in enumerate(sim_vector):
        mrg_cst = mrg_csts[i]
        result_dir = os.path.join(base_dir,'deterministic_simulation_output_index_{}'.format(sim_index))
        bus_detail_df = pd.read_csv(os.path.join(result_dir,'bus_detail.csv'))
        bus_results = bus_detail_df.loc[bus_detail_df['Bus'] == "CopperSheet"]
        lmp = np.copy(bus_results["LMP"])
        lmp[lmp > 200] = 200
        hours = np.arange(0,len(lmp))
        mrg_vector = np.ones(len(lmp))*mrg_cst
        start_vector = np.array(["Start Profile {}".format(int(i)) for i in np.ones(len(lmp))*start_profile])
        df_i = pd.DataFrame({'Hour':hours,'Startup Profile':start_vector,'Marginal Cost':mrg_vector,'LMP':lmp})
        df = df.append(df_i)



plt.figure(figsize=(16,8))
lmp_plt = sns.boxplot(y='LMP',x='Startup Profile',data=df,hue='Marginal Cost')
lmp_plt.set_ylim(0,50)
lmp_plt.set_ylabel("LMP [$/MWh]")
lmp_plt.set_xlabel("")
plt.tight_layout()
plt.legend(loc=2, prop={'size': 24},title = "Marginal Cost [$/MWh]")
plt.savefig("LMP_boxplot.png")
plt.savefig("LMP_boxplot.pdf")
