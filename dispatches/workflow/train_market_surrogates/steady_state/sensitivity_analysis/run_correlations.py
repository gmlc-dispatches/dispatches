#################################################################################
# DISPATCHES was produced under the DOE Design Integration and Synthesis
# Platform to Advance Tightly Coupled Hybrid Energy Systems program (DISPATCHES),
# and is copyright (c) 2021 by the software owners: The Regents of the University
# of California, through Lawrence Berkeley National Laboratory, National
# Technology & Engineering Solutions of Sandia, LLC, Alliance for Sustainable
# Energy, LLC, Battelle Energy Alliance, LLC, University of Notre Dame du Lac, et
# al. All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. Both files are also available online at the URL:
# "https://github.com/gmlc-dispatches/dispatches".
#################################################################################

#Calculate correlations on unfiltered data
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('font', size=32)
plt.rc('axes', titlesize=32)     # fontsize of the axes title
import numpy as np
import seaborn as sns
import scipy
import pingouin as pg

f_perturbed_inputs = os.path.join(os.getcwd(),"../prescient_data/prescient_generator_inputs.h5")
f_perturbed_outputs = os.path.join(os.getcwd(),"../prescient_data/prescient_generator_outputs.h5")
f_startups = os.path.join(os.getcwd(),"../prescient_data/prescient_generator_startups.h5")

df_perturbed_inputs = pd.read_hdf(f_perturbed_inputs)
df_perturbed_outputs = pd.read_hdf(f_perturbed_outputs)
df_nstartups = pd.read_hdf(f_startups)

#CORRELATIONS
#fix the index so we can concat the dataframes
df_perturbed_outputs.index = df_perturbed_outputs.iloc[:,0]
# output_subset = ['Total Dispatch [MW]', 'Total Revenue [$]']
output_subset = ['Total Revenue [$]']
gen_in_out = pd.concat([df_perturbed_inputs.iloc[:,1:9],df_perturbed_outputs[output_subset],df_nstartups.iloc[:,1]],axis = 1)

plt.clf()
cmap = sns.color_palette("vlag")
sns.set(font_scale=6.0)

labels = ["X{}".format(i) for i in range(1,9)]
labels.append("Y1")
labels.append("Y2")
gen_in_out.columns = labels

# PEARSON
# for corr_option in corr_options:
corr_pearson =  gen_in_out.corr(method="pearson")
np.savetxt('pearson.csv', corr_pearson.round(2), delimiter='&',fmt='%.2g')
plt.clf()
hmap = sns.heatmap(corr_pearson,xticklabels=True, yticklabels=True,cmap = cmap,vmin = -1,vmax = 1)
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(22,22)
plt.tight_layout()
plt.yticks(rotation=0)
plt.savefig("correlation_figures/fig_correlation_matrix_{}.png".format("pearson"))
plt.savefig("correlation_figures/fig_correlation_matrix_{}.pdf".format("pearson"))

#Just the outputs
plt.clf()
hmap = sns.heatmap(corr_pearson.iloc[8:10,:],xticklabels=True, yticklabels=True,cmap = cmap,vmin = -1,vmax = 1,annot = True,annot_kws={"size": 32},fmt = ".2f")
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(24, 8)
plt.tight_layout()
plt.yticks(rotation=0)
plt.savefig("correlation_figures/fig_correlation_matrix_{}_outputs.png".format("pearson"))
plt.savefig("correlation_figures/fig_correlation_matrix_{}_outputs.pdf".format("pearson"))

#SPEARMAN
corr_spearman = gen_in_out.corr(method="spearman")
plt.clf()
hmap = sns.heatmap(corr_spearman,xticklabels=True, yticklabels=True,cmap = cmap,vmin = -1,vmax = 1)
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(22,22)
plt.tight_layout()
plt.yticks(rotation=0)
plt.savefig("correlation_figures/fig_correlation_matrix_{}.png".format("spearman"))
plt.savefig("correlation_figures/fig_correlation_matrix_{}.pdf".format("spearman"))

#Just the outputs
plt.clf()
hmap = sns.heatmap(corr_spearman.iloc[8:10,:],xticklabels=True, yticklabels=True,cmap = cmap,vmin = -1,vmax = 1,annot = True,annot_kws={"size": 32},
fmt = ".2f")
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(24, 8)
plt.tight_layout()
plt.yticks(rotation=0)
plt.savefig("correlation_figures/fig_correlation_matrix_{}_outputs.png".format("spearman"))
plt.savefig("correlation_figures/fig_correlation_matrix_{}_outputs.pdf".format("spearman"))

#PCC
df_pcc = pd.DataFrame(index = labels,columns = labels)
for i in range(len(labels)):
    for j in range(len(labels)):
        print(i," ",j)
        if i == j:
            df_pcc.iloc[i,j] = 1.0
        else:
            pair_corr = pg.pairwise_corr(gen_in_out,columns = [labels[i],labels[j]],covar = labels, method='pearson').round(3)
            df_pcc.iloc[i,j] = pair_corr['r'][0]
            df_pcc.iloc[j,i] = pair_corr['r'][0]

plt.clf()
hmap = sns.heatmap(df_pcc.astype('float64'),xticklabels=True, yticklabels=True,cmap = cmap,vmin = -1,vmax = 1)
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(22, 22)
plt.tight_layout()
plt.yticks(rotation=0)
plt.savefig("correlation_figures/fig_partial_correlation_matrix_{}.png".format("pcc"))
plt.savefig("correlation_figures/fig_partial_correlation_matrix_{}.pdf".format("pcc"))

plt.clf()
hmap = sns.heatmap(df_pcc.astype('float64').iloc[8:10,:],xticklabels=True, yticklabels=True,cmap = cmap,vmin = -1,vmax = 1,annot = True,annot_kws={"size": 32},
fmt = ".2f")
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(24, 8)
plt.tight_layout()
plt.yticks(rotation=0)
plt.savefig("correlation_figures/fig_partial_correlation_matrix_{}_outputs.png".format("pcc"))
plt.savefig("correlation_figures/fig_partial_correlation_matrix_{}_outputs.pdf".format("pcc"))


#PRCC
df_prcc = pd.DataFrame(index = labels,columns = labels)
for i in range(len(labels)):
    for j in range(len(labels)):
        print(i," ",j)
        if i == j:
            df_prcc.iloc[i,j] = 1.0
        else:
            pair_corr = pg.pairwise_corr(gen_in_out,columns = [labels[i],labels[j]],covar = labels, method='spearman').round(3)
            df_prcc.iloc[i,j] = pair_corr['r'][0]
            df_prcc.iloc[j,i] = pair_corr['r'][0]


np.savetxt('prcc.csv', df_prcc.round(2), delimiter='&',fmt='%.2g')
plt.clf()
hmap = sns.heatmap(df_prcc.astype('float64'),xticklabels=True, yticklabels=True,cmap = cmap,vmin = -1,vmax = 1)
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(22, 22)
plt.tight_layout()
plt.yticks(rotation=0)
plt.savefig("correlation_figures/fig_partial_correlation_matrix_{}.png".format("prcc"))
plt.savefig("correlation_figures/fig_partial_correlation_matrix_{}.pdf".format("prcc"))

plt.clf()
hmap = sns.heatmap(df_prcc.astype('float64').iloc[8:10,:],xticklabels=True, yticklabels=True,cmap = cmap,vmin = -1,vmax = 1,annot = True,annot_kws={"size": 32},
fmt = ".2f")
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(24, 8)
plt.tight_layout()
plt.yticks(rotation=0)
plt.savefig("correlation_figures/fig_partial_correlation_matrix_{}_outputs.png".format("prcc"))
plt.savefig("correlation_figures/fig_partial_correlation_matrix_{}_outputs.pdf".format("prcc"))
