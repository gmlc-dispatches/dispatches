from idaes.surrogate import alamopy
import numpy as np
import pandas as pd
import os
import sys
import copy
import pickle

sys.path.insert(1, os.path.join(sys.path[1], '..'))
import alamopy_writer

#GET PRESCIENT DATA
f_perturbed_inputs = os.path.join(os.getcwd(),"../simulation_sweep_summary/prescient_perturbed_gen_inputs.h5")
f_perturbed_outputs = os.path.join(os.getcwd(),"../simulation_sweep_summary/prescient_perturbed_gen_outputs.h5")

df_perturbed_inputs = pd.read_hdf(f_perturbed_inputs)
df_perturbed_outputs = pd.read_hdf(f_perturbed_outputs)

f_perturbed_inputs_raw = os.path.join(os.getcwd(),"../simulation_sweep_summary/prescient_input_combinations.csv")
df_perturbed_inputs_raw = pd.read_csv(f_perturbed_inputs_raw)


perturbed_revenue = df_perturbed_outputs["Total Revenue [$]"]
perturbed_dispatch = df_perturbed_outputs["Total Dispatch [MW]"]

perturbed_dispatch_array = perturbed_dispatch.to_numpy()

cutoff = 0.0
#cutoff = 0.0
dispatch_inds = np.nonzero(perturbed_dispatch_array >= cutoff*np.max(perturbed_dispatch_array))[0]

#pmax,pmin,ramp_rate,marg_cst
# x = df_perturbed_inputs.iloc[:,1:].to_numpy()[dispatch_inds]
x = df_perturbed_inputs_raw.iloc[:,[1,2,3,4,5,6,7,9]].to_numpy()[dispatch_inds]

#x = df_perturbed_inputs.iloc[:,1:].to_numpy()[dispatch_inds]
z = perturbed_revenue.to_numpy()[dispatch_inds]/1e6

xm = np.mean(x,axis = 0)
xstd = np.std(x,axis = 0)
zm = np.mean(z)
zstd = np.std(z)

x_scaled = (x - xm) / xstd
z_scaled = (z - zm) / zstd

modeler = 1
cases = [[(1,2),(0)],[(1,2),(1)],[(1,2),(1,2)],[(1,2,3),(1,2)],[(1,2,3),(1,2,3)]]
i = 4
for i in range(len(cases)):
    result = alamopy.alamo(x_scaled,z_scaled,
                    zlabels = ['revenue'],
                    xlabels=['pmax','pmin_multi','ramp_multi','min_up_time','min_down_multi',
                                    'marg_cst','no_load_cst','startup_cst'],
                    expandoutput=True,
                    showalm=True,
                    monomialpower=cases[i][0],
                    multi2power=cases[i][1],
                    modeler=modeler)

    alamopy_writer.write_alamo_func(result['model']['revenue'],result['xlabels'],'revenue_scaled_cutoff_{}_all_terms_model_{}'.format(int(cutoff*100),i))
    #remove the lambda function since we can't serialize it with pickle
    result_to_save = copy.copy(result)
    result_to_save.pop('f(model)', None)
    result_to_save['xm'] = xm
    result_to_save['xstd'] = xstd
    result_to_save['zm'] = zm
    result_to_save['zstd'] = zstd

    with open('revenue_cutoff_{}_all_terms_model_{}.pkl'.format(int(cutoff*100),i), "wb") as output_file:
        pickle.dump(result_to_save, output_file, pickle.HIGHEST_PROTOCOL)
