#Create 2 dataframes with the Prescient results
#1.) Prescient generator outputs for perturbed generator
#2.) Prescient generator inputs for perturbed generator

#perturbed_gen_summary.h5 contains input and output for the perturbed generator

#input and outputs
f_perturbed = os.path.join(os.getcwd(),"perturbed_gen_summary.h5")
df_perturbed = pd.read_hdf(f_perturbed)

#split into separate input and output dataframes with the same Simulation Index
perturbed_gen_inputs = df_perturbed.iloc[:,0:14]
perturbed_gen_outputs = pd.concat([df_perturbed.iloc[:,0], df_perturbed.iloc[:,14:-1]], axis=1, sort=False)

perturbed_gen_inputs.to_hdf("prescient_perturbed_gen_inputs.h5","data",complevel = 9,index = False)
perturbed_gen_inputs.to_csv("prescient_perturbed_gen_inputs.csv",index = False)

perturbed_gen_outputs.to_hdf("prescient_perturbed_gen_outputs.h5","data",complevel = 9,index = False)
perturbed_gen_outputs.to_csv("prescient_perturbed_gen_outputs.csv",index = False)
