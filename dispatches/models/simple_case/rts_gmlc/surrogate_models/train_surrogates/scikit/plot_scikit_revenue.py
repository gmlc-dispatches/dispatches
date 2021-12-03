# produce plot
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('font', size=32)
plt.rc('axes', titlesize=32)

import pickle


f_perturbed_outputs = os.path.join(os.getcwd(),"../../../prescient_simulation_sweep_summary_results/prescient_perturbed_gen_outputs.h5")
f_perturbed_inputs_raw = os.path.join(os.getcwd(),"../../../prescient_simulation_sweep_summary_results/prescient_input_combinations.csv")

df_perturbed_inputs_raw = pd.read_csv(f_perturbed_inputs_raw)
df_perturbed_outputs = pd.read_hdf(f_perturbed_outputs)

# scale revenue data, x is input, z is output
x = df_perturbed_inputs_raw.iloc[:,[1,2,3,4,5,6,7,9]].to_numpy()
perturbed_revenue = df_perturbed_outputs["Total Revenue [$]"]
z = perturbed_revenue.to_numpy()/1e6

xm = np.mean(x,axis = 0)
xstd = np.std(x,axis = 0)
zm = np.mean(z)
zstd = np.std(z)

x_scaled = (x - xm) / xstd
z_scaled = (z - zm) / zstd

#load up revenue model
with open("models/scikit_revenue.pkl", 'rb') as f:
    model = pickle.load(f)


predicted_revenue = model.predict(x_scaled)
predict_unscaled = predicted_revenue*zstd + zm

# plot results
plt.figure(figsize=(12,12))
plt.scatter(z, predict_unscaled, color = "green", alpha = 0.01)
plt.plot([min(z), max(z)],[min(z), max(z)])
plt.xlabel("True Revenue [MM$]")
plt.ylabel("Predicted Revenue [MM$]")
plt.savefig("figures/scikit_revenue.png")
plt.savefig("figures/scikit_revenue.pdf")
