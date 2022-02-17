import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('font', size=32)
plt.rc('axes', titlesize=32)     # fontsize of the axes title
import pickle
import numpy as np
import json

with open('Si_delta_sensitivies.json') as f:
    Si_delta = json.load(f)

with open('Si_sobol_sensitivies.json') as f:
    Si_sobol = json.load(f)

x = ["X{}".format(i) for i in range(1,14)]

sobol1 = Si_sobol["S1"]
sobol1_conf = Si_sobol["S1_conf"]

sobolT = Si_sobol["ST"]


delta_Si = Si_delta["S1"]
delta_conf = Si_delta["S1_conf"]

fig = plt.figure(figsize = (12,12))
ax = fig.add_subplot(111)
h1 = ax.errorbar(x, sobol1, yerr=sobol1_conf, fmt='o',markersize = 10,capsize = 10)
h2 = ax.scatter(x,sobolT,s = 100,color = "purple")
# ax.errorbar(x, sobolT, yerr=sobolT_conf, fmt='o',markersize = 10,capsize = 10)
h3 = ax.errorbar(x, delta_Si, yerr=delta_conf, fmt='o',markersize = 10,capsize = 10)
ax.set_ylabel("Sensitivity Index [Revenue]")
ax.set_xlabel("Input Parameter")
ax.legend([h1,h2,h3],["Sobol S1","Sobol ST","Delta S1"])
plt.xticks(rotation = 45)
plt.tight_layout()

fig.savefig("sensitivies_revenue.png")
fig.savefig("sensitivies_revenue.pdf")
