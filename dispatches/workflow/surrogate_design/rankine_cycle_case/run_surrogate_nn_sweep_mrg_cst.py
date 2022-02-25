"""
	This script runs the trained surrogates ove a range of marginal costs. The main objective is to
	explore how revenue and capacity change with marginal cost using the surrogates.
"""
import pyomo.environ as pyo
from pyomo.environ import *
import json
import pickle
import numpy as np

surrogate_dir = os.path.join(this_file_dir(),"../../train_market_surrogates/steady_state/surrogate_models/scikit/models")

# load scaling and bounds for each surrogate
with open(os.path.join(surrogate_dir,"training_parameters_revenue.json"), 'rb') as f:
    rev_data = json.load(f)

with open(os.path.join(surrogate_dir,"training_parameters_zones.json"), 'rb') as f:
    zone_data = json.load(f)

with open(os.path.join(surrogate_dir,"training_parameters_nstartups.json"), 'rb') as f:
    nstartups_data = json.load(f)

# load scikit neural networks
with open(os.path.join(surrogate_dir,'scikit_revenue.pkl'), 'rb') as f:
    nn_revenue = pickle.load(f)

with open(os.path.join(surrogate_dir,'scikit_zones.pkl'), 'rb') as f:
    nn_zones = pickle.load(f)

with open(os.path.join(surrogate_dir,'scikit_nstartups.pkl'), 'rb') as f:
    nn_startups = pickle.load(f)

#representative startup costs
startup_csts = [0., 49.66991167, 61.09068702, 101.4374234,  135.2230393]
start_cst_index=4

pmax = 177.5
pmin_multi = 0.3
ramp_multi = 1.0
min_up_time = 4.0
min_dn_multi = 1.0
no_load_cst = 1.0
startup_cst = startup_csts[start_cst_index]

revenues = []
zone_hours_all = []
capacity_factors = []

#each zone corresponds to an average power output
zone_outputs = np.array([0.0,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,1.0])
pmin = pmax*pmin_multi

#these are actual power outputs
raw_zone_outputs = zone_outputs*(pmax-pmin) + pmin
mrg_csts = np.linespace(5,30,100)

for mrg_cst in mrg_csts:
	inputs = np.array([pmax,pmin_multi,ramp_multi,min_up_time,min_dn_multi,mrg_cst,no_load_cst,startup_cst])
	scaled_inputs = (inputs - rev_data['xm_inputs'])/rev_data['xstd_inputs']

	scaled_revenue = nn_revenue.predict([scaled_inputs])
	revenue = (scaled_revenue*rev_data['zstd_revenue'] + rev_data['zm_revenue'])[0]

	scaled_zone_hours = nn_zones.predict([scaled_inputs])[0]
	zone_hours = scaled_zone_hours*np.array(zone_data['zstd_zones']) + np.array(zone_data['zm_zones'])

	#the surrogate can predict negative hours but we use a smoothmax in the design problem to knock these to zero
	zone_hours[zone_hours < 0] = 0
	zone_hours_on = zone_hours[1:]

	#capactiy factor
	cap_factor = (np.dot(zone_hours_on,raw_zone_outputs) / (pmax*8736))

	revenues.append(revenue)
	zone_hours_all.append(zone_hours)
	capacity_factors.append(cap_factor)


data = {'mrg_csts':mrg_csts,'revenue':list(revenues),'capacity_factor':list(capacity_factors)}

with open('rankine_results/scikit_surrogate/mrg_cst_sweep.json', 'w') as outfile:
    json.dump(data, outfile)
