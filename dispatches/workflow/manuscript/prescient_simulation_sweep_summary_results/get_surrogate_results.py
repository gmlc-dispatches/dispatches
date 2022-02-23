import json
import pickle
import numpy as np
from pyomo.common.fileutils import this_file_dir
import sys 
import os
sys.path.append(os.path.join(this_file_dir(),"../../"))
from read_scikit_to_omlt import load_scikit_mlp

# load scaling and bounds for each surrogate
with open(os.path.join(this_file_dir(),"../surrogate_models/scikit/models/training_parameters_revenue.json"), 'rb') as f:
    rev_data = json.load(f)

with open(os.path.join(this_file_dir(),"../surrogate_models/scikit/models/training_parameters_zones.json"), 'rb') as f:
    zone_data = json.load(f)

# with open("surrogate_models/scikit/models/training_parameters_nstartups.json", 'rb') as f:
#     nstartups_data = json.load(f)

# load surrogates
with open(os.path.join(this_file_dir(),'../surrogate_models/scikit/models/scikit_revenue.pkl'), 'rb') as f:
    nn_revenue = pickle.load(f)

with open(os.path.join(this_file_dir(),'../surrogate_models/scikit/models/scikit_zones.pkl'), 'rb') as f:
    nn_zones = pickle.load(f)

# with open('surrogate_models/scikit/models/scikit_nstartups.pkl', 'rb') as f:
#     nn_nstartups = pickle.load(f)


zone_outputs = np.array([0.0,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,1.0])

def predict_with_surrogate(pmax,pmin_multi,ramp_multi,min_up_time,min_dn_multi,no_load_cst,startup_cst,mrg_csts):

	revenues = []
	zone_hours_all = []
	capacity_factors = []

	pmin = pmax*pmin_multi
	raw_zone_outputs = zone_outputs*(pmax-pmin) + pmin

	for mrg_cst in mrg_csts:
		inputs = np.array([pmax,pmin_multi,ramp_multi,min_up_time,min_dn_multi,mrg_cst,no_load_cst,startup_cst])
		scaled_inputs = (inputs - rev_data['xm_inputs'])/rev_data['xstd_inputs']

		scaled_revenue = nn_revenue.predict([scaled_inputs])
		revenue = (scaled_revenue*rev_data['zstd_revenue'] + rev_data['zm_revenue'])[0]

		scaled_zone_hours = nn_zones.predict([scaled_inputs])[0]
		zone_hours = scaled_zone_hours*np.array(zone_data['zstd_zones']) + np.array(zone_data['zm_zones'])

		zone_hours[zone_hours < 0] = 0

		zone_hours_on = zone_hours[1:]

		cap_factor = (np.dot(zone_hours_on,raw_zone_outputs) / (pmax*8736))
		
		revenues.append(revenue)
		zone_hours_all.append(zone_hours)
		capacity_factors.append(cap_factor)


	data = {'mrg_csts':mrg_csts,'revenue':list(revenues),'capacity_factor':list(capacity_factors)}
	return data