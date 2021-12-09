#This file writes out the true input combinations as opposed to splitting out the startup profiles
from pyomo.environ import value
from functools import reduce
import math
import operator
import pandas as pd

index = 0
#values are: [lag_time_per_minimum_down_time,cost_per_capacity], units are: [hr/min_dn (hr),$/MW capacity]
#NOTE: I think lag ratios less than 1 get removed from the simulation
startup_cost_data = { 'yellow' : [ (0.75, 94.00023429), (2.5, 135.2230393), (3, 147.0001888) ],
                      'blue'   : [ (0.375, 93.99890632), (1.375, 101.4374234), (7.5, 146.9986814) ],
                      'brown'  : [ (0.166666667, 58.99964891), (0.25, 61.09068702), (2, 104.9994673) ],
                      'dark_blue': [ (0.111111111, 35.00249986), (0.222222222, 49.66991167), (0.444444444, 79.00473527) ],
                     }

p_max_base = 355
p_maxs = [ 0.5*p_max_base, 0.75*p_max_base, 1.0*p_max_base, 1.25*p_max_base ]

p_min_multis = [ 0., 0.15, 0.30, 0.45 ]

ramp_multis = [ 0.5, 0.75, 1. ]

min_ups = [ 1, 2, 4, 8, 16 ]

min_dn_multis = [ 0.5, 1., 2. ]

startup_cost_profiles = [ startup_cost_data['yellow'],
                          startup_cost_data['blue'],
                          startup_cost_data['brown'],
                          startup_cost_data['dark_blue'],
                          [ (1.0, 0.) ], #lag is just min_dn, no startup cost
                        ]

startup_cost_profiles = [4,3,2,1,0]
representative_startup_cost = [0,
                                startup_cost_data['dark_blue'][1][1],
                                startup_cost_data['brown'][1][1],
                                startup_cost_data['blue'][1][1],
                                startup_cost_data['yellow'][1][1]
                                ]


marginal_costs = [ 5., 10., 15., 20., 25., 30. ] #$/MWh

no_load_costs = [ 0., 1., 2.5 ] # $/(MWh) at max capacity

parameters = ( ('p_max', p_maxs),
               ('p_min_multi', p_min_multis),
               ('ramp_multi', ramp_multis),
               ('min_up', min_ups),
               ('min_dn_multi', min_dn_multis),
               ('marginal_cost', marginal_costs),
               ('no_load_cost', no_load_costs),
               ('startup_cost_profile', startup_cost_profiles),
             )

total_combinations = reduce(operator.mul, (len(l) for _,l in parameters), 1)

def get_vals(idx):
    assert 0 <= idx < total_combinations
    param = dict()
    for n,l in parameters:
        count = len(l)
        param[n] = l[idx % count]
        idx //= count
    return param

combinations = {parameters[i][0]:[] for i in range(len(parameters))}
for idx in range(total_combinations):
    print(idx)
    param = get_vals(idx)
    for k in param.keys():
        combinations[k].append(param[k])

df = pd.DataFrame(combinations)
df['repr_startup_cost'] = pd.Series()
for i in startup_cost_profiles:
    df['repr_startup_cost'][df.index[df['startup_cost_profile'] == i]] = representative_startup_cost[i]

df.to_csv("simulation_sweep_summary/prescient_input_combinations.csv")
