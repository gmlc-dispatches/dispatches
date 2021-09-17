from pyomo.environ import value
from functools import reduce
import math
import operator

index = 0


#values are: [lag,cost], units are: [hr/min_dn (hr),$/MW capacity]
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

#print("Number of combinations:", total_combinations) 

def get_vals(idx):
    assert 0 <= idx < total_combinations
    param = dict()
    for n,l in parameters:
        count = len(l)
        param[n] = l[idx % count]
        idx //= count
    return param

param = get_vals(index)

## THE CONSTANTS FOR THIS RUN
gen = '123_STEAM_3'

## BEGIN get data
pmax = param['p_max']
p_min_coeff = param['p_min_multi']

ramp_coeff = param['ramp_multi']

min_up = param['min_up']
min_dn_coeff = param['min_dn_multi']

startup_cost_profile = param['startup_cost_profile']

marginal_cost = param['marginal_cost']
fixed_run_cost = param['no_load_cost']
## END get data

def change_gen_123_STEAM_3(data, market):

    pmin = p_min_coeff*pmax
    hr_ramp_rate = ramp_coeff*(pmax-pmin)

    startup_shutdown_rate = min( pmin+0.5*hr_ramp_rate, pmax)

    min_dn = int(math.ceil(min_dn_coeff*min_up))

    data_none = data[None]
    ## change the p_max
    data_none['MaximumPowerOutput'][gen] = pmax 

    ## change the p_min
    data_none['MinimumPowerOutput'][gen] = pmin

    ## fix the initial state, if needed (for the initial inital conditions)
    power_gen_t0 = value(data_none['PowerGeneratedT0'][gen])
    unit_on_t0 = value(data_none['UnitOnT0'][gen])

    if unit_on_t0:
        if power_gen_t0 > pmax:
            data_none['PowerGeneratedT0'][gen] = pmax
        if power_gen_t0 < pmin:
            data_none['PowerGeneratedT0'][gen] = pmin

    ## change the ramp rate
    data_none['NominalRampUpLimit'][gen] = hr_ramp_rate
    data_none['NominalRampDownLimit'][gen] = hr_ramp_rate

    ## change the startup/shutdown ramp rate
    data_none['StartupRampLimit'][gen] = startup_shutdown_rate
    data_none['ShutdownRampLimit'][gen] = startup_shutdown_rate
    
    ## change the cost
    #is fixed run cost actually $/MWh at max capacity?
    data_none['CostPiecewisePoints'][gen] = [pmin, pmax]
    data_none['CostPiecewiseValues'][gen] = [pmax*fixed_run_cost + pmin*marginal_cost, 
                                            pmax*fixed_run_cost + pmax*marginal_cost]

    ## change the uptime/downtime
    data_none['MinimumUpTime'][gen] = min_up
    data_none['MinimumDownTime'][gen] = min_dn

    #I think this is dollars?
    #[lag,cost] => [time/minimum_dwnn_time,$/MW]
    raw_startup_costs = [ [min_dn*time_over_min_dn, pmax*cost_over_capacity] \
                        for time_over_min_dn, cost_over_capacity in startup_cost_profile ]

    
    #I think this is trying to keep the shortest lag time as min_dn
    for idx, (startup_time, _) in enumerate(raw_startup_costs):
        print(startup_time)
        if startup_time > min_dn:
            ## go back to the index before this was the case
            idx -= 1
            break
    startup_costs = raw_startup_costs[idx:]

    #startup lag from hot state will always be minimum downtime
    startup_costs[0][0] = min_dn

    startup_costs_rounded = [ (int(math.ceil(lag)), cost) for lag, cost in startup_costs ]

    startup_costs = []
    ## eliminate duplicate lag times from math.ceil
    for idx, (lag, cost) in enumerate(startup_costs_rounded[:-1]):
        next_lag = startup_costs_rounded[idx+1][0]
        if lag == next_lag:
            continue
        else:
            startup_costs.append((lag,cost))
    ## put on the last cost
    startup_costs.append(startup_costs_rounded[-1])

    data_none['StartupLags'][gen] = [ lag for lag, _ in startup_costs ]
    data_none['StartupCosts'][gen] = [ cost for _, cost in startup_costs ]

data_dict_callback = change_gen_123_STEAM_3
