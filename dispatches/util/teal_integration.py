#################################################################################
# DISPATCHES was produced under the DOE Design Integration and Synthesis Platform
# to Advance Tightly Coupled Hybrid Energy Systems program (DISPATCHES), and is
# copyright (c) 2020-2023 by the software owners: The Regents of the University
# of California, through Lawrence Berkeley National Laboratory, National
# Technology & Engineering Solutions of Sandia, LLC, Alliance for Sustainable
# Energy, LLC, Battelle Energy Alliance, LLC, University of Notre Dame du Lac, et
# al. All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. Both files are also available online at the URL:
# "https://github.com/gmlc-dispatches/dispatches".
#################################################################################

__author__ = "Gabriel J. Soto"

# This file contains utility functions for constructing and using TEAL cashflows and metrics.
# First, we add TEAL to the current Python path. Note that DISPATCHES, TEAL, and RAVEN are all
#   assumed to be subdirectories within the same directory.
import numpy as np
import operator
from TEAL.src import CashFlows
from TEAL.src import main as RunCashFlow
from TEAL.src.Amortization import MACRS

###################
def checkAmortization(compLife, amortYears=None):
  """
    Check proposed amortization schedule against intended component life.
    If amortization schedule not provided, calculates an appropriate
    one that is less than component life.

    @ In, compLife, CashFlow, component lifetime
    @ In, amortYears, float or int, intended amortization years (defaults to None)
    @ Out, amortYears, float or int, corrected amortization years
  """
  MACRS_yrs = np.array(list(MACRS.keys())) # available amortization years
  amortIsCorrect = bool(amortYears is not None and compLife > amortYears) # check if recalc is needed

  # amortization years longer than intended project life, must recalculate
  if not amortIsCorrect:
    assert isinstance(amortYears, (float, int))
    amortYears = MACRS_yrs[compLife > MACRS_yrs].max() # largest value less than project life
    print("Proposed amortization schedule cannot be longer than intended project life.")
    print(f"Returning a shortened schedule: {amortYears} yrs")

  return amortYears

def build_econ_settings(cfs, life=5, dr=0.1, tax=0.21, infl=0.02184, metrics=None):
  """
    Constructs global settings for economic run.
    Repurposed from TEAL/tests/PyomoTest.py

    @ In, cfs, CashFlow, cash flow components
    @ In, life, float, life time of the years to evaluate
    @ In, dr, float, discount rate
    @ In, tax, float, the amount of tax ratio to apply
    @ In, infl, float, the amount of inflation ratio to apply
    @ In, metrics, list, economic metrics to calculate with cashflows
    @ Out, settings, CashFlow.GlobalSettings, settings
  """
  available_metrics = ['NPV',] # TODO: add IRR, PI

  # check against possible economic metrics supported by TEAL that provide Pyomo expressions
  if metrics is not None:
    residual = set(metrics) - set(available_metrics) # extra metrics that are not supported
    if len(residual) > 0:
      raise Exception(f"Requested metrics not in supported list: {available_metrics}")
  else:
    metrics = available_metrics

  active = []
  for comp_name, cf_list in cfs.items():
    for cf in cf_list:
      active.append(f'{comp_name}|{cf}')
  assert 'NPV' in metrics
  params = {'DiscountRate': dr,
            'tax': tax,
            'inflation': infl,
            'ProjectTime': life,
            'Indicator': {'name': metrics, # TODO: check IRR, PI
                          'active': active},
            'Output' : True,
           }
  settings = CashFlows.GlobalSettings()
  settings.setParams(params)
  setattr(settings, "_verbosity", 0)
  return settings

def build_TEAL_Component(name, comp, mdl, scenario=None, scenario_ind=None):
  """
    Constructs TEAL component which holds all desired cash flows (capex, etc.)
    for a given Plant component (PEM, H2 Tank, etc.)

    @ In, name, str, name of Plant Component
    @ In, comp, dict, dictionary of Plant component cash flows
    @ In, mdl, Pyomo model, multiperiod Pyomo model
    @ In, scenario, Pyomo model, submodel of mdl pertaining to a scenario (defaults to None)
    @ In, scenario_ind, int, index number of scenario (defaults to None)
    @ Out, tealComp, CashFlow.Component, object with all cash flows for given plant component
  """
  # if scenario is None, we are doing LMP Deterministic (only 1 scenario)
  # otherwise, mdl is the main Pyomo model and scenario is the submodel for the scenario within mdl
  scenario = mdl if scenario is None else scenario
  life = comp['Lifetime']
  tealComp = CashFlows.Component()
  tealComp.setParams({'name': name,
                      'Life_time': life})
  cashFlows = []
  for cfName, cfDict in comp.items():
    if cfName == 'Capex':
      alpha, driver = getCapexVarFromModel(cfDict, scenario) # get Pyomo expressions
      capex = createCapex(alpha, driver, life) # create actual TEAL cash flow
      cashFlows.append(capex)

      if 'Amortization' in cfDict.keys():
        # check desired time < comp lifetime
        amort = checkAmortization( life, cfDict['Amortization'] )
        capex.setAmortization('MACRS', amort) # calculate schedule
        amorts = getattr(tealComp, '_createDepreciation')(capex) # create actual TEAL cash flow
        cashFlows.extend(amorts)

    elif cfName == 'FixedOM':
      alpha, driver = getCapexVarFromModel(cfDict, scenario)
      fixedOM = createRecurringYearly(alpha, driver, mdl.project_years)
      cashFlows.append(fixedOM)

    elif cfName == 'Hourly':
      alpha, driver = getDispatchVarFromModel(cfDict, mdl, scenario, scenario_ind)
      hourly = createRecurringHourly(alpha, driver, mdl.plant_life)
      cashFlows.append(hourly)

  tealComp.addCashflows(cashFlows)
  return tealComp

def calculate_TEAL_metrics(tealSettings, tealComponentList):
  """
    Calculates desired TEAL metrics (NPV, PI, etc.) given a list of
    plant components and associated cash flows.

    @ In, tealSettings, CashFlow.GlobalSetting, global economic settings
    @ In, tealComponentList, list, list of component cashflows
    @ Out, metrics, dict, dictionary of Pyomo expressions for requested economic metrics
  """
  metrics = RunCashFlow.run(tealSettings, tealComponentList, {}, pyomoVar=True)

  return metrics

####################################################################
# Methods to get Pyomo Expressions and convert to Cash Flow drivers
####################################################################

def getCapexVarFromModel(cfDict, scenario):
  """
    Get Capex parameters from dictionary and convert into a Pyomo expression.
    More specifically, a string is extracted from the dictionary which specifies the
    IDAES Pyomo expression meant to represent the Capex cashflow driver.

    @ In, cfDict, dict, cash flow dictionary
    @ In, scenario, Pyomo model, submodel of mdl pertaining to a scenario (defaults to None)
    @ Out, alpha, float, conversion of driver units to monetary value
    @ Out, driver, Pyomo Expression, cash flow driver
  """
  alpha = cfDict['Value']
  mults = cfDict['Multiplier']
  exprs = cfDict['Expressions']
  assert( len(mults)==len(exprs) )  # NOTE: Multiplier same length as Driver

  # extraction of attribute looks like: model.expression[i]
  pyomoExpr = [operator.attrgetter(exprs[i])(scenario) for i in range(len(exprs))]
  driver = [m*pexp for m, pexp in zip(mults, pyomoExpr)]
  driver = driver[0]
  return alpha, driver

def getDispatchVarFromModel(cfDict, mdl, scenario, scenario_ind=None):
  """
    Get Dispatch parameters from dictionary and convert into a Pyomo expression.
    More specifically, a string is extracted from the dictionary which specifies the
    IDAES Pyomo expression meant to represent the Hourly Dispatch cash flow driver.

    @ In, cfDict, dict, cash flow dictionary
    @ In, mdl, Pyomo model, multiperiod Pyomo model
    @ In, scenario, Pyomo model, submodel of mdl pertaining to a scenario (defaults to None)
    @ In, scenario_ind, int, index number of scenario (defaults to None)
    @ Out, alpha, float, conversion of driver units to monetary value
    @ Out, dispatch_array, numpy array, array of hourly cashflows (Pyomo expressions)
  """
  alpha = cfDict['Value']
  mults = cfDict['Multiplier']
  exprs = cfDict['Expressions']
  assert( len(mults)==len(exprs) )  # NOTE: Multiplier same length as Driver

  # time indeces for HERON/TEAL
  n_hours = len(mdl.set_time)
  n_days  = len(mdl.set_days)
  n_years = len(mdl.set_years)
  n_hours_per_year = n_hours * n_days # sometimes number of days refers to clusters < 365

  n_projLife = mdl.plant_life + 1
  fullYearsArray = np.hstack([0, mdl.project_years]) # looks like [0, 2022, 2022, 2022, 2022, ... 2032, ... ]

  # template array for holding dispatch Pyomo expressions/objects
  dispatch_array = np.zeros((n_projLife, n_hours_per_year), dtype=object)

  # time indeces for DISPATCHES, as array of tuples
  indeces    = np.array([tuple(i) for i in mdl.set_period], dtype="i,i,i")
  time_shape = (n_years, n_hours_per_year) # reshaping the tuples array to match HERON dispatch
  indeces    = indeces.reshape(time_shape)

  is_stochastic = getattr(mdl, '_stochastic_model')
  weights_days = mdl.weights_days[scenario_ind] if is_stochastic else mdl.weights_days

  # currently, taking this to mean that we are using the LMP signal...
  # TODO: needs to be more general here
  if alpha == []:
    # # one extra year in first axis to account for construction year (no production)
    alpha = np.zeros([n_projLife, n_hours_per_year])
    # it necessary to have alpha be [year, clusterhour] instead of [year, cluster, hour]
    #    clusterhour loops through hours first, then cluster
    signal = mdl.LMP[scenario_ind] if is_stochastic else mdl.LMP

    realized_alpha = [[signal[y][d][h] \
                          for d in mdl.set_days
                            for h in mdl.set_time] # order here matches *indeces*
                              for y in fullYearsArray[1:]] #shape here is [year, hour]
    # # first column of year axis is 0 for project year 0
    realized_alpha = np.array(realized_alpha)
    alpha[1:,:] = realized_alpha

  pcount = -1
  for p, pyear in enumerate(fullYearsArray):
    if pyear == 0:
      continue

    if pyear > fullYearsArray[p-1]:
      pcount +=1

    for time in range(n_hours_per_year):
      ind = tuple(indeces[pcount,time])
      # looping through all DISPATCHES variables pertaining to this specific dispatch
      #   e.g., turbine costs due to work done by turbine + compressor, separate variables
      dispatch_driver = 0
      for ds, dStr in enumerate(exprs):
        dispatch_driver += operator.attrgetter(dStr)(scenario.period[ind])[0] * mults[ds]

      # getting weights for each day/cluster
      dy, yr = ind[1:]
      weight = weights_days[yr][dy]  # extracting weight for year + day

      # storing individual Pyomo dispatch
      dispatch_array[p, time] = dispatch_driver * weight

  return alpha, dispatch_array

###################################
# Methods to create TEAL components
###################################

def createCapex(alpha, driver, projLife):
  """
    Constructs the TEAL Capex Cashflow
    @ In, alpha, float, price
    @ In, driver, Pyomo Expression, quantity used in cashflow
    @ In, projLife, float, component life
    @ Out, cf, TEAL.src.CashFlows.Component, cashflow sale for each capital expenditures
  """
  # extract alpha, driver as just one value
  cf = CashFlows.Capex()
  cf.name = 'Cap'
  # life = comp._lifetime
  cf.initParams(projLife)
  cfParams = {'name': 'Cap',
               'alpha': alpha,
               'driver': driver,
               'reference': 1.0,
               'X': 1.0,
               'depreciate': 1,
               'mult_target': None,
               'inflation': False,
              }
  cf.setParams(cfParams)
  return cf

def createRecurringYearly(alpha, driver, lifeVector):
  """
    Constructs a TEAL Yearly Cashflow
    @ In, alpha, float, yearly price/cost to populate
    @ In, driver, Pyomo Expression, quantity used in cashflow
    @ In, lifeVector, numpy array, years in project life
    @ Out, cf, TEAL.src.CashFlows.Component, cashflow sale for the recurring yearly
  """
  lifeVector = np.hstack([0, lifeVector])
  cf = CashFlows.Recurring()
  cfParams = {'name': 'FixedOM',
               'X': 1,
               'mult_target': None,
               'inflation': False}
  cf.setParams(cfParams)

  # convert to binary mask, includes a zero for year 0 (construction)
  projYears = np.array([y>1 for y in lifeVector], dtype=int)
  projYears = projYears.astype(object)

  # 0 for first year (build year) -> TODO couldn't this be automatic?
  alphas  = projYears * alpha
  drivers = projYears * driver

  # construct annual summary cashflows
  cf.computeYearlyCashflow(alphas, drivers)
  return cf

def createRecurringHourly(alpha, driver, projLife):
  """
    Constructs a TEAL Hourly Cashflow
    @ In, alpha, float, price
    @ In, driver, Pyomo Expression, quantity used in cashflow
    @ In, projLife, float, component life
    @ Out, cf, TEAL.src.CashFlows.Component, cashflow sale for each capital expenditures
  """
  projLife += 1
  cf = CashFlows.Recurring()
  cfParams = {'name': 'Hourly',
               'X': 1,
               'mult_target': None,
               'inflation': False}
  cf.setParams(cfParams)
  cf.initParams(projLife, pyomoVar=True)
  for year in range(projLife):
    if isinstance(alpha, float):
      cf.computeIntrayearCashflow(year, alpha, driver[year, :])
    else:
      cf.computeIntrayearCashflow(year, alpha[year, :], driver[year, :])
  return cf
