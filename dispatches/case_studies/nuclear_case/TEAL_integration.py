#################################################################################
# DISPATCHES was produced under the DOE Design Integration and Synthesis
# Platform to Advance Tightly Coupled Hybrid Energy Systems program (DISPATCHES),
# and is copyright (c) 2022 by the software owners: The Regents of the University
# of California, through Lawrence Berkeley National Laboratory, National
# Technology & Engineering Solutions of Sandia, LLC, Alliance for Sustainable
# Energy, LLC, Battelle Energy Alliance, LLC, University of Notre Dame du Lac, et
# al. All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. Both files are also available online at the URL:
# "https://github.com/gmlc-dispatches/dispatches".
#
#################################################################################
__author__ = "Gabriel J. Soto"

# This file contains utility functions for constructing and using TEAL cashflows and metrics.
# First, we add TEAL to the current Python path. Note that DISPATCHES, TEAL, and RAVEN are all
#   assumed to be subdirectories within the same directory.
import numpy as np
import operator
import os
import sys
from os import path
cwd = os.getcwd()
proj_dir = path.dirname( path.abspath( path.join(cwd, '../../..') ) )
TEAL_dir = path.abspath( path.join(proj_dir, 'TEAL') )
raven_dir = path.abspath( path.join(proj_dir, 'raven') )
sys.path.append( proj_dir )
sys.path.append( TEAL_dir )
sys.path.append( raven_dir )
sys.path.append( path.abspath( path.join(TEAL_dir, 'src') ) )

from TEAL.src import CashFlows
from TEAL.src import main as RunCashFlow
from TEAL.src.Amortization import MACRS

###################
def checkAmortization(projLife, amortYears=None):
  """
    Check proposed amortization schedule against intended project life.
    If amortization schedule not provided, calculates an appropriate
    one that is less than project life.

    @ In, projLife, CashFlow, project lifetime
    @ In, amortYears, float or int, intended amortization years (defaults to None)
    @ Out, amortYears, float or int, corrected amortization years
  """
  MACRS_yrs = np.array(list(MACRS.keys())) # available amortization years
  amortIsCorrect = bool(amortYears is not None and projLife > amortYears) # check if recalc is needed

  # amortization years longer than intended project life, must recalculate
  if not amortIsCorrect:
    assert isinstance(amortYears, (float, int))
    amortYears = MACRS_yrs[projLife > MACRS_yrs].max() # largest value less than project life
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
                          'active': active}
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
  life = np.min([mdl.plant_life, comp['Lifetime']])
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
        # check desired time < proj years
        amort = checkAmortization( life, cfDict['Amortization'] )
        capex.setAmortization('MACRS', amort) # calculate schedule
        amorts = getattr(tealComp, '_createDepreciation')(capex) # create actual TEAL cash flow
        cashFlows.extend(amorts)

    elif cfName == 'FixedOM':
      alpha, driver = getCapexVarFromModel(cfDict, scenario)
      fixedOM = createRecurringYearly(alpha, driver, mdl.set_years)
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
  n_projLife = mdl.plant_life + 1

  yearsMapArray = np.hstack([0, mdl.set_years]) # looks like [0, 2022, 2022, 2022, 2022, ... 2032, ... ]

  n_hours_per_year = n_hours * n_days # sometimes number of days refers to clusters < 365

  dispatch_array = np.zeros((n_projLife, n_hours_per_year), dtype=object)

  indeces = np.array([tuple(i) for i in scenario.period_index], dtype="i,i,i")
  time_shape = (n_years, n_hours_per_year) # reshaping the tuples array to match HERON dispatch
  indeces = indeces.reshape(time_shape)

  if mdl.stochastic:
    weights_days = mdl.weights_days[scenario_ind]
  else:
    weights_days = mdl.weights_days

  # currently, taking this to mean that we are using the LMP signal...
  # TODO: needs to be more general here
  if alpha == []:
    signal = mdl.LMP

    # # plus 1 to year term to allow for 0 recurring costs during build year
    alpha = np.zeros([n_projLife, n_hours_per_year])
    # it necessary to have alpha be [year, clusterhour] instead of [year, cluster, hour]
    #    clusterhour loops through hours first, then cluster
    if mdl.stochastic:
      signal = signal[scenario_ind]


    realized_alpha = [[signal[y][d][h] \
                          for d in mdl.set_days
                            for h in mdl.set_time] # order here matches *indeces*
                              for y in yearsMapArray[1:]] #shape here is [year, hour]
    # # first column of year axis is 0 for project year 0
    realized_alpha = np.array(realized_alpha)
    alpha[1:,:] = realized_alpha

  # TODO: check that all periods and LMPs match up...
  pcount = -1
  for p, pyear in enumerate(yearsMapArray):
    if pyear == 0:
      continue

    if pyear > yearsMapArray[p-1]:
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

###################################
# Helper methods
###################################

def restructure_LMP(m):
  """
    Restructures LMP signal from JSON to be more compatible with TEAL (might be deprecated)
    @ In, m, Pyomo model, multiperiod Pyomo model
    @ Out, None, None
  """
  # list of available years in LMP data
  if m.stochastic:
    years = list(m.LMP[0].keys())
  else:
    years = list(m.LMP.keys())

  n_years_data = len(years)
  set_scenarios = list( m.LMP.keys() ) if m.stochastic else [0]

  # template dictionary full of 0s, same structure as LMP
  zeroDict = {cluster: {hour: 0
                      for hour in m.set_time}
                for cluster in m.set_days}

  ## CHECKING THAT WE HAVE ENOUGH DATA FOR SIM ##

  # Case where we have less data than the project life/sim time
  #    Here, we assume (as in the default nuclear case demo) that
  #    the years 2022-2031 all have the same LMP data, which
  #    helps to cut down on variables just for the demonstration
  if n_years_data < m.plant_life:
    print("Requested LMP Data less than project life")
    projLifeRange = np.arange( years[0]-1,   # year-1 is the construction year
                      years[0] + m.plant_life) # full project time with first year of data as starting point

    # initializing empty dicts and lists
    newLMP      = {} # going to replace existing LMP dictionary

    for s in set_scenarios:
      newYearsVec = [] # list of years used
      stuckYear   = 0  # ugly way of duplicating years
      # looping through possible years in lifetime (e.g., 2021 -> 2041)
      for i,y in enumerate(projLifeRange):
        # data not available for given year within project lifetime
        if y not in years:
          if i == 0: # construction year
            newLMP[y] = zeroDict
            newYearsVec.append(0)
          else: # duplicate previous year's values
            newLMP[y] = newLMP[y-1]
            newYearsVec.append(stuckYear)
        # data for current year is available in LMP dict
        else:
          stuckYear = y # update year for duplication (word?)
          newLMP[y] = m.LMP[y] if not m.stochastic else m.LMP[s][y] # keep current LMP value
          newYearsVec.append(y) # update current year

      # save to model object
      if m.stochastic:
        m.LMP[s] = newLMP
      else:
        m.LMP = newLMP
      m.yearsFullVec = newYearsVec

  elif n_years_data > m.plant_life:
    print("LMP Data more than project life, must curtail.")
    # TODO fill this out
  else:
    print("LMP Data matches project life")
    # TODO fill this out
    # years.insert(0,0)
