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
__author__ = "Gabriel J. Soto, Anna Wrobel"

# This file contains utility functions for generating synthetic histories from a trained reduced
#    order model from RAVEN.
# Note that DISPATCHES, TEAL, and RAVEN are all assumed to be subdirectories within
#    the same directory.
import os
import sys
from os import path
cwd = os.getcwd()
proj_dir = path.dirname( path.abspath( path.join(cwd, '../../..') ) )
raven_dir = path.abspath( path.join(proj_dir, 'raven') )
sys.path.append( proj_dir )
sys.path.append( raven_dir )
sys.path.append( path.abspath( path.join(raven_dir, 'scripts')))
sys.path.append( path.abspath( path.join(raven_dir, 'plugins')))

from raven.ravenframework.utils import xmlUtils
from raven.scripts import externalROMloader

import numpy as np
import operator

###################
class SynHist_integration():
  """
    Class to facilitate integration of RAVEN's Synthetic History generation.
    Primary use is to load an already-trained price signal ARMA model. This model is
    built during the constructor of this class.
    Additional methods included for sampling from ARMA model and restructure data to accommodate
    DISPATCHES workflow.
  """

  def __init__(self, target_file):
    """
      Constructor for SynHist_integration.
      @ In, target_file, str, name of synthetic history file
      @ Out, None
    """
    self.target_file = target_file # full path for ARMA file
    self.inp = {'scaling': [1]} # this is a dummy input used for sampling call to ARMA model object
    self.runner = self.buildRunner() # builds object to run ARMA model sampling

  def buildRunner(self):
    """
      Builds synthetic history object from a pickledROM.
      Currently only ARMA models are supported for pickledROM.
    """
    if not path.exists(self.target_file):
      raise Exception(f"Target file not found at {self.target_file}")

    runner = externalROMloader.ravenROMexternal(self.target_file, raven_dir)

    nodes = []
    node = xmlUtils.newNode('ROM', attrib={'name': 'SyntheticHistory', 'subType': 'pickledRom'})
    node.append(xmlUtils.newNode('clusterEvalMode', text='clustered'))
    nodes.append(node)
    runner.setAdditionalParams(nodes)
    return runner

  def generateSyntheticHistory(self, signal_name, set_years):
    """
      Generates a sampled synthetic history from saved ROM object.
      @In, signal_name, str, name of signal in ARMA model
      @In, set_years, list, project years to simulate
      @Out, synHist, dict, dictionary including synthetic history and metadata
    """
    # sampling from ARMA model object
    synHist = self.runner.evaluate(self.inp)[0]
    newSynHist = {} # empty dictionary with restructured data

    # check that signal name included in synthetic history dictionary
    if signal_name not in synHist.keys():
      raise Exception(f"Signal name {signal_name} not found in sampled history keys: {synHist.keys()}")

    # extract actual synthetic history data
    synHistData = synHist[signal_name]

    # extract hourly, daily/cluster/ and year data arrays
    indexMap = synHist['_indexMap'][0][signal_name]
    if ('Time' in indexMap) or ('hour' in indexMap):
      hourKey = 'Time' if 'Time' in indexMap else 'hour'
    if '_ROM_Cluster' in indexMap:
      clusterKey = '_ROM_Cluster'
    if 'Year' in indexMap:
      yearKey = 'Year'

    # time sets from generated synthetic history
    synHistHours = np.asarray(synHist[hourKey]+1, dtype=int)
    synHistDays  = np.asarray(synHist[clusterKey]+1, dtype=int) # 20 clusters
    synHistYears = synHist[yearKey] # 2018-2045

    # checking that simulation years are included within synthetic history year set
    if not set(set_years).issubset(synHistYears):
      raise Exception(f"Years requested ({set_years}) not provided by ARMA Model: {synHistYears}")

    # getting cluster weights from ROM (deep hierarchy)
    #   using attrgetter to extract protected members -> self.runner.rom._segmentROM._macroSteps
    cluster_steps = operator.attrgetter("runner.rom._segmentROM._macroSteps")(self)
    newSynHist['weights_days'] = {}
    newSynHist['LMP'] = {}

    # loop through ROM years to extract clusters per year
    for year in set_years:
      newSynHist['weights_days'][year] = {}

      for cluster in synHistDays:
        # using attrgetter to extract protected members -> cluster_steps[year]._clusterInfo['map']
        cluster_map = operator.attrgetter('_clusterInfo')(cluster_steps[year])['map']
        cluster_ind = int(cluster-1)
        newSynHist['weights_days'][year][cluster] = len(cluster_map[cluster_ind])

    newSynHist['LMP'] = {year: {day: {hour: synHistData[y, int(day-1), int(hour-1)]
                            for hour in synHistHours}
                  for day in synHistDays}
            for y, year in enumerate(set_years)}

    return newSynHist
