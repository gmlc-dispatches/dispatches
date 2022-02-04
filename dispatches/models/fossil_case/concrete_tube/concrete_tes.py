##############################################################################
# DISPATCHES was produced under the DOE Design Integration and Synthesis
# Platform to Advance Tightly Coupled Hybrid Energy Systems program (DISPATCHES),
# and is copyright (c) 2021 by the software owners: The Regents of the University
# of California, through Lawrence Berkeley National Laboratory, National
# Technology & Engineering Solutions of Sandia, LLC, Alliance for Sustainable
# Energy, LLC, Battelle Energy Alliance, LLC, University of Notre Dame du Lac, et
# al. All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. Both files are also available online at the URL:
# "https://github.com/gmlc-dispatches/dispatches".
#
##############################################################################
"""
TES surrogates for concrete wall temperature.
Heating source: steam
Thermal material: Concrete
Author: Andres J Calderon, Jaffer Ghouse, Storeworks
Date: July 19, 2021
"""

# Pyomo imports
from pyomo.environ import (Var,
                           RangeSet,
                           NonNegativeReals,
                           units as pyunits)
from pyomo.common.config import ConfigValue, In

# IDAES imports
from idaes.core import (declare_process_block_class,
                        UnitModelBlockData)
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util import get_solver
import idaes.logger as idaeslog

# Set up logger
_log = idaeslog.getLogger(__name__)


@declare_process_block_class("ConcreteBlock")
class ConcreteBlockData(UnitModelBlockData):
    """
    Concrete block Model Class.
    """
    def build(self):
        super().build()

        # Declare variables
        self.kappa = Var(within=NonNegativeReals,
                         bounds=(0, 10),
                         initialize=1,
                         doc="Thermal conductivity of the concrete",
                         units=pyunits.W / (pyunits.m * pyunits.K))
        self.delta_time = Var(within=NonNegativeReals,
                              bounds=(0, 4000),
                              initialize=3600,
                              doc="Delta for discretization of time",
                              units=pyunits.s)
        self.density = Var(within=NonNegativeReals,
                           bounds=(0, 3000),
                           initialize=2240,
                           doc="Concrete density",
                           units=pyunits.kg / pyunits.m ** 3)
        self.specific_heat = Var(within=NonNegativeReals,
                                 bounds=(0, 1000),
                                 initialize=900,
                                 doc="Concrete specific heat",
                                 units=pyunits.J / (pyunits.kg * pyunits.K))
        self.face_area = Var(within=NonNegativeReals,
                             bounds=(0.003, 0.015),
                             initialize=0.01,
                             doc='Face (cross-sectional) area of the concrete wall',
                             units=pyunits.m ** 2)
        self.delta_z = Var(within=NonNegativeReals,
                           bounds=(0, 100),
                           initialize=5,
                           doc='Delta for discretizing tube length',
                           units=pyunits.m)

        temperature_wall_index = self.parent_block().tube_charge.temperature_wall_index
        self.init_temperature = Var(temperature_wall_index,
                                    within=NonNegativeReals,
                                    bounds=(300, 900),
                                    initialize=600,
                                    doc='Initial concrete wall temperature profile',
                                    units=pyunits.K)
        self.q_fluid = Var(temperature_wall_index,
                           within=NonNegativeReals,
                           bounds=(-5000, 5000),
                           initialize=600,
                           doc='Q transferred from the steam to the concrete segment',
                           units=pyunits.W)
        self.temperature = Var(temperature_wall_index,
                               within=NonNegativeReals,
                               bounds=(300, 900),
                               initialize=600,
                               doc='Final concrete wall temperature',
                               units=pyunits.K)
        num_segments = len(temperature_wall_index)

        # Declare constraints
        @self.Constraint(RangeSet(num_segments))
        def temp_segment_constraint(blk, s):

            # Define 'alpha': thermal diffusivity
            volume = blk.face_area * blk.delta_z

            # Define the derivative value
            seg = temperature_wall_index.ordered_data()[s - 1]

            return (
                blk.temperature[seg] == blk.init_temperature[seg] +
                + (blk.delta_time * blk.q_fluid[seg]) / (blk.density * blk.specific_heat * volume)
            )

    def initialize(self, state_args=None, outlvl=idaeslog.NOTSET, solver=None, optarg=None):

        init_log = idaeslog.getInitLogger(self.name, outlvl, tag="unit")
        solve_log = idaeslog.getSolveLogger(self.name, outlvl, tag="unit")
        solver = get_solver(solver=solver, options=optarg)

        # TODO: Need to print a proper error message
        assert degrees_of_freedom(self) == 0

        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            res = solver.solve(self, tee=slc.tee)

        init_log.info_high("Initialization Step: {}.".format(idaeslog.condition(res)))
        init_log.info("Initialization Complete.")