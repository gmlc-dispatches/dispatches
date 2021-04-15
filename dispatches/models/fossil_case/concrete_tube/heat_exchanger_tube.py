##############################################################################
# Institute for the Design of Advanced Energy Systems Process Systems
# Engineering Framework (IDAES PSE Framework) Copyright (c) 2018-2020, by the
# software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia
# University Research Corporation, et al. All rights reserved.
#
# Please see the files COPYRIGHT.txt and LICENSE.txt for full copyright and
# license information, respectively. Both files are also available online
# at the URL "https://github.com/IDAES/idaes-pse".
##############################################################################
"""
Basic IDAES ConcreteTubeSide model Model.

"""

# Import Pyomo libraries
from pyomo.environ import (
    SolverFactory,
    Var,
    Constraint,
    units as pyunits
)
from pyomo.common.config import ConfigBlock, ConfigValue, In

# Import IDAES cores
from idaes.core import (
    ControlVolume1DBlock,
    UnitModelBlockData,
    declare_process_block_class,
    MaterialBalanceType,
    EnergyBalanceType,
    MomentumBalanceType,
    FlowDirection,
    useDefault,
)
from idaes.generic_models.unit_models.heat_exchanger \
    import HeatExchangerFlowPattern
from idaes.core.util.config import is_physical_parameter_block
from idaes.core.util.misc import add_object_reference
from idaes.core.util.exceptions import ConfigurationError
from idaes.core.util.tables import create_stream_table_dataframe
from idaes.core.util.constants import Constants as c
from idaes.core.util import scaling as iscale

import idaes.logger as idaeslog


__author__ = "Konica Mulani, Jaffer Ghouse"

# Set up logger
_log = idaeslog.getLogger(__name__)

@declare_process_block_class("ConcreteTubeSide")
class HeatExchanger1DDataNoshell(UnitModelBlockData):


    """Standard Heat Exchanger 1D Unit Model Class."""

    CONFIG = UnitModelBlockData.CONFIG()
    # Template for config arguments for tube side
    _SideTemplate = ConfigBlock()
    _SideTemplate.declare(
        "dynamic",
        ConfigValue(
            default=useDefault,
            domain=In([useDefault, True, False]),
            description="Dynamic model flag",
            doc="""Indicates whether this model will be dynamic or not,
**default** = useDefault.
**Valid values:** {
**useDefault** - get flag from parent (default = False),
**True** - set as a dynamic model,
**False** - set as a steady-state model.}""",
        ),
    )
    _SideTemplate.declare(
        "has_holdup",
        ConfigValue(
            default=useDefault,
            domain=In([useDefault, True, False]),
            description="Holdup construction flag",
            doc="""Indicates whether holdup terms should be constructed or not.
Must be True if dynamic = True,
**default** - False.
**Valid values:** {
**useDefault** - get flag from parent (default = False),
**True** - construct holdup terms,
**False** - do not construct holdup terms}""",
        ),
    )
    _SideTemplate.declare(
        "material_balance_type",
        ConfigValue(
            default=MaterialBalanceType.useDefault,
            domain=In(MaterialBalanceType),
            description="Material balance construction flag",
            doc="""Indicates what type of mass balance should be constructed,
**default** - MaterialBalanceType.useDefault.
**Valid values:** {
**MaterialBalanceType.useDefault - refer to property package for default
balance type
**MaterialBalanceType.none** - exclude material balances,
**MaterialBalanceType.componentPhase** - use phase component balances,
**MaterialBalanceType.componentTotal** - use total component balances,
**MaterialBalanceType.elementTotal** - use total element balances,
**MaterialBalanceType.total** - use total material balance.}""",
        ),
    )
    _SideTemplate.declare(
        "energy_balance_type",
        ConfigValue(
            default=EnergyBalanceType.useDefault,
            domain=In(EnergyBalanceType),
            description="Energy balance construction flag",
            doc="""Indicates what type of energy balance should be constructed,
**default** - EnergyBalanceType.useDefault.
**Valid values:** {
**EnergyBalanceType.useDefault - refer to property package for default
balance type
**EnergyBalanceType.none** - exclude energy balances,
**EnergyBalanceType.enthalpyTotal** - single enthalpy balance for material,
**EnergyBalanceType.enthalpyPhase** - enthalpy balances for each phase,
**EnergyBalanceType.energyTotal** - single energy balance for material,
**EnergyBalanceType.energyPhase** - energy balances for each phase.}""",
        ),
    )
    _SideTemplate.declare(
        "momentum_balance_type",
        ConfigValue(
            default=MomentumBalanceType.pressureTotal,
            domain=In(MomentumBalanceType),
            description="Momentum balance construction flag",
            doc="""Indicates what type of momentum balance should be constructed,
**default** - MomentumBalanceType.pressureTotal.
**Valid values:** {
**MomentumBalanceType.none** - exclude momentum balances,
**MomentumBalanceType.pressureTotal** - single pressure balance for material,
**MomentumBalanceType.pressurePhase** - pressure balances for each phase,
**MomentumBalanceType.momentumTotal** - single momentum balance for material,
**MomentumBalanceType.momentumPhase** - momentum balances for each phase.}""",
        ),
    )
    _SideTemplate.declare(
        "has_pressure_change",
        ConfigValue(
            default=False,
            domain=In([True, False]),
            description="Pressure change term construction flag",
            doc="""Indicates whether terms for pressure change should be
constructed,
**default** - False.
**Valid values:** {
**True** - include pressure change terms,
**False** - exclude pressure change terms.}""",
        ),
    )
    _SideTemplate.declare(
        "has_phase_equilibrium",
        ConfigValue(
            default=False,
            domain=In([True, False]),
            description="Phase equilibrium term construction flag",
            doc="""Argument to enable phase equilibrium on the shell side.
- True - include phase equilibrium term
- False - do not include phase equilibrium term""",
        ),
    )
    _SideTemplate.declare(
        "property_package",
        ConfigValue(
            default=None,
            domain=is_physical_parameter_block,
            description="Property package to use for control volume",
            doc="""Property parameter object used to define property calculations
(default = 'use_parent_value')
- 'use_parent_value' - get package from parent (default = None)
- a ParameterBlock object""",
        ),
    )
    _SideTemplate.declare(
        "property_package_args",
        ConfigValue(
            default={},
            description="Arguments for constructing shell property package",
            doc="""A dict of arguments to be passed to the PropertyBlockData
and used when constructing these
(default = 'use_parent_value')
- 'use_parent_value' - get package from parent (default = None)
- a dict (see property package for documentation)""",
        ),
    )
    _SideTemplate.declare(
        "transformation_method",
        ConfigValue(
            default=useDefault,
            description="Discretization method to use for DAE transformation",
            doc="""Discretization method to use for DAE transformation. See Pyomo
documentation for supported transformations.""",
        ),
    )
    _SideTemplate.declare(
        "transformation_scheme",
        ConfigValue(
            default=useDefault,
            description="Discretization scheme to use for DAE transformation",
            doc="""Discretization scheme to use when transformating domain. See
Pyomo documentation for supported schemes.""",
        ),
    )

    # Create individual config blocks for tube side
    CONFIG.declare(
        "tube_side", _SideTemplate(doc="tube side config arguments"))

    # config args for tube side
    CONFIG.declare(
        "finite_elements",
        ConfigValue(
            default=20,
            domain=int,
            description="Number of finite elements length domain",
            doc="""Number of finite elements to use when discretizing length
domain (default=20)""",
        ),
    )
    CONFIG.declare(
        "collocation_points",
        ConfigValue(
            default=5,
            domain=int,
            description="Number of collocation points per finite element",
            doc="""Number of collocation points to use per finite element when
discretizing length domain (default=3)""",
        ),
    )
    CONFIG.declare(
        "flow_type",
        ConfigValue(
            default=HeatExchangerFlowPattern.cocurrent,
            domain=In(HeatExchangerFlowPattern),
            description="Flow configuration of heat exchanger",
            doc="""Flow configuration of heat exchanger
- HeatExchangerFlowPattern.cocurrent: shell and tube flows from 0 to 1
(default)
- HeatExchangerFlowPattern.countercurrent: shell side flows from 0 to 1
tube side flows from 1 to 0""",
        ),
    )

    def build(self):
        """
        Begin building model (pre-DAE transformation).

        Args:
            None

        Returns:
            None
        """
        # Call UnitModel.build to setup dynamics
        super(HeatExchanger1DDataNoshell, self).build()

        # Set flow directions for the control volume blocks and specify
        # dicretisation if not specified.
        if self.config.flow_type == HeatExchangerFlowPattern.cocurrent:

            set_direction_tube = FlowDirection.forward

            if self.config.tube_side.transformation_method is useDefault:
                _log.warning(
                    "Discretization method was "
                    "not specified for the tube side of the "
                    "co-current heat exchanger. "
                    "Defaulting to finite "
                    "difference method on the tube side."
                )
                self.config.tube_side.transformation_method = \
                    "dae.finite_difference"

            if self.config.tube_side.transformation_scheme is useDefault:
                _log.warning(
                    "Discretization scheme was "
                    "not specified for the tube side of the "
                    "co-current heat exchanger. "
                    "Defaulting to backward finite "
                    "difference on the tube side."
                )
                self.config.tube_side.transformation_scheme = "BACKWARD"
        elif self.config.flow_type == HeatExchangerFlowPattern.countercurrent:
            set_direction_tube = FlowDirection.backward

            if self.config.tube_side.transformation_method is useDefault:
                _log.warning(
                    "Discretization method was "
                    "not specified for the tube side of the "
                    "counter-current heat exchanger. "
                    "Defaulting to finite "
                    "difference method on the tube side."
                )
                self.config.tube_side.transformation_method = \
                    "dae.finite_difference"

            if self.config.tube_side.transformation_scheme is useDefault:
                _log.warning(
                    "Discretization scheme was "
                    "not specified for the tube side of the "
                    "counter-current heat exchanger. "
                    "Defaulting to forward finite "
                    "difference on the tube side."
                )
                self.config.tube_side.transformation_scheme = "BACKWARD"
        else:
            raise ConfigurationError(
                "{} HeatExchanger1D only supports cocurrent and "
                "countercurrent flow patterns, but flow_type configuration"
                " argument was set to {}.".
                format(self.name, self.config.flow_type)
            )

        self.tube = ControlVolume1DBlock(
            default={
                "dynamic": self.config.tube_side.dynamic,
                "has_holdup": self.config.tube_side.has_holdup,
                "property_package": self.config.tube_side.property_package,
                "property_package_args":
                self.config.tube_side.property_package_args,
                "transformation_method":
                self.config.tube_side.transformation_method,
                "transformation_scheme":
                self.config.tube_side.transformation_scheme,
                "finite_elements": self.config.finite_elements,
                "collocation_points": self.config.collocation_points,
            }
        )

        self.tube.add_geometry(flow_direction=set_direction_tube)

        self.tube.add_state_blocks(
            information_flow=set_direction_tube,
            has_phase_equilibrium=self.config.tube_side.has_phase_equilibrium,
        )

        # Populate tube
        self.tube.add_material_balances(
            balance_type=self.config.tube_side.material_balance_type,
            has_phase_equilibrium=self.config.tube_side.has_phase_equilibrium,
        )

        self.tube.add_energy_balances(
            balance_type=self.config.tube_side.energy_balance_type,
            has_heat_transfer=True,
        )

        self.tube.add_momentum_balances(
            balance_type=self.config.tube_side.momentum_balance_type,
            has_pressure_change=self.config.tube_side.has_pressure_change,
        )

        self.tube.apply_transformation()

        # Add Ports for tube side
        self.add_inlet_port(name="tube_inlet", block=self.tube)
        self.add_outlet_port(name="tube_outlet", block=self.tube)

        # Add reference to control volume geometry
        add_object_reference(self, "tube_area", self.tube.area)
        add_object_reference(self, "tube_length", self.tube.length)

        self._make_performance()

    def _make_performance(self):
        """
        Constraints for unit model.

        Args:
            None

        Returns:
            None
        """
        tube_units = self.config.tube_side.property_package.\
            get_metadata().get_derived_units

        self.d_tube_outer = Var(initialize=0.011,
                                doc="Outer diameter of tube",
                                units=tube_units("length"))
        self.d_tube_inner = Var(initialize=0.010,
                                doc="Inner diameter of tube",
                                units=tube_units("length"))

        self.tube_heat_transfer_coefficient = Var(
            self.flowsheet().config.time,
            self.tube.length_domain,
            initialize=50,
            doc="Heat transfer coefficient",
            units=tube_units("heat_transfer_coefficient")
        )

        self.temperature_wall = Var(
            self.flowsheet().config.time,
            self.tube.length_domain,
            initialize=298.15,
            units=tube_units("temperature")
        )

        # Energy transfer between tube wall and tube
        @self.Constraint(
            self.flowsheet().config.time,
            self.tube.length_domain,
            doc="Convective heat transfer",
        )
        def tube_heat_transfer_eq(self, t, x):
            return self.tube.heat[t, x] == self.tube_heat_transfer_coefficient[
                t, x
            ] * c.pi * pyunits.convert(self.d_tube_inner,
                                       to_units=tube_units("length")) * (
                pyunits.convert(self.temperature_wall[t, x],
                                to_units=tube_units('temperature')) -
                self.tube.properties[t, x].temperature
            )

        # Define tube area in terms of tube diameter
        self.area_calc_tube = Constraint(
            expr=4 * self.tube_area == c.pi * pyunits.convert(
                self.d_tube_inner, to_units=tube_units("length"))**2
        )

    def initialize(
        blk,
        tube_state_args=None,
        outlvl=idaeslog.NOTSET,
        solver="ipopt",
        optarg={"tol": 1e-6},
    ):
        """
        Initialization routine for the unit (default solver ipopt).

        Keyword Arguments:
            state_args : a dict of arguments to be passed to the property
                         package(s) to provide an initial state for
                         initialization (see documentation of the specific
                         property package) (default = {}).
            outlvl : sets output level of initialization routine
            optarg : solver options dictionary object (default={'tol': 1e-6})
            solver : str indicating whcih solver to use during
                     initialization (default = 'ipopt')

        Returns:
            None
        """
        init_log = idaeslog.getInitLogger(blk.name, outlvl, tag="unit")
        solve_log = idaeslog.getSolveLogger(blk.name, outlvl, tag="unit")

        opt = SolverFactory(solver)
        opt.options = optarg

        flags_tube = blk.tube.initialize(
            outlvl=outlvl,
            optarg=optarg,
            solver=solver,
            state_args=tube_state_args,
        )

        init_log.info_high("Initialization Step 1 Complete.")

        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            res = opt.solve(blk, tee=slc.tee)
        init_log.info_high(
            "Initialization Step 2 {}.".format(idaeslog.condition(res))
        )

        blk.tube.release_state(flags_tube)

        init_log.info("Initialization Complete.")

    def _get_performance_contents(self, time_point=0):
        var_dict = {}
        var_dict["Tube Area"] = self.tube.area
        var_dict["Tube Outer Diameter"] = self.d_tube_outer
        var_dict["Tube Inner Diameter"] = self.d_tube_inner
        var_dict["Tube Length"] = self.tube.length
        var_dict["Number of Tubes"] = self.N_tubes

        return {"vars": var_dict}

    def _get_stream_table_contents(self, time_point=0):
        return create_stream_table_dataframe(
            {
                "Tube Inlet": self.tube_inlet,
                "Tube Outlet": self.tube_outlet,
            },
            time_point=time_point,
        )

    def calculate_scaling_factors(self):
        super().calculate_scaling_factors()

        for i, c in self.tube_heat_transfer_eq.items():
            iscale.constraint_scaling_transform(c, iscale.get_scaling_factor(
                self.tube.heat[i], default=1, warning=True))

