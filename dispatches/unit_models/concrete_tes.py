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

# Pyomo imports
from pyomo.environ import (
    Var,
    RangeSet,
    Constraint,
    NonNegativeReals,
    Reference,
    TransformationFactory,
    units as pyunits,
    sqrt,
    log,
)
from pyomo.common.config import ConfigValue, In
from pyomo.network import Arc

# IDAES imports
from idaes.core import (declare_process_block_class,
                        UnitModelBlockData)
from idaes.models.unit_models import Heater
from idaes.core.util.config import is_physical_parameter_block
from idaes.core.util.initialization import propagate_state
from idaes.core.util.constants import Constants
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util import from_json, to_json, StoreSpec
from idaes.core.solvers import get_solver
import idaes.logger as idaeslog
import idaes.core.util.scaling as iscale

# Set up logger
_log = idaeslog.getLogger(__name__)


def u_tes(r, k, a, b):
    zz = r + ((a ** 3 * (4 * b ** 2 - a ** 2) + a * b ** 4 * (4 * log(b / a) - 3)) /
              (4 * k * (b ** 2 - a ** 2) ** 2))
    return 1 / zz


def _add_inlet_outlet_ports(self):
    operating_mode = self.config.operating_mode

    if operating_mode == "charge" or operating_mode == "combined":
        self.charge_inlet = self.config.property_package.build_state_block(
            self.flowsheet().time,
            defined_state=True,
            has_phase_equilibrium=True,
        )

        self.charge_outlet = self.config.property_package.build_state_block(
            self.flowsheet().time,
            defined_state=True,
            has_phase_equilibrium=True,
        )

        @self.Constraint(self.flowsheet().time, self.time_periods)
        def charge_inlet_flow_mol_equality(blk, t, p):
            return (
                blk.charge_inlet[t].flow_mol == 
                blk.num_tubes * blk.period[p].tube_charge.inlet.flow_mol[t]
            )

        @self.Constraint(self.flowsheet().time, self.time_periods)
        def charge_inlet_enth_mol_equality(blk, t, p):
            return (
                blk.charge_inlet[t].enth_mol == blk.period[p].tube_charge.inlet.enth_mol[t]
            )

        @self.Constraint(self.flowsheet().time, self.time_periods)
        def charge_inlet_pressure_equality(blk, t, p):
            return (
                blk.charge_inlet[t].pressure == blk.period[p].tube_charge.inlet.pressure[t]
            )

        p = len(self.time_periods)

        @self.Constraint(self.flowsheet().time)
        def charge_outlet_flow_mol_equality(blk, t):
            return (
                blk.charge_outlet[t].flow_mol == 
                blk.num_tubes * blk.period[p].tube_charge.outlet.flow_mol[t]
            )

        @self.Constraint(self.flowsheet().time)
        def charge_outlet_enth_mol_equality(blk, t):
            return (
                blk.charge_outlet[t].enth_mol == blk.period[p].tube_charge.outlet.enth_mol[t]
            )

        @self.Constraint(self.flowsheet().time)
        def charge_outlet_pressure_equality(blk, t):
            return (
                blk.charge_outlet[t].pressure == blk.period[p].tube_charge.outlet.pressure[t]
            )

        self.add_port(name="inlet_charge", block=self.charge_inlet,)
        self.add_port(name="outlet_charge", block=self.charge_outlet,)

    if operating_mode == "discharge" or operating_mode == "combined":
        self.discharge_inlet = self.config.property_package.build_state_block(
            self.flowsheet().time,
            defined_state=True,
            has_phase_equilibrium=True,
        )

        self.discharge_outlet = self.config.property_package.build_state_block(
            self.flowsheet().time,
            defined_state=True,
            has_phase_equilibrium=True,
        )

        @self.Constraint(self.flowsheet().time, self.time_periods)
        def discharge_inlet_flow_mol_equality(blk, t, p):
            return (
                blk.discharge_inlet[t].flow_mol == blk.num_tubes *
                blk.period[p].tube_discharge.inlet.flow_mol[t]
            )

        @self.Constraint(self.flowsheet().time, self.time_periods)
        def discharge_inlet_enth_mol_equality(blk, t, p):
            return (
                blk.discharge_inlet[t].enth_mol == blk.period[p].tube_discharge.inlet.enth_mol[t]
            )

        @self.Constraint(self.flowsheet().time, self.time_periods)
        def discharge_inlet_pressure_equality(blk, t, p):
            return (
                blk.discharge_inlet[t].pressure == blk.period[p].tube_discharge.inlet.pressure[t]
            )

        p = len(self.time_periods)

        @self.Constraint(self.flowsheet().time)
        def discharge_outlet_flow_mol_equality(blk, t):
            return (
                blk.discharge_outlet[t].flow_mol == blk.num_tubes *
                blk.period[p].tube_discharge.outlet.flow_mol[t]
            )

        @self.Constraint(self.flowsheet().time)
        def discharge_outlet_enth_mol_equality(blk, t):
            return (
                blk.discharge_outlet[t].enth_mol ==
                blk.period[p].tube_discharge.outlet.enth_mol[t]
            )

        @self.Constraint(self.flowsheet().time)
        def discharge_outlet_pressure_equality(blk, t):
            return (
                blk.discharge_outlet[t].pressure ==
                blk.period[p].tube_discharge.outlet.pressure[t]
            )

        self.add_port(name="inlet_discharge", block=self.discharge_inlet,)
        self.add_port(name="outlet_discharge", block=self.discharge_outlet,)


@declare_process_block_class("ConcreteBlock")
class ConcreteBlockData(UnitModelBlockData):
    """
    Concrete block Model Class.
    """
    CONFIG = UnitModelBlockData.CONFIG()
    CONFIG.declare(
        "segments_set",
        ConfigValue(
            doc="""Pointer to the set of segments""",
        ),
    )

    # noinspection PyAttributeOutsideInit
    def build(self):
        super().build()

        # Declare variables
        self.therm_cond = Var(
            within=NonNegativeReals,
            bounds=(0, 10),
            initialize=1,
            doc="Thermal conductivity of the concrete",
            units=pyunits.W / (pyunits.m * pyunits.K),
        )
        self.delta_time = Var(
            within=NonNegativeReals,
            bounds=(0, 4000),
            initialize=3600,
            doc="Delta for discretization of time",
            units=pyunits.s,
        )
        self.dens_mass = Var(
            within=NonNegativeReals,
            bounds=(0, 3000),
            initialize=2240,
            doc="Concrete density",
            units=pyunits.kg / pyunits.m ** 3,
        )
        self.cp_mass = Var(
            within=NonNegativeReals,
            bounds=(0, 1000),
            initialize=900,
            doc="Concrete specific heat",
            units=pyunits.J / (pyunits.kg * pyunits.K),
        )
        self.face_area = Var(
            within=NonNegativeReals,
            bounds=(0.003, 0.015),
            initialize=0.01,
            doc='Face (cross-sectional) area of the concrete wall',
            units=pyunits.m ** 2,
        )
        self.delta_z = Var(
            within=NonNegativeReals,
            bounds=(0, 100),
            initialize=5,
            doc='Length of each concrete segment',
            units=pyunits.m,
        )

        temperature_wall_index = self.config.segments_set
        self.init_temperature = Var(
            temperature_wall_index,
            within=NonNegativeReals,
            bounds=(300, 900),
            initialize=600,
            doc='Initial concrete wall temperature profile',
            units=pyunits.K,
            )
        self.temperature = Var(
            temperature_wall_index,
            within=NonNegativeReals,
            bounds=(300, 900),
            initialize=600,
            doc='Final concrete wall temperature',
            units=pyunits.K,
        )
        self.heat_rate = Var(
            temperature_wall_index,
            bounds=(-10000, 10000),
            initialize=600,
            doc='Heat transferred from/to the concrete segment',
            units=pyunits.W,
        )

        # Declare constraints
        @self.Constraint(temperature_wall_index)
        def temp_segment_constraint(b, s):
            volume = b.face_area * b.delta_z

            return (
                b.temperature[s] == b.init_temperature[s] +
                (b.delta_time * b.heat_rate[s]) / (b.dens_mass * b.cp_mass * volume)
            )

    def initialize_build(self, state_args=None, outlvl=idaeslog.NOTSET, solver=None, optarg=None):

        init_log = idaeslog.getInitLogger(self.name, outlvl, tag="unit")
        solve_log = idaeslog.getSolveLogger(self.name, outlvl, tag="unit")
        solver = get_solver(solver=solver, options=optarg)

        # TODO: Need to print a proper error message
        assert degrees_of_freedom(self) == 0

        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            res = solver.solve(self, tee=slc.tee)

        init_log.info_high("Initialization Step: {}.".format(idaeslog.condition(res)))
        init_log.info("Initialization Complete.")

    def calculate_scaling_factors(self):
        super().calculate_scaling_factors()


@declare_process_block_class("TubeSideHex")
class TubeSideHexData(UnitModelBlockData):
    """ConcreteTubeSide 1D Unit Model Class.
    """

    CONFIG = UnitModelBlockData.CONFIG()
    CONFIG.declare(
        "has_pressure_change",
        ConfigValue(
            default=False,
            domain=In([True, False]),
            description="Pressure change term construction flag",
            doc="""Indicates whether terms for pressure change should be constructed,
            **default** - False.
            **Valid values:** {
            **True** - include pressure change terms,
            **False** - exclude pressure change terms.}""",
        ),
    )
    CONFIG.declare(
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
    CONFIG.declare(
        "property_package_args",
        ConfigValue(
            default={},
            description="Arguments for constructing shell property package",
            doc="""A dict of arguments to be passed to the PropertyBlockData
            and used when constructing these (default = 'use_parent_value')
            - 'use_parent_value' - get package from parent (default = None)
            - a dict (see property package for documentation)""",
        ),
    )
    CONFIG.declare(
        "operating_mode",
        ConfigValue(
            default="charge",
            domain=In(["charge", "discharge"]),
            doc="""Mode of operation. """,
        ),
    )
    CONFIG.declare(
        "segments_set",
        ConfigValue(
            doc="""Pointer to the set of segments""",
        ),
    )
    CONFIG.declare(
        "use_surrogate",
        ConfigValue(
            default=False,
            domain=In([True, False]),
            description="Surrogate construction flag",
            doc="""Indicates whether a surrogate model should be used
            for enthalpy-temperature relation""",
        ),
    )
    CONFIG.declare(
        "surrogate_data",
        ConfigValue(
            default=[],
            doc="""Pointer to the points needed to construct the surrogate""",
        ),
    )

    # noinspection PyAttributeOutsideInit
    def build(self):

        # Call UnitModel.build to setup dynamics
        super().build()

        segments = self.config.segments_set
        op_mode = self.config.operating_mode

        self.hex = Heater(
            segments,
            property_package=self.config.property_package,
            has_pressure_change=self.config.has_pressure_change,
        )

        # Add the tube diameter variable
        self.tube_outer_dia = Var(
            domain=NonNegativeReals,
            initialize=0.01,
            doc="Diameter of the outer surface of the tube",
            units=pyunits.m,
        )

        # Add the tube_length variable
        self.tube_length = Var(
            domain=NonNegativeReals,
            initialize=50,
            doc="Inner surface area of the tube segment",
            units=pyunits.m,
        )

        # Connect heat exchangers with arcs
        if op_mode == "charge":
            for i in range(1, len(segments)):
                setattr(self, "sc" + str(i), Arc(source=self.hex[i].outlet,
                                                 destination=self.hex[i + 1].inlet))
                
        elif op_mode == "discharge":
            for i in range(1, len(segments)):
                setattr(self, "sc" + str(i), Arc(source=self.hex[i + 1].outlet,
                                                 destination=self.hex[i].inlet))

        tube_units = self.config.property_package.get_metadata().get_derived_units
        # Add the additional variables and heat transfer constraint
        for i in segments:
            obj = self.hex[i]

            # Add a variable for tube diameter, segment length, heat transfer coefficient 
            # and the temperature of the wall
            obj.tube_outer_dia = Var(
                domain=NonNegativeReals,
                initialize=0.01,
                doc="Outer diameter of the tube",
                units=tube_units("length"),
            )
            obj.segment_length = Var(
                domain=NonNegativeReals,
                initialize=4,
                doc="Length of each segment",
                units=tube_units("length"),
            )
            obj.htc = Var(
                domain=NonNegativeReals,
                initialize=70,
                doc="Heat transfer coefficient",
                units=tube_units("heat_transfer_coefficient"),
            )
            obj.temperature_wall = Var(
                domain=NonNegativeReals,
                bounds=(300, 900),
                initialize=500,
                doc="Temperature of the concrete block",
                units=tube_units("temperature"),
            )
            
            if self.config.use_surrogate:
                pass
            else:
                # Add energy transfer between tube wall and tube
                obj.tube_heat_transfer_eq = Constraint(
                    expr=obj.heat_duty[0] == obj.htc
                         * (Constants.pi * obj.tube_outer_dia * obj.segment_length)
                         * (obj.temperature_wall -
                            obj.control_volume.properties_out[0].temperature),
                    doc="Convective heat transfer",
                )

        # Add constraints to equate the tube diameter and length
        @self.Constraint(segments)
        def equate_tube_diameter(blk, s):
            return blk.tube_outer_dia == blk.hex[s].tube_outer_dia

        @self.Constraint(segments)
        def equate_tube_length(blk, s):
            return blk.tube_length == blk.hex[s].segment_length * len(segments)

        TransformationFactory("network.expand_arcs").apply_to(self)

    @property
    def inlet(self):
        if self.config.operating_mode == "charge":
            return self.hex[1].inlet
        elif self.config.operating_mode == "discharge":
            return self.hex[len(self.config.segments_set)].inlet

    @property
    def outlet(self):
        if self.config.operating_mode == "charge":
            return self.hex[len(self.config.segments_set)].outlet
        elif self.config.operating_mode == "discharge":
            return self.hex[1].outlet

    def initialize_build(
        self, state_args=None, outlvl=idaeslog.NOTSET, solver=None, optarg=None
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
        init_log = idaeslog.getInitLogger(self.name, outlvl, tag="unit")
        solve_log = idaeslog.getSolveLogger(self.name, outlvl, tag="unit")

        solver = get_solver(solver=solver, options=optarg)
        segment_length = self.tube_length.value / len(self.config.segments_set)

        if self.config.operating_mode == "charge":
            for i in self.config.segments_set:
                self.hex[i].tube_outer_dia.fix(self.tube_outer_dia.value)
                self.hex[i].segment_length.fix(segment_length)

                self.hex[i].initialize(outlvl=outlvl)

                self.hex[i].tube_outer_dia.unfix()
                self.hex[i].segment_length.unfix()

                if i != len(self.config.segments_set):
                    propagate_state(eval("self.sc" + str(i)))

        elif self.config.operating_mode == "discharge":
            for i in range(len(self.config.segments_set), 0, -1):
                self.hex[i].tube_outer_dia.fix(self.tube_outer_dia.value)
                self.hex[i].segment_length.fix(self.tube_length)

                self.hex[i].initialize(outlvl=outlvl)

                self.hex[i].tube_outer_dia.unfix()
                self.hex[i].segment_length.unfix()

                if i != 1:
                    propagate_state(eval("self.sc" + str(i - 1)))

        init_log.info_high("Initialization Step 1 Complete.")

        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            res = solver.solve(self, tee=slc.tee)
        init_log.info_high("Initialization Step 2 {}.".format(idaeslog.condition(res)))

        init_log.info("Initialization Complete.")

    def calculate_scaling_factors(self):
        super().calculate_scaling_factors()

        # for i in self.config.segments_set:
        #     iscale.constraint_scaling_transform(
        #         self.hex[i].tube_heat_transfer_eq, 1)


@declare_process_block_class("ConcreteTES")
class ConcreteTESData(UnitModelBlockData):
    """
    Concrete TES Model Class.
    """

    CONFIG = UnitModelBlockData.CONFIG()

    CONFIG.declare(
        "model_data",
        ConfigValue(
            default=None,
            description="Pointer to the dictionary containing model data"
        ),
    )
    CONFIG.declare(
        "has_pressure_change",
        ConfigValue(
            default=False,
            domain=In([True, False]),
            description="Pressure change term construction flag",
            doc="""Indicates whether terms for pressure change should be constructed,
                **default** - False.
                **Valid values:** {
                **True** - include pressure change terms,
                **False** - exclude pressure change terms.}""",
        ),
    )
    CONFIG.declare(
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
    CONFIG.declare(
        "property_package_args",
        ConfigValue(
            default={},
            description="Arguments for constructing shell property package",
            doc="""A dict of arguments to be passed to the PropertyBlockData
                and used when constructing these (default = 'use_parent_value')
                - 'use_parent_value' - get package from parent (default = None)
                - a dict (see property package for documentation)""",
        ),
    )
    CONFIG.declare(
        "operating_mode",
        ConfigValue(
            default="charge",
            domain=In(["charge", "discharge", "combined"]),
            doc="""Mode of operation. """,
        ),
    )
    CONFIG.declare(
        "use_surrogate",
        ConfigValue(
            default=False,
            domain=In([True, False]),
            description="Surrogate construction flag",
            doc="""Indicates whether a surrogate model should be used
                for enthalpy-temperature relation""",
        ),
    )
    CONFIG.declare(
        "inlet_pressure_data",
        ConfigValue(
            default={"charge": 24235081, "discharge": 8.5e5},
            doc="""Pointer to the points needed to construct the surrogate""",
        ),
    )

    # noinspection PyAttributeOutsideInit
    def build(self):
        super().build()

        data = self.config.model_data
        required_prop_list = [
            "num_tubes", "num_segments", "num_time_periods", "tube_length", 
            "tube_diameter", "therm_cond_concrete", "dens_mass_concrete", 
            "cp_mass_concrete", "init_temperature_concrete", "face_area", 
        ]
        for prop in required_prop_list:
            if prop not in data:
                raise KeyError(f"Property {prop}, a required argument, is not in model_data")

        data["delta_time"] = 3600 / data["num_time_periods"]

        property_package = self.config.property_package
        operating_mode = self.config.operating_mode

        self.time_periods = RangeSet(data['num_time_periods'])
        self.num_tubes = Var(
            within=NonNegativeReals,
            bounds=(1, 100000),
            doc='Number of tubes of the concrete TES',
            units=None,
        )
        self.num_tubes.fix(data['num_tubes'])
        self.segments = RangeSet(data["num_segments"])

        # Construct the model for each time period
        @self.Block(self.time_periods)
        def period(m):
            # Add the concrete side model
            m.concrete = ConcreteBlock(
                segments_set=m.parent_block().segments,
            )

            # Add the tube side charge model
            if operating_mode == "charge" or operating_mode == "combined":
                m.tube_charge = TubeSideHex(
                    property_package=property_package,
                    has_pressure_change=self.config.has_pressure_change,
                    segments_set=m.parent_block().segments,
                    operating_mode="charge",
                )

                @m.Constraint(m.parent_block().segments)
                def temperature_equality_constraints_charge(blk, s):
                    return blk.tube_charge.hex[s].temperature_wall == blk.concrete.temperature[s]

            # Add the tube side discharge model
            if operating_mode == "discharge" or operating_mode == "combined":
                m.tube_discharge = TubeSideHex(
                    property_package=property_package,
                    has_pressure_change=self.config.has_pressure_change,
                    segments_set=m.parent_block().segments,
                    operating_mode="discharge",
                )

                @m.Constraint(m.parent_block().segments)
                def temperature_equality_constraints_discharge(blk, s):
                    return blk.tube_discharge.hex[s].temperature_wall == blk.concrete.temperature[s]

            # Linking constraint for the net Q_fluid var in the concrete model
            # Q_fluid = Q_charge - Q_discharge
            @m.Constraint(m.parent_block().segments)
            def heat_balance_constraints(blk, s):
                if operating_mode == "charge":
                    return blk.concrete.heat_rate[s] == -blk.tube_charge.hex[s].heat_duty[0]
                elif operating_mode == "discharge":
                    return blk.concrete.heat_rate[s] == -blk.tube_discharge.hex[s].heat_duty[0]
                elif operating_mode == "combined":
                    return (blk.concrete.heat_rate[s] == -blk.tube_charge.hex[s].heat_duty[0]
                            - blk.tube_discharge.hex[s].heat_duty[0])

            return m

        _add_inlet_outlet_ports(self)

        # Add constraints connecting different time periods
        @self.Constraint(self.time_periods, self.segments)
        def initial_temperature_constraints(blk, p, s):
            if p == 1:
                return Constraint.Skip
            return blk.period[p].concrete.init_temperature[s] == blk.period[p - 1].concrete.temperature[s]

        # Add the heat transfer coefficient surrogate model
        def htc_surrogate():
            # TODO: Need to declare a variable for outer radius of the
            #       concrete block and do the following calculation in a constraint
            # Also, do this inside each hex model instead of doing it
            # outside. Reduces a lot of additional constraints!
            face_area = data["face_area"]
            a = data["tube_diameter"] / 2  # Inner radius of the concrete block
            b = sqrt(face_area / Constants.pi + a ** 2)  # Outer radius
            k_red = 0.8  # Reduction factor for thermal conductivity
            k = data["therm_cond_concrete"] * k_red
            correction_factor = 1.31

            # Tube calculations are a bit complex. If inner diameter of the
            # tube is available, then r = log(OD/ID) * (OD/2) / k_tube. Since
            # ID is not readily available, we are setting r = 0.0001 (works very well)
            return u_tes(r=0.0001, k=k, a=a, b=b) / correction_factor

        # Fix the geometry of the tube, tube inlet conditions, and material properties of concrete
        for p in self.time_periods:
            self.period[p].concrete.delta_time.fix(data["delta_time"])
            self.period[p].concrete.therm_cond.fix(data["therm_cond_concrete"])
            self.period[p].concrete.dens_mass.fix(data["dens_mass_concrete"])
            self.period[p].concrete.cp_mass.fix(data["cp_mass_concrete"])
            self.period[p].concrete.face_area.fix(data["face_area"])
            self.period[p].concrete.delta_z.fix(data["tube_length"] / data["num_segments"])

            if operating_mode == "charge" or operating_mode == "combined":
                self.period[p].tube_charge.tube_outer_dia.fix(data["tube_diameter"])
                self.period[p].tube_charge.tube_length.fix(data["tube_length"])

                htc = htc_surrogate()
                for s in self.segments:
                    self.period[p].tube_charge.hex[s].htc.fix(htc)

            if operating_mode == "discharge" or operating_mode == "combined":
                self.period[p].tube_discharge.tube_outer_dia.fix(data["tube_diameter"])
                self.period[p].tube_discharge.tube_length.fix(data["tube_length"])

                htc = htc_surrogate()
                for s in self.segments:
                    self.period[p].tube_discharge.hex[s].htc.fix(htc)

        # Fixing the initial concrete temperature for the first time block
        for s in self.segments:
            self.period[1].concrete.init_temperature[s].fix(data["init_temperature_concrete"][s - 1])

    def initialize_build(self, state_args=None, outlvl=idaeslog.INFO_HIGH, solver=None, optarg=None):
        """
        Initialization routine for the entire model (default solver ipopt).

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
        init_log = idaeslog.getInitLogger(self.name, outlvl, tag="unit")
        solve_log = idaeslog.getSolveLogger(self.name, outlvl, tag="unit")

        # Store original specification so initialization doesn't change the model
        # This will only restore the values of variables that were originally fixed
        sp = StoreSpec.value_isfixed_isactive(only_fixed=True)

        if hasattr(self, "inlet_charge"):
            state_charge = to_json(self.charge_inlet, return_dict=True, wts=sp)
            self.inlet_charge.fix()

        if hasattr(self, "inlet_discharge"):
            state_discharge = to_json(self.discharge_inlet, return_dict=True, wts=sp)
            self.inlet_discharge.fix()

        operating_mode = self.config.operating_mode
        flags_charge = {}
        flags_discharge = {}
        if solver is None:
            solver = get_solver()

        solver.options = {
            # "tol": 1e-6,
            "max_iter": 100,
            # "halt_on_ampl_error": "yes",
            "bound_push": 1e-6,
            # "mu_init": 1e-5
        }

        for p in self.time_periods:
            if operating_mode == "charge" or operating_mode == "combined":
                tube_inlet = self.period[p].tube_charge.inlet
                tube_inlet.flow_mol[0].value = (self.inlet_charge.flow_mol[0].value /
                                                self.num_tubes.value)
                tube_inlet.pressure[0].value = self.inlet_charge.pressure[0].value
                tube_inlet.enth_mol[0].value = self.inlet_charge.enth_mol[0].value

            if operating_mode == "discharge" or operating_mode == "combined":
                tube_inlet = self.period[p].tube_discharge.inlet
                tube_inlet.flow_mol[0].value = (self.inlet_discharge.flow_mol[0].value /
                                                self.num_tubes.value)
                tube_inlet.pressure[0].value = self.inlet_discharge.pressure[0].value
                tube_inlet.enth_mol[0].value = self.inlet_discharge.enth_mol[0].value

        for p in self.time_periods:
            # Fix the initial temperature for p > 1
            if p > 1:
                for t in self.segments:
                    self.period[p].concrete.init_temperature[t].fix(
                        self.period[p - 1].concrete.temperature[t].value)

            # **************************************************
            #               INITIALIZING TUBE SIDE
            # **************************************************
            # TODO: Need to print a proper error message
            if operating_mode == "charge" or operating_mode == "combined":
                # Use the initial concrete block temperature for flux calculation
                for s in self.segments:
                    self.period[p].tube_charge.hex[s].temperature_wall.fix(
                        self.period[p].concrete.init_temperature[s].value)

                # Hold the state of the inlet
                flags_charge[p] = self.period[p].tube_charge.hex[1].control_volume.initialize(hold_state=True)
                self.period[p].tube_charge.initialize(outlvl=outlvl, optarg=solver.options)

                # Unfix the final wall temperature
                for s in self.segments:
                    self.period[p].tube_charge.hex[s].temperature_wall.unfix()

            if operating_mode == "discharge" or operating_mode == "combined":
                # Use the initial concrete block temperature for flux calculation
                for s in self.segments:
                    self.period[p].tube_discharge.hex[s].temperature_wall.fix(
                        self.period[p].concrete.init_temperature[s].value)

                # Hold the state of the inlet
                flags_discharge[p] = self.period[p].tube_discharge.hex[len(self.segments)].\
                    control_volume.initialize(hold_state=True)
                self.period[p].tube_discharge.initialize(outlvl=outlvl, optarg=solver.options)

                # Unfix the final wall temperature
                for s in self.segments:
                    self.period[p].tube_discharge.hex[s].temperature_wall.unfix()

            # **************************************************
            #            INITIALIZING CONCRETE SIDE
            # **************************************************
            for s in self.segments:
                if operating_mode == "charge":
                    heat_rate = -self.period[p].tube_charge.hex[s].heat_duty[0].value
                elif operating_mode == "discharge":
                    heat_rate = -self.period[p].tube_discharge.hex[s].heat_duty[0].value
                else:
                    heat_rate = - self.period[p].tube_charge.hex[s].heat_duty[0].value \
                                - self.period[p].tube_discharge.hex[s].heat_duty[0].value

                self.period[p].concrete.heat_rate[s].fix(heat_rate)

            self.period[p].concrete.initialize(outlvl=outlvl, optarg=solver.options)

            # Unfix q_fluid
            self.period[p].concrete.heat_rate.unfix()

            # **************************************************
            #               INITIALIZING TUBE + CONCRETE SIDES
            # **************************************************
            assert degrees_of_freedom(self.period[p]) == 0
            with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
                res = solver.solve(self.period[p], tee=slc.tee)
            init_log.info_high("Initialization of tube and concrete: {}.".format(idaeslog.condition(res)))

            # Unfix the initial concrete temperature for p > 1
            if p > 1:
                self.period[p].concrete.init_temperature.unfix()

        # **************************************************
        #               INITIALIZING THE ENTIRE MODEL
        # **************************************************
        p = len(self.time_periods)

        if operating_mode == "charge" or operating_mode == "combined":
            tube_outlet = self.period[p].tube_charge.outlet
            self.outlet_charge.flow_mol[0].value = \
                (tube_outlet.flow_mol[0].value * self.num_tubes.value)
            self.outlet_charge.pressure[0].value = tube_outlet.pressure[0].value
            self.outlet_charge.enth_mol[0].value = tube_outlet.enth_mol[0].value

        if operating_mode == "discharge" or operating_mode == "combined":
            tube_outlet = self.period[p].tube_discharge.outlet
            self.outlet_discharge.flow_mol[0].value = \
                (tube_outlet.flow_mol[0].value * self.num_tubes.value)
            self.outlet_discharge.pressure[0].value = tube_outlet.pressure[0].value
            self.outlet_discharge.enth_mol[0].value = tube_outlet.enth_mol[0].value

        for p in self.time_periods:
            if operating_mode == "charge" or operating_mode == "combined":
                self.period[p].tube_charge.hex[1].control_volume.release_state(flags_charge[p])

            if operating_mode == "discharge" or operating_mode == "combined":
                self.period[p].tube_discharge.hex[len(self.segments)].control_volume.release_state(flags_discharge[p])

        assert degrees_of_freedom(self) == 0
        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            res = solver.solve(self, tee=slc.tee)

        if hasattr(self, "inlet_charge"):
            from_json(self.charge_inlet, sd=state_charge, wts=sp)
        if hasattr(self, "inlet_discharge"):
            from_json(self.discharge_inlet, sd=state_discharge, wts=sp)

        init_log.info_high("Initialization TES: {}.".format(idaeslog.condition(res)))
        init_log.info("Initialization Complete.")

    def set_default_scaling_factors(self):
        for p in self.time_periods:
            tube_charge = self.period[p].tube_charge
            tube_discharge = self.period[p].tube_discharge
            concrete = self.period[p].concrete

            # iscale.set_scaling_factor(concrete.q_fluid, 1e-3)
            # iscale.set_scaling_factor(concrete.temperature, 1e-2)

            # if p != 1:
            #     iscale.set_scaling_factor(concrete.init_temperature, 1e-2)

            for i in self.segments:
                # Charge model
                iscale.set_scaling_factor(
                    tube_charge.hex[i].control_volume.heat[0], 1e-3)
                iscale.set_scaling_factor(
                    tube_charge.hex[i].inlet.flow_mol[0], 1)
                iscale.set_scaling_factor(
                    tube_charge.hex[i].outlet.flow_mol[0], 1)
                # iscale.set_scaling_factor(
                #     tube_charge.hex[i].d_tube_outer, 1e2)
                # iscale.set_scaling_factor(
                #     tube_charge.hex[i].tube_length, 1e-1)
                # iscale.set_scaling_factor(
                #     tube_charge.hex[i].htc, 1e-2)
                # iscale.set_scaling_factor(
                #     tube_charge.hex[i].temperature_wall, 1e-2)

                # Discharge model
                iscale.set_scaling_factor(
                    tube_discharge.hex[i].control_volume.heat[0], 1e-3)
                iscale.set_scaling_factor(
                    tube_discharge.hex[i].inlet.flow_mol[0], 1)
                iscale.set_scaling_factor(
                    tube_discharge.hex[i].outlet.flow_mol[0], 1)
                # iscale.set_scaling_factor(
                #     tube_discharge.hex[i].d_tube_outer, 1e2)
                # iscale.set_scaling_factor(
                #     tube_discharge.hex[i].tube_length, 1e-1)
                # iscale.set_scaling_factor(
                #     tube_discharge.hex[i].htc, 1e-2)
                # iscale.set_scaling_factor(
                #     tube_discharge.hex[i].temperature_wall, 1e-2)

