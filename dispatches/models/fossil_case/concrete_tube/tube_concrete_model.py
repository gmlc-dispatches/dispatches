# Pyomo imports
from pyomo.environ import (Var,
                           RangeSet,
                           Constraint,
                           NonNegativeReals,
                           units as pyunits)
from pyomo.common.config import ConfigValue, In
from pyomo.util.calc_var_value import calculate_variable_from_constraint

# IDAES imports
from idaes.core import (declare_process_block_class,
                        UnitModelBlockData)
from idaes.generic_models.unit_models.heat_exchanger \
    import HeatExchangerFlowPattern

from idaes.generic_models.properties import iapws95
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util import get_solver
import idaes.core.util.scaling as iscale

from heat_exchanger_tube import ConcreteTubeSide
from concrete_tes import ConcreteBlock

import idaes.logger as idaeslog

# Set up logger
_log = idaeslog.getLogger(__name__)


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
        "property_package",
        ConfigValue()
    )

    def build(self):
        super().build()

        data = self.config.model_data
        property_package = self.config.property_package

        self.time_periods = RangeSet(data['time_periods'])

        self.number_tubes = Var(within=NonNegativeReals,
                                bounds=(1, 100000),
                                doc='Number of tubes of the concrete TES',
                                units=None)
        self.number_tubes.fix(data['number_tubes'])
        self.constant_htc = Var(within=NonNegativeReals,
                                bounds=(20, 500),
                                initialize=100,
                                units=pyunits.W / (pyunits.m ** 2 * pyunits.K))
        self.capex = Var(within=NonNegativeReals,
                         bounds=(0, 2000),
                         initialize=700,
                         doc="Bare erected cost (BER) per tube (in USD)")

        # Construct the model for each time period
        @self.Block(self.time_periods)
        def period(m):

            # Add the tube side model
            m.tube_charge = ConcreteTubeSide(default={
                "property_package": property_package,
                "flow_type": HeatExchangerFlowPattern.cocurrent,
                "transformation_method": "dae.finite_difference",
                "transformation_scheme": "BACKWARD",
                "has_pressure_change": True,
                "finite_elements": data["segments"] - 1})

            m.tube_discharge = ConcreteTubeSide(default={
                "property_package": property_package,
                "flow_type": HeatExchangerFlowPattern.countercurrent,
                "transformation_method": "dae.finite_difference",
                "transformation_scheme": "FORWARD",
                "has_pressure_change": True,
                "finite_elements": data["segments"] - 1})

            # Add the concrete side model
            m.concrete = ConcreteBlock()

            @m.Constraint(m.tube_charge.temperature_wall_index)
            def temperature_equality_constraints_charge(blk, t, s):
                return (blk.tube_charge.temperature_wall[t, s] ==
                        blk.concrete.temperature[t, s])

            @m.Constraint(m.tube_discharge.temperature_wall_index)
            def temperature_equality_constraints_discharge(blk, t, s):
                return (blk.tube_discharge.temperature_wall[t, s] ==
                        blk.concrete.temperature[t, s])

            # @m.Constraint(m.tube_charge.temperature_wall_index)
            # def heat_balance_constraints(blk, t, s):
            #     return (blk.concrete.q_fluid[t, s] ==
            #             - blk.tube_charge.tube.heat[t, s]
            #             * blk.concrete.delta_z)

            # Linking constraint for the net Q_fluid var in the concrete model
            # Q_fluid = Q_charge - Q_discharge
            @m.Constraint(m.tube_charge.temperature_wall_index)
            def heat_balance_constraints(blk, t, s):
                return (blk.concrete.q_fluid[t, s] ==
                         (- blk.tube_discharge.tube.heat[t, s]
                        - blk.tube_charge.tube.heat[t, s])
                        * blk.concrete.delta_z)

            return m

        # Add constraints connecting different time periods
        @self.Constraint(self.time_periods, self.period[1].tube_charge.temperature_wall_index)
        def initial_temperature_constraints(blk, p, t, s):
            if p == 1:
                return Constraint.Skip
            return (blk.period[p].concrete.init_temperature[t, s] ==
                    blk.period[p - 1].concrete.temperature[t, s])

        # Add constraints to equate the heat transfer coefficients
        @self.Constraint(self.time_periods, self.period[1].tube_charge.temperature_wall_index)
        def set_htc_charge_constraints(blk, p, t, s):
            return blk.constant_htc == blk.period[p].tube_charge.tube_heat_transfer_coefficient[t, s]

        @self.Constraint(self.time_periods, self.period[1].tube_discharge.temperature_wall_index)
        def set_htc_discharge_constraints(blk, p, t, s):
            return blk.constant_htc == blk.period[p].tube_discharge.tube_heat_transfer_coefficient[t, s]

        # Add the heat transfer coefficient surrogate model
        @self.Constraint()
        def htc_surrogate(blk):
            face_area = blk.period[1].concrete.face_area
            d_tube_outer = blk.period[1].tube_charge.d_tube_outer
            inlet_pressure = blk.period[1].tube_charge.tube_inlet.pressure[0]

            return (
                blk.constant_htc == 179.56349723936409645830
                - 153.30491285800843570541 * face_area
                + 0.90901584753593953850542E-003 * face_area ** -2
                - 0.15780375527247002417718E-005 * face_area ** -3
                - 8386.3028048661253706086 * d_tube_outer
                - 0.24273547980088984796351E-006 * d_tube_outer ** -4
                - 0.48582603845628324901185 * (inlet_pressure / 1000000)
            )

        # Add the CAPEX surrogate
        @self.Constraint()
        def capex_surrogate(blk):
            face_area = blk.period[1].concrete.face_area
            d_tube_outer = blk.period[1].tube_charge.d_tube_outer
            inlet_pressure = blk.period[1].tube_charge.tube_inlet.pressure[0]
            tube_length = blk.period[1].tube_charge.tube_length

            # Capex per tube = f(tube_length, concrete_area, tube_diameter, inlet_pressure)
            return (
                blk.capex == - 5.0330935178250681971690 * tube_length - 17482.067135713899915572 * face_area
                + 46204.834011172177270055 * d_tube_outer - 23.091759612009159496893 * (inlet_pressure / 1000000)
                - 0.15657040320651968794933E-005 * face_area ** -3
                - 0.15531449903948327771752E-003 * d_tube_outer ** -3
                + 5860311.3183172149583697 * (inlet_pressure / 1000000) ** -3
                - 219928.20303379974211566 * tube_length ** -2
                + 0.88773788864419283291957E-003 * face_area ** -2
                - 689783.54911381844431162 * (inlet_pressure / 1000000) ** -2
                + 6.6453732481704159695823 * d_tube_outer ** -1
                - 1.6141255301306027813979 * (d_tube_outer * (inlet_pressure / 1000000)) ** -3
                - 0.22290734378486113154439E-007 * (face_area * d_tube_outer) ** -2
                + 21.785281602999724270830 * (d_tube_outer * (inlet_pressure / 1000000)) ** -2
                + 209417.65877250320045277 * (tube_length * (inlet_pressure / 1000000)) ** -1
                + 883.99716857681733017671 * tube_length * face_area
                + 853.32799887986300291232 * tube_length * d_tube_outer
                + 1574960.0062072663567960 * face_area * d_tube_outer
            )

        # Adjust bounds on variables
        # TODO: Is there a better way to set the bounds?
        for p in self.time_periods:
            for (t, s) in self.period[p].tube_charge.temperature_wall_index:
                self.period[p].tube_charge.tube.properties[t, s].enth_mol.setlb(3000)
                self.period[p].tube_charge.tube.properties[t, s].enth_mol.value = 50000

                self.period[p].tube_charge.tube.properties[t, s].pressure.setlb(40 * 1e5)
                self.period[p].tube_charge.tube.properties[t, s].pressure.setub(300 * 1e5)
                self.period[p].tube_charge.tube.properties[t, s].pressure.value = 200 * 1e5

                self.period[p].tube_charge.temperature_wall[t, s].setlb(300)
                self.period[p].tube_charge.temperature_wall[t, s].setub(900)
                self.period[p].tube_charge.temperature_wall[t, s].value = 400

                self.period[p].tube_charge.tube.heat[t, s].value = -200

        # Fix the geometry of the tube, tube inlet conditions, and material properties of concrete
        for p in self.time_periods:
            self.period[p].tube_charge.tube_length.fix(data["tube_length"])
            self.period[p].tube_charge.d_tube_outer.fix(data["tube_diameter"])
            self.period[p].tube_charge.d_tube_inner.fix(data["tube_diameter"])

            self.period[p].tube_discharge.tube_length.fix(data["tube_length"])
            self.period[p].tube_discharge.d_tube_outer.fix(data["tube_diameter"])
            self.period[p].tube_discharge.d_tube_inner.fix(data["tube_diameter"])

            self.period[p].tube_charge.tube.deltaP.fix(data['deltaP'])
            self.period[p].tube_discharge.tube.deltaP.fix(data['deltaP'])

            self.period[p].concrete.delta_time.fix(data["delta_time"])
            self.period[p].concrete.kappa.fix(data["concrete_conductivity"])
            self.period[p].concrete.density.fix(data["concrete_density"])
            self.period[p].concrete.specific_heat.fix(data["concrete_specific_heat"])
            self.period[p].concrete.face_area.fix(data["concrete_area"])
            self.period[p].concrete.delta_z.fix(data["tube_length"] / (data["segments"] - 1))

        # Fixing the initial concrete temperature for the first time block
        for idx, t in enumerate(self.period[1].tube_charge.temperature_wall_index):
            self.period[1].concrete.init_temperature[t].fix(data['concrete_init_temp'][idx])

        # Calculate the heat transfer coefficient
        calculate_variable_from_constraint(self.constant_htc, self.htc_surrogate)

        # Calculate the capex
        calculate_variable_from_constraint(self.capex, self.capex_surrogate)

        # Set scaling factors
        for p in self.time_periods:
            iscale.set_scaling_factor(self.period[p].tube_charge.tube.area, 1e-2)
            iscale.set_scaling_factor(self.period[p].tube_charge.tube.heat, 1e-2)
            iscale.set_scaling_factor(self.period[p].tube_charge.tube_heat_transfer_coefficient, 1e-2)
            iscale.calculate_scaling_factors(self.period[p].tube_charge)

    def initialize(self, state_args=None, outlvl=idaeslog.INFO_HIGH, solver=None, optarg=None):
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

        data = self.config.model_data

        init_log = idaeslog.getInitLogger(self.name, outlvl, tag="unit")
        solve_log = idaeslog.getSolveLogger(self.name, outlvl, tag="unit")

        solver = get_solver(solver=solver)

        solver.options = {
            "tol": 1e-6,
            "max_iter": 100,
            "halt_on_ampl_error": "yes",
            "bound_push": 1e-1,
            # "mu_init": 1e-5
        }

        concrete_init_temp = data['concrete_init_temp']
        concrete_final_temp = data['concrete_final_temp']

        # The final wall temperature should be divided into intervals according to the number of time steps. i.e:
        # final temperature per segment =
        # initial temperature + current time period * (final temperature - initial temperature) / (time periods)
        # This is an approximation to initialize the concrete side of each time block
        T_concrete_end_time = {}
        T_concrete_end_delta = []
        num_time_periods = len(self.time_periods)

        for t in self.time_periods:
            for idx, i in enumerate(concrete_final_temp):
                T_concrete_end_delta.append(concrete_init_temp[idx] + (t / num_time_periods) * (
                            concrete_final_temp[idx] - concrete_init_temp[idx]))
            T_concrete_end_time[t] = T_concrete_end_delta
            T_concrete_end_delta = []

        for p in self.time_periods:
            # Fix the initial temperature for p > 1
            if p > 1:
                for idx, t in enumerate(self.period[p].tube_charge.temperature_wall_index):
                    self.period[p].concrete.init_temperature[t].fix(
                        self.period[p - 1].concrete.temperature[t].value
                    )

            # Fix the final temperature and heat transfer coefficient
            self.period[p].tube_charge.tube_heat_transfer_coefficient.fix(self.constant_htc.value)
            self.period[p].tube_discharge.tube_heat_transfer_coefficient.fix(self.constant_htc.value)
            for idx, t in enumerate(self.period[p].tube_charge.temperature_wall_index):
                self.period[p].concrete.temperature[t].fix(T_concrete_end_time[p][idx])
                self.period[p].tube_charge.temperature_wall[t].fix(T_concrete_end_time[p][idx])
                self.period[p].tube_discharge.temperature_wall[t].fix(T_concrete_end_time[p][idx])

            # **************************************************
            #            INITIALIZING CONCRETE SIDE
            # **************************************************
            self.period[p].concrete.initialize(outlvl=outlvl, optarg=solver.options)

            # **************************************************
            #               INITIALIZING TUBE SIDE
            # **************************************************
            # TODO: Need to print a proper error message
            # assert degrees_of_freedom(self.period[p].tube_charge) == 0
            self.period[p].tube_charge.initialize(outlvl=outlvl, optarg=solver.options)

            # assert degrees_of_freedom(self.period[p].tube_discharge) == 0
            self.period[p].tube_discharge.initialize(
                outlvl=outlvl, optarg=solver.options)

            # Unfix the final temperature
            for idx, t in enumerate(self.period[p].tube_charge.temperature_wall_index):
                self.period[p].concrete.temperature[t].unfix()
                self.period[p].tube_charge.temperature_wall[t].unfix()
                self.period[p].tube_discharge.temperature_wall[t].unfix()

            # **************************************************
            #               INITIALIZING TUBE + CONCRETE SIDES
            # **************************************************
            self.period[p].tube_charge.tube_inlet.fix()
            self.period[p].tube_discharge.tube_inlet.fix()
            assert degrees_of_freedom(self.period[p]) == 0
            with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
                res = solver.solve(self.period[p], tee=slc.tee)
            init_log.info_high("Initialization of tube and concrete: {}.".format(idaeslog.condition(res)))

            # Unfix the heat transfer coefficient and the initial concrete temperature for p > 1
            self.period[p].tube_charge.tube_heat_transfer_coefficient.unfix()
            self.period[p].tube_discharge.tube_heat_transfer_coefficient.unfix()
            if p > 1:
                self.period[p].concrete.init_temperature.unfix()

        # **************************************************
        #               INITIALIZING THE ENTIRE MODEL
        # **************************************************
        assert degrees_of_freedom(self) == 0
        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            res = solver.solve(self, tee=slc.tee)

        init_log.info_high("Initialization TES: {}.".format(idaeslog.condition(res)))
        init_log.info("Initialization Complete.")

        for p in self.time_periods:
            self.period[p].tube_charge.tube_inlet.unfix()
            self.period[p].tube_discharge.tube_inlet.unfix()
