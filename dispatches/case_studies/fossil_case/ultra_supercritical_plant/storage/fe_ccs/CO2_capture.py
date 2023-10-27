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
IDAES carbon capture system block, hybrid mass balances and surrogate models

The unit has 1 inlet stream for fluegas: inlet
The unit has 2 outlet streams: pureCO2 and exhaust_gas

The state variables are component flow mol, temperature, and pressure.
The state variables can be accessed thotough ports named:

* inlet - flue gas from NGCC plant
* pure_CO2 - stream for compression train
* exhaust_gas - exhaust gas to DAC system or stack

Surrogate models are used to compute solvent flow rate and specific reboiler duty.
The inlet variables to estimate such output variables are: lean loading and PZ molality.

Inlet Vars:

* flue gas (component molar flowrate [H20, CO2, O2, N2], temperature, pressure)
* solvent lean loading (0.1 to 0.6, mol CO2/mol PZ)
* solvent molality (PZ = 3, 5, 7 mol)

Output Vars:

* specific reboiler duty (GJ/t CO2 or MJ/kg CO2)
* lean solvent flowrate in kmol/s (0-100)
* L/G ratio for absorber columns (0 - 12) (only for Surrogate.PZ_ref1_90_capture)
* exhaust_gas (component molar flowrate [H20, CO2, O2, N2], temperature, pressure)
* pureCO2 (component molar flowrate [H20, CO2, O2, N2], temperature, pressure)

Surrogate models, the team explored different surrogate models. We developed
different models using commercial software for running simulations for 90% CO2 Capture,
95 % CO2 capture, and 97% CO2 Capture. Then, we provide data from a public source (ref 1)
and finally, we develop a simple correction factor to provide data to train data for 5 molal PZAS technology.
The models available are:

* PZ_90_capture: Conventional plant configuration to capture 90% capture
* PZ_95_capture: Conventional plant configuration to capture 95% capture
* PZ_97_capture: Conventional plant configuration to capture 97% capture
* PZ_90to97_capture: Conventional plant configuration to between 90 to 97% capture (using data obtained for previous surrogates)
* rel_PZAS_5mol_capture: using results for 5 molal PZ_90_capture applied a correction factor for SRD (using FEED studies).
* PZ_ref1_90_capture: Literature review, data obtained for reference 1 (see below).

Users can select any of these models while constructing the models, see examples -->
idaes/power_generation/flowsheets/ngcc/ngcc_withCCS.py
and/or idaes/power_generation/carbon_capture/piperazine_surrogates/tests/test_co2capture.py

Reference 1 (public data used to train surrogate models = PZ_ref1_90_capture):
Gaspar J. von Solms N., Thomsen K., Fosbol F. (2016) Multivariable Optimization
 of the Piperazine CO2 Post-Combustion Process. Energy Procedia 86(2016)229-238
"""
# Import IDAES cores
from idaes.core import declare_process_block_class, UnitModelBlockData, useDefault
import idaes.core.util.scaling as iscale
# Additional import for the unit operation
import pyomo.environ as pyo
from pyomo.environ import Var, units as pyunits, Constraint, exp, log
from pyomo.network import Port
import idaes.logger as idaeslog
# import gen_sm as ifit
from pyomo.common.config import ConfigBlock, ConfigValue, In
from enum import Enum
from idaes.power_generation.carbon_capture.piperazine_surrogates import L_G_ratio  # surrogate model for L_G_ratio

__author__ = "M. Zamarripa"
__version__ = "1.0.0"

class Surrogates(Enum):
    PZ_90_capture = 0
    PZ_95_capture = 1
    PZ_97_capture = 2
    PZ_90to97_capture = 3
    rel_PZAS_5mol_capture = 4
    PZ_ref1_90_capture = 5
    MEA = 6

class Technology(Enum):
    ngcc = 0  # natural gas combined cycle
    scpc = 1  # supercritical coal fired power plant

# ----------------------------------------------------------------------------
@declare_process_block_class("CO2Capture")
class CO2CaptureData(UnitModelBlockData):
    '''
    CO2Capture surrogate model based on total flow of CO2 rich stream flow only
    Assumptions: (toDo: update Assumptions)
    Fixed composition, temperature, and pressure of feed stream
    Fixed CO2 purity in the CO2 product stream
    '''
    CONFIG = ConfigBlock()
    CONFIG.declare("dynamic", ConfigValue(
            domain=In([useDefault, True, False]),
            default=useDefault,
            description="Dynamic model flag",
            doc="""Indicates whether this model will be dynamic or not,
    **default** = useDefault.
    **Valid values:** {
    **useDefault** - get flag from parent (default = False),
    **True** - set as a dynamic model,
    **False** - set as a steady-state model.}"""))
    CONFIG.declare("has_holdup", ConfigValue(
            default=False,
            domain=In([True, False]),
            description="Holdup construction flag",
            doc="""Indicates whether holdup terms should be constructed or not.
    Must be True if dynamic = True,
    **default** - False.
    **Valid values:** {
    **True** - construct holdup terms,
    **False** - do not construct holdup terms}"""))
    CONFIG.declare("CO2_capture_rate_surrogates", ConfigValue(
        default=Surrogates.PZ_90_capture,
        domain=In(Surrogates),
        description='surrogate models to be used in SRD eqn',
        doc='4 surrogates have been used, 90, 95, 97, and 90to97'))
    CONFIG.declare("flue_gas_source", ConfigValue(
        default=Technology.ngcc,
        domain=In(Technology),
        description='surrogate models to be used in SRD eqn',
        doc='4 surrogates have been used, 90, 95, 97, and 90to97'))


    def build(self):
        # Call UnitModel.build to setup dynamics
        super().build()

        self.component_list = ['CO2', 'H2O', 'O2', 'N2']

        self.make_vars()
        self.add_material_balances()

        # Add ports: 3 (1 for inlet and 2 for outlets)
        self.inlet = Port(noruleinit=True,
                          doc="A port for co2 rich inlet stream")
        self.pureCO2 = Port(noruleinit=True,
                            doc="A port for pure CO2 outlet stream")
        self.exhaust_gas = Port(noruleinit=True,
                                doc="A port for vent gas outlet stream")

        # Add state vars to the ports
        # self.inlet.add(self.inlet_flow_mol, "flow_mol")
        self.inlet.add(self.inlet_temperature, "temperature")
        self.inlet.add(self.inlet_pressure, "pressure")
        self.inlet.add(self.inlet_flow_mol_comp, "flow_mol_comp")

        # self.pureco2.add(self.pureco2_flow_mol, "flow_mol")
        self.pureCO2.add(self.pureCO2_temperature, "temperature")
        self.pureCO2.add(self.pureCO2_pressure, "pressure")
        self.pureCO2.add(self.pureCO2_flow_mol_comp, "flow_mol_comp")

        # self.exhaust_gas.add(self.exhaust_gas_flow_mol, "flow_mol")
        self.exhaust_gas.add(self.exhaust_gas_temperature, "temperature")
        self.exhaust_gas.add(self.exhaust_gas_pressure, "pressure")
        self.exhaust_gas.add(self.exhaust_gas_flow_mol_comp, "flow_mol_comp")

    def make_vars(self):
        '''
        This section builds port vars (Fc, T, P), CO2 capture rate

        '''

        # units declaration for vars
        flow_units = pyunits.mol/pyunits.s
        pressure_units = pyunits.Pa
        temperature_units = pyunits.K
        heat_duty_units = pyunits.J/pyunits.s

        # Component mole flows [mol/s]
        self.inlet_flow_mol_comp = Var(
            self.flowsheet().config.time,
            self.component_list,
            initialize=1400/len(self.component_list),
            units=flow_units,
            doc='Inlet stream: Component mole flow [mol/s]')

        self.pureCO2_flow_mol_comp = Var(
            self.flowsheet().config.time,
            self.component_list,
            initialize=1200/len(self.component_list),
            units=flow_units,
            doc='PureCO2 stream: Component mole flow [mol/s]')

        self.exhaust_gas_flow_mol_comp = Var(
            self.flowsheet().config.time,
            self.component_list,
            initialize=100/len(self.component_list),
            units=flow_units,
            doc='exhaust_gas stream: Component mole flow [mol/s]')

        # Temperature [K]
        self.inlet_temperature = Var(self.flowsheet().config.time,
                                     initialize=110,
                                     units=temperature_units,
                                     doc='Inlet temperature [K]')

        self.pureCO2_temperature = Var(self.flowsheet().config.time,
                                       initialize=110,
                                       units=temperature_units,
                                       doc='PureCO2 temperature [K]')

        self.exhaust_gas_temperature = Var(self.flowsheet().config.time,
                                           initialize=110,
                                           units=temperature_units,
                                           doc='exhaust_gas temperature [K]')

        # Pressue [Pa]
        self.inlet_pressure = Var(self.flowsheet().config.time,
                                  initialize=17,
                                  units=pressure_units,
                                  doc='Inlet pressure [Pa]')

        self.pureCO2_pressure = Var(self.flowsheet().config.time,
                                    initialize=17,
                                    units=pressure_units,
                                    doc='PureCO2 pressure [Pa]')

        self.exhaust_gas_pressure = Var(self.flowsheet().config.time,
                                        initialize=17,
                                        units=pressure_units,
                                        doc='exhaust_gas pressure [Pa]')
        # CO2 Capture rate
        self.CO2_capture_rate = Var(self.flowsheet().config.time,
                                    initialize=0.9,
                                    doc='CO2 capture rate')
        # lean loading
        self.lean_loading = Var(self.flowsheet().config.time,
                                initialize=0.1,
                                bounds=(0.1, 6.0),
                                doc='lean loading')
        # Pz molality
        self.Pz_mol = Var(self.flowsheet().config.time,
                        # domain=pyo.Integers,
                          initialize=3,
                          bounds=(3, 7),
                          doc='Pz molality')

        self.SRD = Var(self.flowsheet().config.time,
                       initialize=8,
                       bounds=(0, 100),
                       doc='Specific reboiler duty GJ/ton CO2 or MJ/kg CO2')

    def add_material_balances(self):
        ''' This section is for material balance constraints'''

        # pureCO2 mass balance
        @self.Constraint(self.flowsheet().config.time,
                         self.component_list,
                         doc="pureCO2 mass balances")
        def pureCO2_eqn(b, t, c):
            if c == "CO2":
                return b.pureCO2_flow_mol_comp[t, c] == \
                    b.inlet_flow_mol_comp[t, c] * b.CO2_capture_rate[t]
            else:
                return b.pureCO2_flow_mol_comp[t, c] == 0.0

        # water drop in flue gas
        @self.Expression(self.flowsheet().config.time, doc="water drop")
        def water_drop(b, t):
            return b.inlet_flow_mol_comp[0, 'H2O']*0.5

        # Overall mass balances
        @self.Constraint(self.flowsheet().config.time,
                         self.component_list,
                         doc="Inlet component mole flow eqn")
        def flow_mol_comp_inlet_eqn(b, t, c):
            if c == "H2O":
                return b.inlet_flow_mol_comp[t, c] == \
                    b.exhaust_gas_flow_mol_comp[t, c] + b.water_drop[t]
            elif c == "CO2":
                return b.inlet_flow_mol_comp[t, c] == \
                    b.exhaust_gas_flow_mol_comp[t, c] \
                    + b.pureCO2_flow_mol_comp[t, c]
            else:
                return b.inlet_flow_mol_comp[t, c] == \
                    b.exhaust_gas_flow_mol_comp[t, c]

        # Pressure equations
        @self.Constraint(self.flowsheet().config.time,
                         doc="Pressure drop")
        def exh_pressure_eqn(b, t):
            return b.inlet_pressure[t] == b.exhaust_gas_pressure[t]

        @self.Constraint(self.flowsheet().config.time,
                         doc="Pressure drop")
        def pureCO2_pressure_eqn(b, t):
            return b.inlet_pressure[t] == b.pureCO2_pressure[t]

        # Temperature equations
        @self.Constraint(self.flowsheet().config.time,
                         doc="Temperature")
        def pureCO2_temp_eqn(b, t):
            return b.inlet_temperature[t] == b.pureCO2_temperature[t]

        @self.Constraint(self.flowsheet().config.time,
                         doc="Temperature")
        def exh_temp_eqn(b, t):
            return b.inlet_temperature[t] == b.exhaust_gas_temperature[t]


        @self.Constraint(self.flowsheet().config.time,
                         doc="Specific reboiler duty in GJ/t CO2 or MJ/ kg CO2")
        def SRD_eqn(b, t):
            CO2leanloading = b.lean_loading[t]
            molality = b.Pz_mol[t]
            Capture = b.CO2_capture_rate[t]
            if b.config.flue_gas_source == Technology.ngcc:
                if b.config.CO2_capture_rate_surrogates == Surrogates.PZ_90_capture:
                    return b.SRD[t] == - 3.6233485374100991016633 * log(molality/4.) + 0.33347583941977870791717E-001 * (molality/4.)**3 + 6.8390136776297296705707 * (molality*CO2leanloading/4.)**0.5 - 5.7867174977020807702388 * (molality*CO2leanloading/4.)**2 + 5.1342721429236659602680 * (molality*CO2leanloading/4.)**3 + 0.19963960036749664461730 * (molality/CO2leanloading/4.)
                elif b.config.CO2_capture_rate_surrogates == Surrogates.PZ_95_capture:
                    return b.SRD[t] == - 2.1195314593911489531308 * log(molality/4.) + 2.3838807672964539285942 * exp(CO2leanloading) + 0.17427301681432125213256 * (molality/4.)**2 + 0.12496410061822192660852 * (molality/CO2leanloading/4.) + 0.93458997890860340262975 * (CO2leanloading/molality/0.25)
                elif b.config.CO2_capture_rate_surrogates == Surrogates.PZ_97_capture:
                    return b.SRD[t] == - 912.90813190561812007218 * CO2leanloading + 963.87626280230563224904 * exp(CO2leanloading) - 840.95461087585590576055 * (molality/2.)**2 - 614.38277546147764951456 * CO2leanloading**2 + 274.40532024830611135258 * (molality/2.)**3 - 0.61984641382653959951199 * (molality*CO2leanloading/2.)**3 + 0.26430328623580695568407 * (molality/CO2leanloading/2.) - 11.869291574687059309667 * (CO2leanloading/molality/0.5) - 0.23630953358941128236714E-002 * (molality/CO2leanloading/2.)**2
                elif b.config.CO2_capture_rate_surrogates == Surrogates.PZ_90to97_capture:
                    return b.SRD[t] ==  - 0.10526756950236890175709 * Capture + 0.52057178809591142520929 * lean_loading + 0.98788205977359255793857 * molality + 2.0990364304677582296677 * Capture**0.5 + 3.5938248110847506033849 * lean_loading**0.5 - 6.3111812164050720141972 * molality**0.5 + 2.4029805693688151002618 * lean_loading**3 - 0.10937466124959506139080E-001 * Capture/molality - 41.496993880686794398116 * lean_loading/Capture + 0.36946342949766633467767E-001 * molality/lean_loading
                elif b.config.CO2_capture_rate_surrogates == Surrogates.rel_PZAS_5mol_capture:
                    return (- 22.312295301354438947783 * CO2leanloading + 117.51384451486856619340 * CO2leanloading**2 - 269.05987870514132964672 * CO2leanloading**3 + 234.98153128275288281657 * CO2leanloading**4 + 3.9380879255782788028739)
                elif b.config.CO2_capture_rate_surrogates == Surrogates.PZ_ref1_90_capture:
                    return b.SRD[t] == (- 15565.227507172588957474 * CO2leanloading
                                        + 8.9237367058697429911263 * log(CO2leanloading)
                                        + 15285.362653727905126289 * exp(CO2leanloading)
                                        - 6741.0217492965430210461 * CO2leanloading**2
                                        - 3865.5900513667870654899 * CO2leanloading**3
                                        + 0.17312045830047528404555E-002 * molality**3
                                        - 1.2960483459315448317994 * CO2leanloading*molality
                                        - 15235.956076189686427824)
                else:
                    raise Exception('solvent is not supported for this technology')
            elif b.config.flue_gas_source == Technology.scpc:
                if b.config.CO2_capture_rate_surrogates == Surrogates.MEA:
                    return b.SRD[t] == 1135585.36238969 * CO2leanloading - 440.509698856962 * log(CO2leanloading) - 1127184.48012032 * exp(CO2leanloading) + 533953.122356288 * CO2leanloading**2 + 248202.546339126 * CO2leanloading**3 + 1125580.12992624
                else:
                    raise Exception('solvent not supported for this technology')
            else:
                raise Exception('Flue gas source is not supported')

        @self.Expression(self.flowsheet().config.time,
                         doc="Reboiler duty in MW")
        def reboiler_duty(b, t):   # mol/s /1000 = kgmol/s * 44 kg/kgmol = kg / s
            return b.SRD[t] * (b.pureCO2_flow_mol_comp[t, "CO2"]
                               * 44.01 / 1000)

        @self.Expression(self.flowsheet().config.time,
                         doc="Lean loading flowrate - kmol.hr")
        def LL_flowrate(b, t):
            CO2leanloading = b.lean_loading[t]
            molality = b.Pz_mol[t]
            Capture = b.CO2_capture_rate[t]
            if b.config.flue_gas_source == Technology.ngcc:
                if b.config.CO2_capture_rate_surrogates == Surrogates.PZ_90_capture:
                    return - 33769665655.708782196045 * CO2leanloading - 85501806.085839942097664 * log(CO2leanloading) + 32051499965.332691192627 * exp(CO2leanloading) + 1399851299.6695346832275 * CO2leanloading**0.5 + 207401.67766238830517977 * (molality/4.)**2 - 13931345844.245344161987 * CO2leanloading**2 - 7882628979.2478094100952 * CO2leanloading**3 - 1216008.0524285589344800 * (molality*CO2leanloading/4.) - 721743.35243441292550415 * (molality/CO2leanloading/4.)**0.5 + 3619336.4956937171518803 * (CO2leanloading/molality/0.25) - 32534484771.443439483643
                elif b.config.CO2_capture_rate_surrogates == Surrogates.PZ_95_capture:
                    return - 8128346023.5960435867310 * CO2leanloading + 8097036304.4302654266357 * exp(CO2leanloading) + 143943.79958957055350766 * (molality/4.)**2 - 3906104299.9280681610107 * CO2leanloading**2 - 1694548695.5476403236389 * CO2leanloading**3 - 326827.12907762406393886 * (molality*CO2leanloading/4.)**2 + 14089575.472042093053460 * (CO2leanloading/molality/0.25) - 4257.3482804741406653193 * (molality/CO2leanloading/4.)**2 - 22772347.982399865984917 * (CO2leanloading/molality/0.25)**2 + 131.22409336241202026940 * (molality/CO2leanloading/4.)**3 + 22174561.087608262896538 * (CO2leanloading/molality/0.25)**3 - 8094525387.7033281326294
                elif b.config.CO2_capture_rate_surrogates == Surrogates.PZ_97_capture:
                    return - 11073017307.372232437134 * CO2leanloading + 868882.20868137793149799 * log(CO2leanloading) - 73653243483.647430419922 * exp(molality/2.) + 11052450134.832271575928 * exp(CO2leanloading) + 141797181109.30328369141 * (molality/2.)**2 - 5395322021.3101339340210 * CO2leanloading**2 - 920354.31067807460203767 * (molality*CO2leanloading/2.) - 117144381.90464735031128 * (molality*CO2leanloading/2.)**3 - 6153045658.4519367218018 * (CO2leanloading/molality/0.5)**3
                elif b.config.CO2_capture_rate_surrogates == Surrogates.PZ_90to97_capture:
                    return 21493.893918343124823878 * Capture - 18837259.090658042579889 * lean_loading - 303798.35064064717153087 * molality - 547220.78199295536614954 * Capture**0.5 + 7997911.0828296802937984 * lean_loading**0.5 + 1594229.8231671806424856 * molality**0.5 + 16593654.797202151268721 * lean_loading**2 + 58165.759382280644786078 * Capture*lean_loading + 60595.207259644026635215 * Capture/molality + 60446137.647265411913395 * (lean_loading/molality)**2
                # No data for PZAS, using same as PZ 95 capture
                elif b.config.CO2_capture_rate_surrogates == Surrogates.rel_PZAS_5mol_capture:
                    return - 8128346023.5960435867310 * CO2leanloading + 8097036304.4302654266357 * exp(CO2leanloading) + 143943.79958957055350766 * (molality/4.)**2 - 3906104299.9280681610107 * CO2leanloading**2 - 1694548695.5476403236389 * CO2leanloading**3 - 326827.12907762406393886 * (molality*CO2leanloading/4.)**2 + 14089575.472042093053460 * (CO2leanloading/molality/0.25) - 4257.3482804741406653193 * (molality/CO2leanloading/4.)**2 - 22772347.982399865984917 * (CO2leanloading/molality/0.25)**2 + 131.22409336241202026940 * (molality/CO2leanloading/4.)**3 + 22174561.087608262896538 * (CO2leanloading/molality/0.25)**3 - 8094525387.7033281326294
                elif b.config.CO2_capture_rate_surrogates == Surrogates.PZ_ref1_90_capture:
                    A2 = b.lean_loading[t]
                    B2 = b.Pz_mol[t]
                    return (291019.778242091 * A2 - 4.98043194086976 * B2 - 287661.899241911 * exp(A2) + 130101.667827899 * A2**2 + 71279.9986297754 * A2**3 + 0.0148339624265494 * (B2/A2)**2 + 287396.698159522)
                else:
                    raise Exception('CO2 surrogate model is not supported')
            elif b.config.flue_gas_source == Technology.scpc:
                if b.config.CO2_capture_rate_surrogates == Surrogates.MEA:
                    return 1  # No data available for solvent flowrate
                else:
                    raise Exception('solvent not supported for this technology')
            else:
                raise Exception('Flue gas source is not supported')

        if self.config.CO2_capture_rate_surrogates == Surrogates.PZ_ref1_90_capture:
            # This surrogate model is only valid for reference 1
            @self.Expression(self.flowsheet().config.time,
                             doc="L/G ratio")
            def L_Gratio(b, t):
                x1 = b.lean_loading[t]
                x2 = b.Pz_mol[t]
                return L_G_ratio.f(b.lean_loading[t], b.Pz_mol[t])

        @self.Expression(self.flowsheet().config.time,
                         doc="lean loading cost $/year")
        def solvent_cost(b, t):
            basis = 1 # 1 kg H2O basis for estimating mol fractions
            PZ_MW = 86.1356
            H2O_MW = 18.0153 # kg/kgmol
            mol_H2O = basis / H2O_MW * 1000  # mol fraction
            mol_CO2 = b.lean_loading[t] * b.Pz_mol[t]
            PZ_mol_frac = b.Pz_mol[t]/(b.Pz_mol[t] + mol_H2O + mol_CO2)
            H2O_mol_frac = mol_H2O / (b.Pz_mol[t] + mol_H2O + mol_CO2)
            CO2_mol_frac = mol_CO2 / (b.Pz_mol[t] + mol_H2O + mol_CO2)
            PZ_cost = 9 * PZ_MW  # PZ $9/kg  - converting to $/kmol
            PZ_makeup = 0.02  # 2 % of total flowrate
            return b.LL_flowrate[t] * PZ_makeup * PZ_mol_frac * 24 * 365 * PZ_cost

    def initialize(blk,
                   outlvl=idaeslog.NOTSET,
                   solver='ipopt',
                   optarg={'tol': 1e-6}):
        '''
        CO2 pure pyomo block initialization routine

        Keyword Arguments:
            outlvl : sets output level of initialisation routine

            optarg : solver options dictionary object (default={'tol': 1e-6})
            solver : str indicating whcih solver to use during
                     initialization (default = 'ipopt')

        Returns:
            None
        '''
        iscale.calculate_scaling_factors(blk)  # remove to solve using baron
        init_log = idaeslog.getInitLogger(blk.name, outlvl, tag="unit")
        solve_log = idaeslog.getSolveLogger(blk.name, outlvl, tag="unit")
        opt = pyo.SolverFactory(solver)
        opt.options = optarg

        init_log.info_low("Starting initialization...")

        blk.inlet.flow_mol_comp[0, 'CO2'].fix()
        blk.inlet.flow_mol_comp[0, 'O2'].fix()
        blk.inlet.flow_mol_comp[0, 'H2O'].fix()
        blk.inlet.flow_mol_comp[0, 'N2'].fix()
        blk.CO2_capture_rate.fix()

        # solve model
        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            res = opt.solve(blk, tee=slc.tee)
        init_log.info_high(
                "Initialization Step 1 {}.".format(idaeslog.condition(res))
            )
        init_log.info_high("Initialization Step 1 Complete.")

        # ToDo: release state
        init_log.info("Initialization Complete.")

    def calculate_scaling_factors(self):
        super().calculate_scaling_factors()

        iscale.set_scaling_factor(self.inlet_flow_mol_comp[0.0, 'CO2'], 1e-3)
        iscale.set_scaling_factor(self.inlet_flow_mol_comp[0.0, 'H2O'], 1e-3)
        iscale.set_scaling_factor(self.inlet_flow_mol_comp[0.0, 'O2'], 1e-3)
        iscale.set_scaling_factor(self.inlet_flow_mol_comp[0.0, 'N2'], 1e-3)
        iscale.set_scaling_factor(self.pureCO2_flow_mol_comp[0.0, 'CO2'], 1e-3)
        iscale.set_scaling_factor(self.pureCO2_flow_mol_comp[0.0, 'H2O'], 1e-3)
        iscale.set_scaling_factor(self.pureCO2_flow_mol_comp[0.0, 'O2'], 1e-3)
        iscale.set_scaling_factor(self.pureCO2_flow_mol_comp[0.0, 'N2'], 1e-3)
        iscale.set_scaling_factor(self.exhaust_gas_flow_mol_comp[0.0, 'CO2'],
                                  1e-3)
        iscale.set_scaling_factor(self.exhaust_gas_flow_mol_comp[0.0, 'H2O'],
                                  1e-3)
        iscale.set_scaling_factor(self.exhaust_gas_flow_mol_comp[0.0, 'O2'],
                                  1e-3)
        iscale.set_scaling_factor(self.exhaust_gas_flow_mol_comp[0.0, 'N2'],
                                  1e-3)
        iscale.set_scaling_factor(self.inlet_temperature[0.0], 1e-2)
        iscale.set_scaling_factor(self.pureCO2_temperature[0.0], 1e-2)
        iscale.set_scaling_factor(self.exhaust_gas_temperature[0.0], 1e-2)
        iscale.set_scaling_factor(self.inlet_pressure[0.0], 1e-5)
        iscale.set_scaling_factor(self.pureCO2_pressure[0.0], 1e-5)
        iscale.set_scaling_factor(self.exhaust_gas_pressure[0.0], 1e-5)

        for t, c in self.exh_pressure_eqn.items():
            sf = iscale.get_scaling_factor(
                self.inlet_pressure[t], default=1, warning=True)
            iscale.constraint_scaling_transform(c, sf)

        for t, c in self.pureCO2_pressure_eqn.items():
            sf = iscale.get_scaling_factor(
                self.inlet_pressure[t], default=1, warning=True)
            iscale.constraint_scaling_transform(c, sf)

        for t, c in self.pureCO2_temp_eqn.items():
            sf = iscale.get_scaling_factor(
                self.inlet_temperature[t], default=1, warning=True)
            iscale.constraint_scaling_transform(c, sf)

        for t, c in self.exh_temp_eqn.items():
            sf = iscale.get_scaling_factor(
                self.inlet_temperature[t], default=1, warning=True)
            iscale.constraint_scaling_transform(c, sf)

        for t, c in self.flow_mol_comp_inlet_eqn.items():
            sf = iscale.get_scaling_factor(
                self.inlet_flow_mol_comp[t], default=1, warning=True)
            iscale.constraint_scaling_transform(c, sf)

        for t, c in self.pureCO2_eqn.items():
            sf = iscale.get_scaling_factor(
                self.inlet_flow_mol_comp[t], default=1, warning=True)
            iscale.constraint_scaling_transform(c, sf)
