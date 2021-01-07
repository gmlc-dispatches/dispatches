# Import Pyomo libraries
from pyomo.environ import Reference, Var, Reals, Constraint, Set, units as pyunits
from pyomo.network import Port
from pyomo.common.config import ConfigBlock, ConfigValue, In

# Import IDAES cores
from idaes.core import (Component,
                        ControlVolume0DBlock,
                        declare_process_block_class,
                        EnergyBalanceType,
                        MomentumBalanceType,
                        MaterialBalanceType,
                        UnitModelBlockData,
                        useDefault)
from idaes.core.util.config import is_physical_parameter_block
import idaes.logger as idaeslog

_log = idaeslog.getLogger(__name__)



def _make_pem_electrolyzer_config_block(config):
    """
    TODO: Declare configuration options for PEM_Electrolyzer block.
    """
    config.declare("property_package", ConfigValue(
        default=useDefault,
        domain=is_physical_parameter_block,
        description="Property package to use for control volume",
        doc="""Property parameter object used to define property calculations,
**default** - useDefault.
**Valid values:** {
**useDefault** - use default package from parent model or flowsheet,
**PropertyParameterObject** - a PropertyParameterBlock object.}"""))
    config.declare("property_package_args", ConfigBlock(
        implicit=True,
        description="Arguments to use for constructing property packages",
        doc="""A ConfigBlock with arguments to be passed to a property block(s)
and used when constructing these,
**default** - None.
**Valid values:** {
see property package for documentation.}"""))


@declare_process_block_class("PEM_Electrolyzer", doc="Simple 0D proton-exchange membrane electrolyzer model.")
class PEMElectrolyzerData(UnitModelBlockData):
    """
    Simple 0D proton-exchange membrane electrolyzer model.
    Unit model to convert electricity and water into H2 gas.
    """
    # TODO: change it so that dynamic is not an option, by creating a new config block
    CONFIG = UnitModelBlockData.CONFIG()    

    _make_pem_electrolyzer_config_block(CONFIG)

    def build(self):
        """Building model

        Args:
            None
        Returns:
            None
        """
        # Call UnitModel.build to setup dynamics
        super(PEMElectrolyzerData, self).build()

        self.electricity_to_mol = Var(self.flowsheet().config.time,
                                      domain=Reals,
                                      initialize=0.0,
                                      doc="Efficiency",
                                      units=pyunits.mol/pyunits.kW)    # use proper pyomo units for automatic unit convs

        self.electricity = Var(self.flowsheet().config.time,
                               domain=Reals,
                               initialize=0.0,
                               doc="Electricity into control volume",
                               units=pyunits.kW)

        # TODO: eventually two inlet & two oulet ports for electricity and materials
        self.inlet = Port(noruleinit=True, doc="A port for electricity flow")
        self.inlet.add(self.electricity, "electricity")

        self.outlet_state = self.config.property_package.build_state_block(self.flowsheet().config.time,
                                                                           default=self.config.property_package_args)
        self.add_outlet_port(name="outlet",
                             block=self.outlet_state,
                             doc="H2 out of electrolyzer")

        self.outlet.temperature.fix(300)
        self.outlet.pressure.fix(101325)

        @self.Constraint(self.flowsheet().config.time)
        def efficiency_curve(b, t):
            return b.outlet.flow_mol[t] == b.electricity[t] * b.electricity_to_mol[t]

    def _get_performance_contents(self, time_point=0):
        return {"vars": {"Efficiency": self.electricity_to_mol[time_point]}}

    def initialize(self, **kwargs):
        self.outlet_state.initialize(hold_state=False)

