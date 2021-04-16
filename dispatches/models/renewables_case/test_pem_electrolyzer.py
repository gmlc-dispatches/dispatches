# Import objects from pyomo package
from pyomo.environ import ConcreteModel, SolverFactory, Var, TerminationCondition, SolverStatus

# Import the main FlowsheetBlock from IDAES. The flowsheet block will contain the unit model
from idaes.core import FlowsheetBlock
from idaes.core.util.testing import initialization_tester

# Import the H2 property package to create a properties block for the flowsheet
from idaes.generic_models.properties.core.generic.generic_property \
    import GenericParameterBlock

from dispatches.models.nuclear_case.h2_ideal_vap import configuration
from dispatches.models.renewables_case.pem_electrolyzer import PEM_Electrolyzer


def test_pem():
    # Create the ConcreteModel and the FlowsheetBlock, and attach the flowsheet block to it.
    m = ConcreteModel()
    m.fs = FlowsheetBlock(default={"dynamic": False})   # dynamic or ss flowsheet needs to be specified here

    # Add properties parameter block to the flowsheet with specifications
    m.fs.properties = GenericParameterBlock(default=configuration)

    m.fs.unit = PEM_Electrolyzer(default={"property_package": m.fs.properties})

    assert hasattr(m.fs.unit, "efficiency_curve")
    assert hasattr(m.fs.unit, "electricity_in")
    assert hasattr(m.fs.unit, "outlet")
    assert hasattr(m.fs.unit, "outlet_state")
    assert isinstance(m.fs.unit.electricity, Var)
    assert isinstance(m.fs.unit.electricity_to_mol, Var)

    m.fs.unit.electricity_in.electricity.fix(1)
    m.fs.unit.electricity_to_mol.fix(5)
    initialization_tester(m)

    solver = SolverFactory('ipopt')
    results = solver.solve(m.fs)

    assert results.solver.termination_condition == TerminationCondition.optimal
    assert results.solver.status == SolverStatus.ok

    assert m.fs.unit.electricity_in.electricity[0].value == 1
    assert m.fs.unit.outlet.flow_mol[0].value == 5.0
    assert m.fs.unit.outlet.temperature[0].value == 300
    assert m.fs.unit.outlet.pressure[0].value == 101325
