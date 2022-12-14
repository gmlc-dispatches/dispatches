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

import pandas as pd
import pytest

from pyomo.environ import (
    ConcreteModel, 
    units as pyunits, 
    value,
    assert_optimal_termination,
)
from pyomo.common import unittest as pyo_unittest
from idaes.core import FlowsheetBlock
from idaes.models.properties import iapws95
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.solvers import get_solver
from dispatches.unit_models import ConcreteTES


def _get_data():
    data = {
        "num_tubes": 10000,
        "num_segments": 20,
        "num_time_periods": 2,
        "tube_length": 64.9,
        "tube_diameter": 0.0105664,
        "face_area": 0.00847,
        "therm_cond_concrete": 1,
        "dens_mass_concrete": 2240,
        "cp_mass_concrete": 900,
        "init_temperature_concrete": [
            750, 732.631579, 715.2631579, 697.8947368, 680.5263158, 663.1578947,
            645.7894737, 628.4210526, 611.0526316, 593.6842105, 576.3157895, 558.9473684,
            541.5789474, 524.2105263, 506.8421053, 489.4736842, 472.1052632, 454.7368421,
            437.3684211, 420
        ],
        "flow_mol_charge": 0.00958 * 1000 / 18.01528,
        "inlet_pressure_charge": 19600000,
        "inlet_temperature_charge": 865,
        "flow_mol_discharge": 3 / 18.01528,
        "inlet_pressure_discharge": 8.5e5, #15 MPa
        "inlet_temperature_discharge": 355
    }

    return data


def _get_charge_results():
    exp_conc_temp = {
        1: [768.8794598487062, 750.9141725711494, 733.1558692075599, 715.5779731910243,
            698.1627726680688, 680.9003463323493, 663.7878525182592, 646.8291235216258,
            630.034517306009, 613.4209816138464, 597.0123062127739, 580.8395649489671,
            564.9418055323642, 549.3670467067806, 534.1731714688473, 519.4256478712385,
            505.4539745384297, 491.5937379825899, 477.7335015065516, 463.87326495071187],
        2: [784.6536656409681, 766.7404977929137, 749.063068065682, 731.6061482700076,
            714.3620773742523, 697.3306181729016, 680.5189998846788, 663.9421290510368,
            647.6229432955979, 631.5928719729783, 615.8923779344503, 600.5715793487628,
            585.6910142546371, 571.3226417304624, 557.5507863291356, 544.4703166829731,
            532.390904452725, 521.0060428032424, 509.9453853507483, 498.88472783457166],
    }
    exp_fluid_temp = {
        1: [843.4689736714969, 823.1455699108972, 803.8469084691471, 785.4414129181083,
            767.841394508302, 750.9977353406474, 734.896366025036, 719.5562603922092,
            705.0286981563756, 691.3975854791795, 678.7807006374081, 667.3318857141337,
            657.2444584467377, 648.7561522064175, 642.1535350190497, 637.7607287892795,
            637.2090239563571, 637.2090239563571, 637.2090239563571, 637.2090239563571],
        2: [846.9748522858338, 829.2675993812405, 811.9096875462226, 794.9307240888364,
            778.362757053882, 762.2438094603676, 746.6208988669331, 731.5526842636623,
            717.1118033575298, 703.3868998737142, 690.4843091626235, 678.5293902512656,
            667.6675857884796, 658.0654390163991, 649.9117405507793, 643.4175156823585,
            638.8141031331337, 637.2090239563571, 637.2090239563571, 637.2090239563571],
    }
    exp_heat_rate = {
        1: [581.1733601639454, 562.799805895126, 550.797916698378, 544.3495732558932,
            542.9095419858858, 546.1724178208048, 554.0507185658779, 566.6624213505045,
            584.3263730220131, 607.5642883293052, 637.1084902874657, 673.9155426951835,
            719.1874594203609, 774.4024252814344, 841.3422677079749, 922.0223200143666,
            1026.585653456652, 1134.579389291451, 1242.5731245044672, 1350.5668603392658],
        2: [485.58318181093017, 487.1875933892568, 489.6771562987536, 493.40121182477066,
            498.6691581145785, 505.77910440776037, 515.0410664427249, 526.7959467978917,
            541.4309897782801, 559.3919881808116, 581.1921956747592, 607.4178606502138,
            638.7305278065372, 675.8671612729662, 719.6417212065214, 770.9592564697472,
            829.209426341959, 905.4098030223171, 991.5902743491622, 1077.7707461721984],
    }

    return exp_conc_temp, exp_heat_rate, exp_fluid_temp


def _get_discharge_results():
    exp_conc_temp = {
        1: [746.1063169450176, 728.4696928862526, 710.5578357626713, 692.1005335939977,
            672.5608778723413, 650.8774474530392, 625.0196314618721, 592.1687287491123,
            577.7317976976101, 563.8715611417704, 550.0113246657321, 536.1510881098923,
            522.290851633854, 508.4306150780142, 494.57037860197596, 480.7101420461362,
            464.3881408074005, 446.8174177132283, 429.1096925824503, 411.20460039012323],
        2: [741.6634944375896, 723.2734354446304, 703.9106073179977, 682.6706352362868,
            657.8170956334103, 626.15173157226, 588.8776537858718, 562.6621860671783,
            551.1413184902073, 540.0806609740306, 529.0200035215366, 517.9593460053599,
            506.89868855286596, 495.83803103668924, 484.7773735841953, 473.44693663450266,
            456.95681164006436, 439.3490009451654, 421.58088889572184, 403.6002459881725],
    }
    exp_fluid_temp = {
        1: [730.7230417677312, 712.0267933383869, 691.9679135183114, 669.2086286565905,
            641.0907962507835, 602.35950271216, 542.9615404396385, 448.94200337801783,
            446.0868872570418, 446.0868872570418, 446.0868872570418, 446.0868872570418,
            446.0868872570418, 446.0868872570418, 446.0868872570418, 446.0868872570418,
            433.8991113548745, 415.5291277145009, 396.4808700496551, 376.4554822461086],
        2: [724.1106632927483, 702.7439118578757, 677.6485456948188, 645.4147211613656,
            599.5669360468476, 528.4646610349995, 446.0868872570418, 446.0868872570418,
            446.0868872570418, 446.0868872570418, 446.0868872570418, 446.0868872570418,
            446.0868872570418, 446.0868872570418, 446.0868872570418, 444.7512554661709,
            427.5969022946328, 409.8425648384156, 391.83587441116816, 373.5567435615619],
    }
    exp_heat_rate = {
        1: [-119.86067835688637, -128.11687181357422, -144.8456629119361, -178.36508955714856,
            -245.20300700827698, -378.0335268077842, -639.3656969182383, -1115.968625957053,
            -1025.7274920841446, -917.733756249346, -809.7400210363302, -701.7462852015307,
            -593.7525499885147, -485.75881415371583, -377.7650789407, -269.77134310590094,
            -237.55901851243217, -243.7865552461684, -254.23147916782654, -270.7520228978944],
        2: [-136.76504020483577, -159.9583050481918, -204.62407939245335, -290.2840313480109,
            -453.8632743728761, -761.1408110214691, -1112.5717990481025, -908.3107617856863,
            -818.544506737832, -732.3640349147958, -646.1835635879512, -560.003091764914,
            -473.82262043806975, -387.64214861503314, -301.46167728818835, -223.58592504590587,
            -228.76134048724117, -229.90302174864954, -231.7619344883195, -234.0876399576691],
    }

    return exp_conc_temp, exp_heat_rate, exp_fluid_temp


def _get_combined_results():
    exp_conc_temp = {
        1: [765.6955354841449, 747.5945530427604, 729.647450335955, 711.7483058524213,
            693.7247605780229, 675.2594659952538, 655.7351805481906, 633.9399187030289,
            607.6602996332637, 583.7078042836023, 569.918113445112, 556.5135719077973,
            543.4847736612935, 530.8394836200084, 518.5979406151248, 507.0088118612352,
            495.47770245750166, 483.64954991662637, 468.15745487706835, 451.77760745990577],
        2: [778.777670818477, 760.5255613795055, 742.4336515266298, 724.3518312101746,
            706.0253151971591, 686.9897863434737, 666.3750612481672, 642.5521353237004,
            612.6541872856708, 579.6760329417091, 566.1488472205821, 555.2224540652642,
            544.9926995318799, 535.4321187480766, 526.5379435762707, 518.6505998781274,
            510.9949538017873, 503.1420971147642, 490.91749609805186, 474.31213027291903],
    }
    exp_ch_fluid_temp = {
        1: [842.7627356797586, 821.88773432906, 802.1459839962807, 783.3568033560631,
            765.3691390138662, 748.0374738844148, 731.1872574255896, 714.5669752940619,
            697.8186750308802, 681.9136482443657, 668.4732167804137, 657.3247432678278,
            648.4122533589966, 641.7819555204223, 637.5542867815433, 637.2090239563571,
            637.2090239563571, 637.2090239563571, 637.2090239563571, 637.2090239563571],
        2: [845.6675508303674, 826.9126764153997, 808.6935241637494, 790.9589510331716,
            773.6362952280797, 756.6082192733007, 739.6737206245535, 722.493139268127,
            704.5608788442003, 686.3698150058249, 671.3190816046152, 659.2570353085273,
            649.856068774408, 642.9220654011634, 638.3585923186711, 637.2090239563571,
            637.2090239563571, 637.2090239563571, 637.2090239563571, 637.2090239563571],
    }
    exp_di_fluid_temp = {
        1: [750.6387090138653, 732.4175206365848, 713.9787935937404, 694.87288803486,
            674.2251779729404, 650.2926945836386, 619.5768887884733, 575.1169573149741,
            504.0994017758527, 446.0868872570418, 446.0868872570418, 446.0868872570418,
            446.0868872570418, 446.0868872570418, 446.0868872570418, 446.0868872570418,
            446.0868872570418, 444.3192347948272, 420.74808737074466, 391.8940779199924],
        2: [763.5730679369417, 745.2266465784485, 726.6898708802531, 707.5390824349447,
            687.0117206923077, 663.71584095563, 635.1127490396054, 596.6365248938914,
            540.4774909473294, 457.0534131014283, 446.0868872570418, 446.0868872570418,
            446.0868872570418, 446.0868872570418, 446.0868872570418, 446.0868872570418,
            446.0868872570418, 446.0868872570418, 434.5469947343781, 400.4452784614916],
    }
    exp_heat_rate = {
        1: [483.1614447654906, 460.6107363327419, 442.796967475617, 426.4595142540948,
            406.2925826295586, 372.52712256003264, 306.16235392796943, 169.88928619226263,
            -104.4274033102113, -307.1073838768562, -196.94201634240838, -74.92045305658068,
            58.66775114546273, 204.0616326381136, 361.88420465288016, 539.7902877227838,
            719.48239872132, 890.0305245956872, 947.7901571239249, 978.2217845978254],
        2: [402.71218622254713, 398.05998824109133, 393.6023365745825, 387.9789591850555,
            378.65249944496645, 361.098768037143, 327.5313630634786, 265.11303352984964,
            153.72868135470637, -124.11150090935013, -116.03070927967546, -39.744955678063576,
            46.419036990981446, 141.37677723704962, 244.42003307404175, 358.3734447019817,
            477.6732584796396, 600.046253657646, 700.6307246007051, 693.6885084573833],
    }

    return exp_conc_temp, exp_heat_rate, exp_ch_fluid_temp, exp_di_fluid_temp


@pytest.fixture(scope="module")
def build_concrete_tes_charge():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic="False")
    m.fs.props_water = iapws95.Iapws95ParameterBlock(
        phase_presentation=iapws95.PhaseType.LG,
    )

    data = _get_data()
    inlet_enthalpy_charge = iapws95.htpx(
        T=data["inlet_temperature_charge"] * pyunits.K,
        P=data["inlet_pressure_charge"] * pyunits.Pa,
    )

    m.fs.tes = ConcreteTES(
        model_data=data, 
        property_package=m.fs.props_water,
        operating_mode="charge",
    )

    m.fs.tes.inlet_charge.flow_mol.fix(data["flow_mol_charge"] * data["num_tubes"])
    m.fs.tes.inlet_charge.pressure.fix(data["inlet_pressure_charge"])
    m.fs.tes.inlet_charge.enth_mol.fix(inlet_enthalpy_charge)

    return m


@pytest.fixture(scope="module")
def build_concrete_tes_discharge():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic="False")
    m.fs.props_water = iapws95.Iapws95ParameterBlock(
        phase_presentation=iapws95.PhaseType.LG,
    )

    data = _get_data()
    inlet_enthalpy_discharge = iapws95.htpx(
        T=data["inlet_temperature_discharge"] * pyunits.K,
        P=data["inlet_pressure_discharge"] * pyunits.Pa,
    )

    m.fs.tes = ConcreteTES(
        model_data=data, 
        property_package=m.fs.props_water,
        operating_mode="discharge",
    )

    m.fs.tes.inlet_discharge.flow_mol.fix(data["flow_mol_discharge"] * data["num_tubes"])
    m.fs.tes.inlet_discharge.pressure.fix(data["inlet_pressure_discharge"])
    m.fs.tes.inlet_discharge.enth_mol.fix(inlet_enthalpy_discharge)

    return m


@pytest.fixture(scope="module")
def build_concrete_tes_charge_discharge():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic="False")
    m.fs.props_water = iapws95.Iapws95ParameterBlock(
        phase_presentation=iapws95.PhaseType.LG,
    )

    data = _get_data()
    inlet_enthalpy_charge = iapws95.htpx(
        T=data["inlet_temperature_charge"] * pyunits.K,
        P=data["inlet_pressure_charge"] * pyunits.Pa,
    )
    inlet_enthalpy_discharge = iapws95.htpx(
        T=data["inlet_temperature_discharge"] * pyunits.K,
        P=data["inlet_pressure_discharge"] * pyunits.Pa,
    )

    m.fs.tes = ConcreteTES(
        model_data=data, 
        property_package=m.fs.props_water,
        operating_mode="combined",
    )

    m.fs.tes.inlet_charge.flow_mol.fix(data["flow_mol_charge"] * data["num_tubes"])
    m.fs.tes.inlet_charge.pressure.fix(data["inlet_pressure_charge"])
    m.fs.tes.inlet_charge.enth_mol.fix(inlet_enthalpy_charge)

    m.fs.tes.inlet_discharge.flow_mol.fix((0.01 / 3) * data["flow_mol_discharge"] * data["num_tubes"])
    m.fs.tes.inlet_discharge.pressure.fix(data["inlet_pressure_discharge"])
    m.fs.tes.inlet_discharge.enth_mol.fix(inlet_enthalpy_discharge)

    return m


@pytest.fixture(scope="module")
def build_concrete_tes_discharge_charge():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic="False")
    m.fs.props_water = iapws95.Iapws95ParameterBlock(
        phase_presentation=iapws95.PhaseType.LG,
    )

    data = _get_data()
    inlet_enthalpy_charge = iapws95.htpx(
        T=data["inlet_temperature_charge"] * pyunits.K,
        P=data["inlet_pressure_charge"] * pyunits.Pa,
    )
    inlet_enthalpy_discharge = iapws95.htpx(
        T=data["inlet_temperature_discharge"] * pyunits.K,
        P=data["inlet_pressure_discharge"] * pyunits.Pa,
    )

    m.fs.tes = ConcreteTES(
        model_data=data, 
        property_package=m.fs.props_water,
        operating_mode="combined",
    )

    m.fs.tes.inlet_charge.flow_mol.fix((0.001 / 9.58) * data["flow_mol_charge"] * data["num_tubes"])
    m.fs.tes.inlet_charge.pressure.fix(data["inlet_pressure_charge"])
    m.fs.tes.inlet_charge.enth_mol.fix(inlet_enthalpy_charge)

    m.fs.tes.inlet_discharge.flow_mol.fix(data["flow_mol_discharge"] * data["num_tubes"])
    m.fs.tes.inlet_discharge.pressure.fix(data["inlet_pressure_discharge"])
    m.fs.tes.inlet_discharge.enth_mol.fix(inlet_enthalpy_discharge)

    return m


@pytest.fixture(scope="module")
def build_concrete_tes_combined():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic="False")
    m.fs.props_water = iapws95.Iapws95ParameterBlock(
        phase_presentation=iapws95.PhaseType.LG,
    )

    data = _get_data()
    inlet_enthalpy_charge = iapws95.htpx(
        T=data["inlet_temperature_charge"] * pyunits.K,
        P=data["inlet_pressure_charge"] * pyunits.Pa,
    )
    inlet_enthalpy_discharge = iapws95.htpx(
        T=data["inlet_temperature_discharge"] * pyunits.K,
        P=data["inlet_pressure_discharge"] * pyunits.Pa,
    )

    m.fs.tes = ConcreteTES(
        model_data=data, 
        property_package=m.fs.props_water,
        operating_mode="combined",
    )

    m.fs.tes.inlet_charge.flow_mol.fix(data["flow_mol_charge"] * data["num_tubes"])
    m.fs.tes.inlet_charge.pressure.fix(data["inlet_pressure_charge"])
    m.fs.tes.inlet_charge.enth_mol.fix(inlet_enthalpy_charge)

    m.fs.tes.inlet_discharge.flow_mol.fix(data["flow_mol_discharge"] * data["num_tubes"])
    m.fs.tes.inlet_discharge.pressure.fix(data["inlet_pressure_discharge"])
    m.fs.tes.inlet_discharge.enth_mol.fix(inlet_enthalpy_discharge)

    return m


@pytest.mark.unit
def test_tes_inputs():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic="False")
    m.fs.props_water = iapws95.Iapws95ParameterBlock(
        phase_presentation=iapws95.PhaseType.LG,
    )

    data = _get_data()
    data.pop("num_segments")

    with pytest.raises(
        KeyError,
        match=(
            f"Property num_segments, a required argument, is not in model_data"
        ),
    ):
        m.fs.tes = ConcreteTES(
            model_data=data, 
            property_package=m.fs.props_water,
            operating_mode="charge",
        )


@pytest.mark.unit
def test_tes_charge(build_concrete_tes_charge):
    m = build_concrete_tes_charge

    assert len(m.fs.tes.time_periods) == 2
    for t in m.fs.tes.time_periods:
        assert hasattr(m.fs.tes.period[t], "concrete")
        assert hasattr(m.fs.tes.period[t], "tube_charge")
        assert not hasattr(m.fs.tes.period[t], "tube_discharge")

    m.fs.tes.initialize()

    assert degrees_of_freedom(m) == 0

    solver = get_solver()
    result = solver.solve(m, tee=True)

    assert_optimal_termination(result)

    conc_temp_prof = {}
    fluid_temp_prof = {}
    heat_rate_prof = {}

    exp_conc_temp, exp_heat_rate, exp_fluid_temp = _get_charge_results()

    for t in m.fs.tes.period:
        conc_temp_prof[t] = [
            value(m.fs.tes.period[t].concrete.temperature[s]) for s in m.fs.tes.segments
        ]
        heat_rate_prof[t] = [
            value(m.fs.tes.period[t].concrete.heat_rate[s]) for s in m.fs.tes.segments
        ]
        fluid_temp_prof[t] = [
            value(m.fs.tes.period[t].tube_charge.hex[s].control_volume.properties_out[0].temperature)
            for s in m.fs.tes.segments
        ]
        # NS: added abstol=1 in order to address the test failures due to triggered default reltol = 1e-7 
        pyo_unittest.assertStructuredAlmostEqual(first=exp_conc_temp[t], second=conc_temp_prof[t], abstol=1)
        pyo_unittest.assertStructuredAlmostEqual(first=exp_fluid_temp[t], second=fluid_temp_prof[t], abstol=1)
        pyo_unittest.assertStructuredAlmostEqual(first=exp_heat_rate[t], second=heat_rate_prof[t], abstol=1)
    
    return


@pytest.mark.unit
def test_tes_discharge(build_concrete_tes_discharge):
    m = build_concrete_tes_discharge

    assert len(m.fs.tes.time_periods) == 2
    for t in m.fs.tes.time_periods:
        assert hasattr(m.fs.tes.period[t], "concrete")
        assert not hasattr(m.fs.tes.period[t], "tube_charge")
        assert hasattr(m.fs.tes.period[t], "tube_discharge")

    m.fs.tes.initialize()

    assert degrees_of_freedom(m) == 0

    solver = get_solver()
    result = solver.solve(m, tee=True)

    assert_optimal_termination(result)

    conc_temp_prof = {}
    fluid_temp_prof = {}
    heat_rate_prof = {}

    exp_conc_temp, exp_heat_rate, exp_fluid_temp = _get_discharge_results()

    for t in m.fs.tes.period:
        conc_temp_prof[t] = [
            value(m.fs.tes.period[t].concrete.temperature[s]) for s in m.fs.tes.segments
        ]
        heat_rate_prof[t] = [
            value(m.fs.tes.period[t].concrete.heat_rate[s]) for s in m.fs.tes.segments
        ]
        fluid_temp_prof[t] = [
            value(m.fs.tes.period[t].tube_discharge.hex[s].control_volume.properties_out[0].temperature)
            for s in m.fs.tes.segments
        ]
        # NS: added abstol=1 in order to address the test failures due to triggered default reltol = 1e-7 
        pyo_unittest.assertStructuredAlmostEqual(first=exp_conc_temp[t], second=conc_temp_prof[t], abstol=1)
        pyo_unittest.assertStructuredAlmostEqual(first=exp_fluid_temp[t], second=fluid_temp_prof[t], abstol=1)
        pyo_unittest.assertStructuredAlmostEqual(first=exp_heat_rate[t], second=heat_rate_prof[t], abstol=1)
    
    return


@pytest.mark.unit
def test_tes_charge_discharge(build_concrete_tes_charge_discharge):
    m = build_concrete_tes_charge_discharge

    assert len(m.fs.tes.time_periods) == 2
    for t in m.fs.tes.time_periods:
        assert hasattr(m.fs.tes.period[t], "concrete")
        assert hasattr(m.fs.tes.period[t], "tube_charge")
        assert hasattr(m.fs.tes.period[t], "tube_discharge")

    m.fs.tes.initialize()

    assert degrees_of_freedom(m) == 0

    solver = get_solver()
    result = solver.solve(m, tee=True)

    assert_optimal_termination(result)

    conc_temp_prof = {}
    ch_fluid_temp_prof = {}
    di_fluid_temp_prof = {}
    heat_rate_prof = {}

    exp_conc_temp, exp_heat_rate, exp_fluid_temp = _get_charge_results()

    for t in m.fs.tes.period:
        conc_temp_prof[t] = [
            value(m.fs.tes.period[t].concrete.temperature[s]) for s in m.fs.tes.segments
        ]
        heat_rate_prof[t] = [
            value(m.fs.tes.period[t].concrete.heat_rate[s]) for s in m.fs.tes.segments
        ]
        ch_fluid_temp_prof[t] = [
            value(m.fs.tes.period[t].tube_charge.hex[s].control_volume.properties_out[0].temperature)
            for s in m.fs.tes.segments
        ]
        di_fluid_temp_prof[t] = [
            value(m.fs.tes.period[t].tube_discharge.hex[s].control_volume.properties_out[0].temperature)
            for s in m.fs.tes.segments
        ]

        pyo_unittest.assertStructuredAlmostEqual(first=exp_conc_temp[t], second=conc_temp_prof[t], abstol=5)
        pyo_unittest.assertStructuredAlmostEqual(first=exp_fluid_temp[t], second=ch_fluid_temp_prof[t], abstol=5)
        pyo_unittest.assertStructuredAlmostEqual(first=exp_heat_rate[t], second=heat_rate_prof[t], abstol=30)
        pyo_unittest.assertStructuredAlmostEqual(first=exp_conc_temp[t], second=di_fluid_temp_prof[t], abstol=10)
    
    return


@pytest.mark.unit
def tes_charge_discharge_test(build_concrete_tes_discharge_charge):
    m = build_concrete_tes_discharge_charge

    assert len(m.fs.tes.time_periods) == 2
    for t in m.fs.tes.time_periods:
        assert hasattr(m.fs.tes.period[t], "concrete")
        assert hasattr(m.fs.tes.period[t], "tube_charge")
        assert hasattr(m.fs.tes.period[t], "tube_discharge")

    m.fs.tes.initialize()

    assert degrees_of_freedom(m) == 0

    solver = get_solver()
    result = solver.solve(m, tee=True)

    assert_optimal_termination(result)

    conc_temp_prof = {}
    ch_fluid_temp_prof = {}
    di_fluid_temp_prof = {}
    heat_rate_prof = {}

    exp_conc_temp, exp_heat_rate, exp_fluid_temp = _get_discharge_results()

    for t in m.fs.tes.period:
        conc_temp_prof[t] = [
            value(m.fs.tes.period[t].concrete.temperature[s]) for s in m.fs.tes.segments
        ]
        heat_rate_prof[t] = [
            value(m.fs.tes.period[t].concrete.heat_rate[s]) for s in m.fs.tes.segments
        ]
        ch_fluid_temp_prof[t] = [
            value(m.fs.tes.period[t].tube_charge.hex[s].control_volume.properties_out[0].temperature)
            for s in m.fs.tes.segments
        ]
        di_fluid_temp_prof[t] = [
            value(m.fs.tes.period[t].tube_discharge.hex[s].control_volume.properties_out[0].temperature)
            for s in m.fs.tes.segments
        ]

        pyo_unittest.assertStructuredAlmostEqual(first=exp_conc_temp[t], second=conc_temp_prof[t], abstol=5)
        pyo_unittest.assertStructuredAlmostEqual(first=exp_conc_temp[t], second=ch_fluid_temp_prof[t], abstol=5)
        pyo_unittest.assertStructuredAlmostEqual(first=exp_heat_rate[t], second=heat_rate_prof[t], abstol=30)
        pyo_unittest.assertStructuredAlmostEqual(first=exp_fluid_temp[t], second=di_fluid_temp_prof[t], abstol=5)
    
    return


@pytest.mark.unit
def test_tes_combined(build_concrete_tes_combined):
    m = build_concrete_tes_combined

    assert len(m.fs.tes.time_periods) == 2
    for t in m.fs.tes.time_periods:
        assert hasattr(m.fs.tes.period[t], "concrete")
        assert hasattr(m.fs.tes.period[t], "tube_charge")
        assert hasattr(m.fs.tes.period[t], "tube_discharge")

    m.fs.tes.initialize()

    assert degrees_of_freedom(m) == 0

    solver = get_solver()
    result = solver.solve(m, tee=True)

    assert_optimal_termination(result)

    conc_temp_prof = {}
    ch_fluid_temp_prof = {}
    di_fluid_temp_prof = {}
    heat_rate_prof = {}

    exp_conc_temp, exp_heat_rate, exp_ch_fluid_temp, exp_di_fluid_temp = _get_combined_results()

    for t in m.fs.tes.period:
        conc_temp_prof[t] = [
            value(m.fs.tes.period[t].concrete.temperature[s]) for s in m.fs.tes.segments
        ]
        heat_rate_prof[t] = [
            value(m.fs.tes.period[t].concrete.heat_rate[s]) for s in m.fs.tes.segments
        ]
        ch_fluid_temp_prof[t] = [
            value(m.fs.tes.period[t].tube_charge.hex[s].control_volume.properties_out[0].temperature)
            for s in m.fs.tes.segments
        ]
        di_fluid_temp_prof[t] = [
            value(m.fs.tes.period[t].tube_discharge.hex[s].control_volume.properties_out[0].temperature)
            for s in m.fs.tes.segments
        ]
        # NS: added abstol=1 in order to address the test failures due to triggered default reltol = 1e-7 
        pyo_unittest.assertStructuredAlmostEqual(first=exp_conc_temp[t], second=conc_temp_prof[t], abstol=1)
        pyo_unittest.assertStructuredAlmostEqual(first=exp_ch_fluid_temp[t], second=ch_fluid_temp_prof[t], abstol=1)
        pyo_unittest.assertStructuredAlmostEqual(first=exp_heat_rate[t], second=heat_rate_prof[t], abstol=1)
        pyo_unittest.assertStructuredAlmostEqual(first=exp_di_fluid_temp[t], second=di_fluid_temp_prof[t], abstol=1)
    
    return
