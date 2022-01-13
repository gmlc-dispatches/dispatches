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
import numpy as np
import pyomo.environ as pyo
import idaes.logger as idaeslog
from pyomo.util.infeasible import log_infeasible_constraints, log_infeasible_bounds, log_close_to_bounds
from idaes.apps.multiperiod.multiperiod import MultiPeriodModel
from RE_flowsheet import *
from load_parameters import *

design_opt = True


def turb_costs(m):
    m.fs.h2_turbine.op_cost = Expression(
        expr=turbine_op_cost,
        doc="fixed cost of operating turbine $/kW-pr"
    )
    m.fs.h2_turbine.var_cost = Expression(
        expr=turbine_var_cost,
        doc="variable operating cost of turbine $/kWh"
    )


def initialize_mp(m, verbose=False):
    outlvl = idaeslog.INFO if verbose else idaeslog.WARNING

    m.fs.mixer.purchased_hydrogen_feed.flow_mol[0].fix(h2_turb_min_flow)
    m.fs.mixer.air_feed.flow_mol[0].fix(m.fs.mixer.purchased_hydrogen_feed.flow_mol[0] * air_h2_ratio)
    m.fs.mixer.air_h2_ratio.deactivate()
    # initial guess of air feed that will be needed to balance out hydrogen feed
    m.fs.mixer.initialize(outlvl=outlvl)
    m.fs.mixer.purchased_hydrogen_feed.flow_mol[0].unfix()
    m.fs.mixer.air_h2_ratio.activate()
    if verbose:
        m.fs.mixer.report(dof=True)

    propagate_state(m.fs.mixer_to_turbine)

    m.fs.h2_turbine.initialize(outlvl=outlvl)
    if verbose:
        m.fs.h2_turbine.report(dof=True)


def turb_model(verbose):
    m = ConcreteModel()

    m.fs = FlowsheetBlock(default={"dynamic": False})
    m.fs.h2turbine_props = GenericParameterBlock(default=hturbine_config)
    m.fs.reaction_params = h2_reaction_props.H2ReactionParameterBlock(
        default={"property_package": m.fs.h2turbine_props})

    m.fs.mixer = Mixer(
        default={
            # using minimize pressure for all inlets and outlet of the mixer
            # because pressure of inlets is already fixed in flowsheet, using equality will over-constrain
            "momentum_mixing_type": MomentumMixingType.minimize,
            "property_package": m.fs.h2turbine_props,
            "inlet_list":
                ["air_feed", "purchased_hydrogen_feed"]}
    )

    m.fs.mixer.air_feed.temperature[0].fix(PEM_temp)
    m.fs.mixer.air_feed.pressure[0].fix(pem_bar * 1e5)
    m.fs.mixer.air_feed.mole_frac_comp[0, "oxygen"].fix(0.2054)
    m.fs.mixer.air_feed.mole_frac_comp[0, "argon"].fix(0.0032)
    m.fs.mixer.air_feed.mole_frac_comp[0, "nitrogen"].fix(0.7672)
    m.fs.mixer.air_feed.mole_frac_comp[0, "water"].fix(0.0240)
    m.fs.mixer.air_feed.mole_frac_comp[0, "hydrogen"].fix(2e-4)

    m.fs.mixer.purchased_hydrogen_feed.pressure[0].fix(pem_bar * 1e5)
    m.fs.mixer.purchased_hydrogen_feed.temperature[0].fix(PEM_temp)
    m.fs.mixer.purchased_hydrogen_feed.mole_frac_comp[0, "hydrogen"].fix(1)

    m.fs.mixer.air_h2_ratio = Constraint(
        expr=m.fs.mixer.air_feed.flow_mol[0] == air_h2_ratio * m.fs.mixer.purchased_hydrogen_feed.flow_mol[0])

    # Return early without adding Turbine
    # return None, m.fs.mixer, m.fs.translator

    # Add the hydrogen turbine
    m.fs.h2_turbine = HydrogenTurbine(
        default={"property_package": m.fs.h2turbine_props,
                 "reaction_package": m.fs.reaction_params})
    min_mole_frac = 1e-4
    m.fs.h2_turbine.compressor.deltaP.fix(compressor_dp * 1e5)
    m.fs.h2_turbine.compressor.efficiency_isentropic.fix(0.86)
    # m.fs.h2_turbine.compressor.control_volume.properties_in[0].mole_frac_phase_comp['Vap', 'hydrogen'].setlb(
    #     min_mole_frac)
    # m.fs.h2_turbine.compressor.control_volume.properties_in[0].mole_frac_phase_comp['Vap', 'argon'].setlb(min_mole_frac)
    # m.fs.h2_turbine.compressor.control_volume.properties_in[0].mole_frac_phase_comp['Vap', 'nitrogen'].setlb(
    #     min_mole_frac)
    # m.fs.h2_turbine.compressor.control_volume.properties_in[0].mole_frac_phase_comp['Vap', 'water'].setlb(min_mole_frac)
    # m.fs.h2_turbine.compressor.control_volume.properties_in[0].mole_frac_phase_comp['Vap', 'oxygen'].setlb(
    #     min_mole_frac)
    # m.fs.h2_turbine.compressor.properties_isentropic[0].mole_frac_phase_comp['Vap', 'hydrogen'].setlb(min_mole_frac)
    # m.fs.h2_turbine.compressor.properties_isentropic[0].mole_frac_phase_comp['Vap', 'argon'].setlb(min_mole_frac)
    # m.fs.h2_turbine.compressor.properties_isentropic[0].mole_frac_phase_comp['Vap', 'nitrogen'].setlb(min_mole_frac)
    # m.fs.h2_turbine.compressor.properties_isentropic[0].mole_frac_phase_comp['Vap', 'water'].setlb(min_mole_frac)
    # m.fs.h2_turbine.compressor.properties_isentropic[0].mole_frac_phase_comp['Vap', 'oxygen'].setlb(min_mole_frac)

    # Specify the Stoichiometric Conversion Rate of hydrogen
    # in the equation shown below
    # H2(g) + O2(g) --> H2O(g) + energy
    # Complete Combustion
    m.fs.h2_turbine.stoic_reactor.conversion.fix(0.99)

    # m.fs.h2_turbine.turbine.deltaP.fix(-(H2_turb_pressure_bar - .101325) * 1e5)
    # m.fs.h2_turbine.turbine.deltaP.setub(-(H2_turb_pressure_bar - .101325) * 1e5 * 0.75)
    # m.fs.h2_turbine.turbine.deltaP.setlb(-(H2_turb_pressure_bar - .101325) * 1e5 * 1.25)

    min_mole_frac = 1e-5

    m.fs.h2_turbine.turbine.deltaP.fix(-compressor_dp * 1e5)
    m.fs.h2_turbine.turbine.efficiency_isentropic.fix(0.89)
    # m.fs.h2_turbine.turbine.control_volume.properties_in[0].mole_frac_phase_comp['Vap', 'hydrogen'].setlb(min_mole_frac)
    # m.fs.h2_turbine.turbine.control_volume.properties_in[0].mole_frac_phase_comp['Vap', 'argon'].setlb(min_mole_frac)
    # m.fs.h2_turbine.turbine.control_volume.properties_in[0].mole_frac_phase_comp['Vap', 'nitrogen'].setlb(min_mole_frac)
    # m.fs.h2_turbine.turbine.control_volume.properties_in[0].mole_frac_phase_comp['Vap', 'water'].setlb(min_mole_frac)
    # m.fs.h2_turbine.turbine.control_volume.properties_in[0].mole_frac_phase_comp['Vap', 'oxygen'].setlb(min_mole_frac)
    # m.fs.h2_turbine.turbine.properties_isentropic[0].mole_frac_phase_comp['Vap', 'hydrogen'].setlb(min_mole_frac)
    # m.fs.h2_turbine.turbine.properties_isentropic[0].mole_frac_phase_comp['Vap', 'argon'].setlb(min_mole_frac)
    # m.fs.h2_turbine.turbine.properties_isentropic[0].mole_frac_phase_comp['Vap', 'nitrogen'].setlb(min_mole_frac)
    # m.fs.h2_turbine.turbine.properties_isentropic[0].mole_frac_phase_comp['Vap', 'water'].setlb(min_mole_frac)
    # m.fs.h2_turbine.turbine.properties_isentropic[0].mole_frac_phase_comp['Vap', 'oxygen'].setlb(min_mole_frac)

    # m.fs.mixer.purchased_hydrogen_feed.flow_mol[0].setlb(0.001)

    m.fs.mixer_to_turbine = Arc(
        source=m.fs.mixer.outlet,
        destination=m.fs.h2_turbine.compressor.inlet
    )

    TransformationFactory("network.expand_arcs").apply_to(m)

    iscale.set_scaling_factor(m.fs.mixer.air_feed_state[0.0].enth_mol_phase['Vap'], 1)
    iscale.set_scaling_factor(m.fs.mixer.purchased_hydrogen_feed_state[0.0].enth_mol_phase['Vap'], 1)
    iscale.set_scaling_factor(m.fs.mixer.mixed_state[0.0].enth_mol_phase['Vap'], 1)
    iscale.set_scaling_factor(m.fs.h2_turbine.compressor.control_volume.properties_in[0.0].enth_mol_phase['Vap'], 1)
    iscale.set_scaling_factor(m.fs.h2_turbine.compressor.control_volume.properties_out[0.0].enth_mol_phase['Vap'], 1)
    iscale.set_scaling_factor(m.fs.h2_turbine.compressor.control_volume.work, 1)
    iscale.set_scaling_factor(m.fs.h2_turbine.compressor.properties_isentropic[0.0].enth_mol_phase['Vap'], 1)
    iscale.set_scaling_factor(m.fs.h2_turbine.stoic_reactor.control_volume.properties_in[0.0].enth_mol_phase['Vap'], 1)
    iscale.set_scaling_factor(m.fs.h2_turbine.stoic_reactor.control_volume.properties_out[0.0].enth_mol_phase['Vap'], 1)
    iscale.set_scaling_factor(m.fs.h2_turbine.stoic_reactor.control_volume.rate_reaction_extent[0, 'R1'], 1)
    iscale.set_scaling_factor(m.fs.h2_turbine.turbine.control_volume.properties_in[0.0].enth_mol_phase['Vap'], 1)
    iscale.set_scaling_factor(m.fs.h2_turbine.turbine.control_volume.properties_out[0.0].enth_mol_phase['Vap'], 1)
    iscale.set_scaling_factor(m.fs.h2_turbine.turbine.control_volume.work, 1)
    iscale.set_scaling_factor(m.fs.h2_turbine.turbine.properties_isentropic[0.0].enth_mol_phase['Vap'], 1)

    iscale.calculate_scaling_factors(m)

    initialize_mp(m, verbose=verbose)

    if verbose:
        solve_log = idaeslog.getInitLogger("infeasibility", idaeslog.INFO,
                                           tag="properties")
        log_infeasible_constraints(m, logger=solve_log, tol=1e-4, log_expression=True, log_variables=True)
        log_infeasible_bounds(m, logger=solve_log, tol=1e-4)
        # log_close_to_bounds(m, logger=solve_log)

    turb_costs(m)

    return m


def turb_optimize(verbose=False):
    # create the multiperiod model object
    mp_model = MultiPeriodModel(n_time_points=n_time_points,
                                process_model_func=partial(turb_model, verbose=verbose),
                                linking_variable_func=lambda x, y: [],
                                periodic_variable_func=lambda x, y: [])

    mp_model.build_multi_period_model()

    m = mp_model.pyomo_model
    blks = mp_model.get_active_process_blocks()

    m.h2_price_per_kg = pyo.Param(default=h2_price_per_kg, mutable=True)
    m.turb_system_capacity = Var(domain=NonNegativeReals, initialize=turb_p_mw * 1e3, units=pyunits.kW)
    # m.turb_system_capacity.setlb(70 * 1e3)

    if not design_opt:
        m.turb_system_capacity.fix(turb_p_mw * 1e3)

    for blk in blks:
        blk_turb = blk.fs.h2_turbine
        # add operating constraints
        blk_turb.electricity = Expression(blk_turb.flowsheet().config.time,
                                          rule=lambda b, t: (-b.turbine.work_mechanical[0]
                                                             - b.compressor.work_mechanical[0]) * 1e-3)
        blk_turb.max_p = Constraint(blk_turb.flowsheet().config.time,
                                    rule=lambda b, t: b.electricity[t] <= m.turb_system_capacity)
        # blk_turb.min_f = Constraint(blk_turb.flowsheet().config.time,
        #                             rule=lambda b, t: b.compressor.control_volume.properties_in[0].flow_mol *
        #                                               b.compressor.control_volume.properties_in[0].mole_frac_comp['hydrogen']
        #                                               >= h2_turb_min_flow)
        # add operating costs
        blk_turb.op_total_cost = Expression(
            expr=m.turb_system_capacity * blk_turb.op_cost / 8760 + blk_turb.var_cost * blk_turb.electricity[0]
        )

        # add market data for each block
        blk.lmp_signal = pyo.Param(default=0, mutable=True)
        blk.revenue = blk.lmp_signal * (blk_turb.electricity[0])
        blk.profit = pyo.Expression(expr=blk.revenue
                                         - blk_turb.op_total_cost
                                    )
        blk.hydrogen_revenue = Expression(expr=m.h2_price_per_kg / h2_mols_per_kg * (
            -blk.fs.mixer.purchased_hydrogen_feed_state[0].flow_mol) * 3600)

    m.turb_cap_cost = pyo.Param(default=turbine_cap_cost, mutable=True)

    n_weeks = 1

    m.annual_revenue = Expression(expr=(sum([blk.profit + blk.hydrogen_revenue for blk in blks])) * 52 / n_weeks)

    m.NPV = Expression(expr=-(
                              m.turb_cap_cost * m.turb_system_capacity
                              ) + PA * m.annual_revenue)
    m.obj = pyo.Objective(expr=-m.NPV)
    # m.obj = pyo.Objective(expr=0)

    opt = pyo.SolverFactory('ipopt')
    h2_purchased = []
    comp_kwh = []
    turb_kwh = []
    h2_turbine_elec = []
    h2_revenue = []
    elec_revenue = []

    for week in range(n_weeks):
        # print("Solving for week: ", week)
        for (i, blk) in enumerate(blks):
            blk.lmp_signal.set_value(weekly_prices[week][i] * 1e-3)     # to $/kWh
        # opt.options['bound_push'] = 10e-9
        opt.options['max_iter'] = 50000
        # opt.options['tol'] = 1e-6
        # opt.options['acceptable_tol'] = 1e-4

        opt.options["halt_on_ampl_error"] = "yes"

        if verbose:
            solve_log = idaeslog.getInitLogger("infeasibility", idaeslog.INFO,
                                               tag="properties")
            log_infeasible_constraints(m, logger=solve_log, tol=1e-4, log_expression=True, log_variables=True)
            log_infeasible_bounds(m, logger=solve_log, tol=1e-4)
            # log_close_to_bounds(m, logger=solve_log)

            print("Badly scaled variables before solve:")
            for v, sv in iscale.badly_scaled_var_generator(m, large=1e2, small=1e-2, zero=1e-12):
                print(f"    {v} -- {sv} -- {iscale.get_scaling_factor(v)}")
        ok = False
        try:
            res = opt.solve(m, tee=True, symbolic_solver_labels=True)
            ok = res.Solver.status == 'ok'
        except:
            pass

        if not ok:
            solve_log = idaeslog.getInitLogger("infeasibility", idaeslog.INFO,
                                                tag="properties")
            log_infeasible_constraints(m, logger=solve_log, tol=1e-4, log_expression=True, log_variables=True)
            log_infeasible_bounds(m, logger=solve_log, tol=1e-4)
            # log_close_to_bounds(m, logger=solve_log)

            # print("Badly scaled variables after solve:")
            # for v, sv in iscale.badly_scaled_var_generator(m, large=1e2, small=1e-2, zero=1e-12):
            #     print(f"    {v} -- {sv} -- {iscale.get_scaling_factor(v)}")

        h2_purchased.append([pyo.value(blks[i].fs.mixer.purchased_hydrogen_feed_state[0].flow_mol) * 3600 / h2_mols_per_kg for i in range(n_time_points)])
        h2_turbine_elec.append([pyo.value(blks[i].fs.h2_turbine.electricity[0]) for i in range(n_time_points)])
        elec_revenue.append([pyo.value(blks[i].profit) for i in range(n_time_points)])
        h2_revenue.append([pyo.value(blks[i].hydrogen_revenue) for i in range(n_time_points)])
        turb_kwh.append([pyo.value(blks[i].fs.h2_turbine.turbine.work_mechanical[0]) * -1e-3 for i in range(n_time_points)])
        comp_kwh.append(
            [pyo.value(blks[i].fs.h2_turbine.compressor.work_mechanical[0]) * 1e-3 for i in range(n_time_points)])

    n_weeks_to_plot = 1
    hours = np.arange(n_time_points * n_weeks_to_plot)
    lmp_array = weekly_prices[0:n_weeks_to_plot].flatten()
    h2_purchased = np.asarray(h2_purchased[0:n_weeks_to_plot]).flatten()
    turb_kwh = np.asarray(turb_kwh[0:n_weeks_to_plot]).flatten()
    comp_kwh = np.asarray(comp_kwh[0:n_weeks_to_plot]).flatten()
    h2_turbine_elec = np.asarray(h2_turbine_elec[0:n_weeks_to_plot]).flatten()
    h2_revenue = np.asarray(h2_revenue[0:n_weeks_to_plot]).flatten()
    elec_revenue = np.asarray(elec_revenue[0:n_weeks_to_plot]).flatten()

    turb_cap = value(m.turb_system_capacity) * 1e-3

    print("avg turb eff", np.average((turb_kwh - comp_kwh)/turb_kwh))
    print("turb mw", turb_cap)
    print("h2 rev week", sum(h2_revenue))
    print("elec rev week", sum(elec_revenue))
    print("annual rev", value(m.annual_revenue))
    print("npv", value(m.NPV))

    fig, ax1 = plt.subplots(3, 1, figsize=(12, 8))
    plt.suptitle(f"Optimal NPV ${round(value(m.NPV) * 1e-6)}mil from {round(turb_cap, 2)} MW Turbine")

    # color = 'tab:green'
    ax1[0].set_xlabel('Hour')
    # ax1[0].set_ylabel('kW', )
    ax1[0].step(hours, comp_kwh, label="Compressor In [kW]")
    ax1[0].step(hours, turb_kwh, label="Turb Out[kW]")
    ax1[0].step(hours, h2_turbine_elec, label="H2 Turbine Net[kW]")
    ax1[0].tick_params(axis='y', )
    ax1[0].legend()
    ax1[0].grid(b=True, which='major', color='k', linestyle='--', alpha=0.2)
    ax1[0].minorticks_on()
    ax1[0].grid(b=True, which='minor', color='k', linestyle='--', alpha=0.2)

    ax2 = ax1[0].twinx()
    color = 'k'
    ax2.set_ylabel('LMP [$/MWh]', color=color)
    ax2.plot(hours, lmp_array[0:len(hours)], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # ax1[1].set_xlabel('Hour')
    # ax1[1].set_ylabel('kg/hr', )
    ax1[1].step(hours, h2_purchased, label="H2 purchased [kg/hr]")

    ax1[1].tick_params(axis='y', )
    ax1[1].legend()
    ax1[1].grid(b=True, which='major', color='k', linestyle='--', alpha=0.2)
    ax1[1].minorticks_on()
    ax1[1].grid(b=True, which='minor', color='k', linestyle='--', alpha=0.2)

    ax2 = ax1[1].twinx()
    color = 'k'
    ax2.set_ylabel('LMP [$/MWh]', color=color)
    ax2.plot(hours, lmp_array[0:len(hours)], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    ax1[2].set_xlabel('Hour')
    ax1[2].step(hours, elec_revenue, label="Elec rev")
    ax1[2].step(hours, h2_revenue, label="H2 rev")
    ax1[2].step(hours, np.cumsum(elec_revenue), label="Elec rev cumulative")
    ax1[2].step(hours, np.cumsum(h2_revenue), label="H2 rev cumulative")
    ax1[2].legend()
    ax1[2].grid(b=True, which='major', color='k', linestyle='--', alpha=0.2)
    ax1[2].minorticks_on()
    ax1[2].grid(b=True, which='minor', color='k', linestyle='--', alpha=0.2)

    plt.show()

    return turb_cap, sum(h2_revenue), sum(elec_revenue), value(m.NPV)


if __name__ == "__main__":
    turb_optimize(False)
