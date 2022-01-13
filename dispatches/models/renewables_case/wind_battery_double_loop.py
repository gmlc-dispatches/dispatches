from wind_battery_LMP import wind_battery_optimize

def transform_design_model_to_operation_model(mp_wind_battery):

    blks = mp_wind_battery.get_active_process_blocks()

    for b in blks:
        b.fs.windpower.system_capacity.fix()
        b.fs.battery.nameplate_power.fix()

        ## TODO: deactivate periodic boundary condition??

    return

if __name__ == "__main__":
    mp_wind_battery = wind_battery_optimize()
    transform_design_model_to_operation_model(mp_wind_battery)
