"""
Info
----

"""
import pandas as pd
import yaml
from vpplib.environment import Environment
from vpplib.user_profile import UserProfile
from vpplib.photovoltaic import Photovoltaic
from vpplib.heat_pump import HeatPump

class EnergyDistrictData:
    def __init__(self, config_file="./config.yaml"):
        """
        Args:
            config_file (str): The path to the YAML configuration file containing simulation parameters.
        """
        
        try:
            with open(config_file, "r") as file:
                self.config = yaml.safe_load(file)
                self.actor_lookup = {actor["name"]: actor for actor in self.config["actors"]} 
            self.initialize_vpplib_env(self.config["general"])

        except Exception as e:
            print(f"An error occurred during initialization: {e}")

    def initialize_vpplib_env(self, general_config):
        """
        Initializes the VPP Lib components of the energy system to prepare the load shape and obtain COP values.

        This method sets up the environment, user profile, photovoltaic system, and heat pump
        components using parameters retrieved from the configuration file.
        """
        # Initialize Environment
        self.VppLib_env = Environment(
            timebase= general_config.get("timebase"),
            start=pd.Timestamp(self.config["general"]["start"]).strftime("%Y-%m-%d %H:%M:%S"), # "2015-01-01 00:00:00",  
            end=pd.Timestamp(self.config["general"]["end"]).strftime("%Y-%m-%d %H:%M:%S"), # "2015-12-31 23:45:00",       
            year= general_config.get("year"),
            time_freq=general_config.get("time_freq"),
        )

        self.baseload = pd.read_csv(general_config.get("slp_data_file"), usecols=range(1, 12))  #Spalte mit Zeit nicht in Dataframe
        self.temperature = pd.read_csv(general_config.get("mean_temp_15min"), parse_dates=["time"], index_col="time")
        self.electrical_demand = pd.DataFrame()
        self.thermal_demand = pd.DataFrame()
        self.pv_generation = pd.DataFrame()
        self.heat_pump_cops = pd.DataFrame()

        for actor in self.config["actors"]:
            if actor.get("yearly_electrical_energy_demand"):
                self.get_electrical_demand(actor)
            if actor.get("yearly_thermal_energy_demand") and actor.get("SLP_type"):
                self.get_thermal_demand(actor, general_config)
            if actor.get("heating") == "HP_Ground" or actor.get("heating") == "HP_Air":
                self.get_heat_pump_cop(actor, self.config["heat_pumps"], general_config)
            if actor.get("P_pv_nom"):
                self.get_pv_generation(actor, self.config["pv_systems"], general_config)

        for df in [self.electrical_demand, self.thermal_demand, self.heat_pump_cops, self.pv_generation]:
            if not df.empty:
                df.index = pd.to_datetime(df.index).tz_localize(None).round("15min")

    def get_electrical_demand(self, actor):
        """
        Retrieves the electrical demand data for the energy system simulation.

        This method calculates the electrical demand based on SLP data.
        """ 
        electrical_demand = self.baseload[actor.get("SLP_type")] / 1e6 * actor.get("yearly_electrical_energy_demand") # convert from 1000 MWh to 1 kWh and scale with yearly demand
        electrical_demand.index = pd.date_range(start="2015-01-01 00:00", end="2015-12-31 23:45", freq="15min")
        self.electrical_demand[f"electrical_demand_{actor.get('name')}"] = electrical_demand

    def get_thermal_demand(self, actor, general_config):
        """
        Retrieves the thermal demand data for the energy system simulation.

        This method calculates the thermal demand based on the user profile and
        environmental data obtained from the VPP Lib environment.
        """ 
        # Reload Userprofile without faulty HP paramaters
        user_profile = UserProfile(
            identifier= actor.get('name'),
            latitude= general_config.get("latitude"),
            longitude= general_config.get("longitude"),
            thermal_energy_demand_yearly= actor.get("yearly_thermal_energy_demand"),
            building_type= actor.get("building_type"), 
            comfort_factor= None,
            t_0= general_config.get("t_0")
        )
        self.thermal_demand[f"thermal_demand_{actor.get('name')}"] = user_profile.get_thermal_energy_demand()

    def get_heat_pump_cop(self, actor, hp_config, general_config):
        """
        Retrieves the coefficient of performance (COP) for the heat pump.

        This method calculates the COP value for the heat pump based on the current temperature and
        other parameters, such as the heat pump type and system temperature.
        """
        user_profile = UserProfile(
            identifier= actor.get('name'),
            latitude= general_config.get("latitude"),
            longitude= general_config.get("longitude"),
            thermal_energy_demand_yearly= actor.get("yearly_thermal_energy_demand"),
            mean_temp_days= pd.read_csv(general_config.get("mean_temp_days"), index_col="time"),
            mean_temp_hours= pd.read_csv(general_config.get("mean_temp_hours"), index_col="time"),
            mean_temp_quarter_hours= pd.read_csv(general_config.get("mean_temp_15min"), index_col="time"),
            building_type= actor.get("building_type"),
            comfort_factor= None,
            t_0= general_config.get("t_0")
        )

        user_profile.get_thermal_energy_demand()

        hp_identifier = f"{actor.get('name')}_{actor.get('heating')}"
        hp_cop_minus10 = hp_config[actor.get('heating')]["cop_at_minus10"]
        thermal_demand_max = self.thermal_demand[f"thermal_demand_{actor.get('name')}"].max()
        el_power_nom = thermal_demand_max/hp_cop_minus10

        heatpump = HeatPump(
            identifier=hp_identifier,
            unit='kW',
            environment=self.VppLib_env,
            user_profile=user_profile,
            el_power=el_power_nom,
            th_power=thermal_demand_max,
            ramp_up_time=1/15,
            ramp_down_time=1/15,
            min_runtime=1,
            min_stop_time=2,
            heat_pump_type=hp_config[actor.get('heating')]["type"],
            heat_sys_temp=hp_config[actor.get('heating')]["temp"]
        )
        heatpump.prepare_time_series()
        self.heat_pump_cops[f"hp_cop_{actor.get('name')}"] = heatpump.timeseries.cop

    def get_pv_generation(self, actor, pv_config, general_config):
        """
        Retrieves the photovoltaic (PV) generation data for the energy system simulation.

        This method calculates the PV generation based on the configured PV system parameters
        and the environmental data obtained from the VPP Lib environment.
        """
        self.VppLib_env.get_pv_data(file=general_config.get("pv_data_file"))

        user_profile = UserProfile(
            identifier= actor.get('name'),
            latitude= general_config.get("latitude"),
            longitude= general_config.get("longitude"),
            thermal_energy_demand_yearly= actor.get("yearly_thermal_energy_demand"),
            mean_temp_days= pd.read_csv(general_config.get("mean_temp_days"), index_col="time"),
            mean_temp_hours= pd.read_csv(general_config.get("mean_temp_hours"), index_col="time"),
            mean_temp_quarter_hours= pd.read_csv(general_config.get("mean_temp_15min"), index_col="time"),
            building_type= actor.get("building_type"),
            comfort_factor= None,
            t_0= general_config.get("t_0")
        )
        
        PV = Photovoltaic(
            unit='kW',
            identifier=f"{actor.get('name')}_pv",
            environment=self.VppLib_env,
            user_profile=user_profile,
            module_lib=pv_config.get("module_lib"),
            module=pv_config.get("module"),
            inverter_lib=pv_config.get("inverter_lib"),
            inverter=pv_config.get("inverter"),
            surface_tilt=pv_config.get("surface_tilt"),
            surface_azimuth=pv_config.get("surface_azimuth"),
            modules_per_string=pv_config.get("modules_per_string"),
            strings_per_inverter=pv_config.get("strings_per_inverter"),
            temp_lib=pv_config.get("temp_lib"),
            temp_model=pv_config.get("temp_model")
        )
        PV.prepare_time_series()

        pv_generation_normed = PV.timeseries
        pv_generation = pv_generation_normed * actor.get("P_pv_nom")
        self.pv_generation[f"pv_gen_{actor.get('name')}"] = pv_generation.clip(lower=0)

