"""
Info
----

"""
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
from vpplib.environment import Environment
from vpplib.user_profile import UserProfile
from vpplib.photovoltaic import Photovoltaic
from vpplib.heat_pump import HeatPump
from vpplib.battery_electric_vehicle import BatteryElectricVehicle
import logging
import pypsa

class EnergySystemSimulation:
    """
    A class to simulate an energy system using photovoltaic and electrical storage components.

    Attributes:
        parameters (dict): A dictionary containing all the parameters required for the simulation.
        VppLib_env (Environment): The environment setup for the simulation.
        user_profile (UserProfile): The user profile containing location and demand data.
        pv (Photovoltaic): The photovoltaic system instance.
        hp (HeatPump): The heat pump instance.
        loadshape (pd.DataFrame): The load shape data for the simulation.
        network (pypsa.Network): The PyPSA network instance for the energy system.
    """

    def __init__(self, config_file="./config.yaml"):
        """
        Initializes the energy system simulation with parameters loaded from a configuration file.

        Args:
            config_file (str): The path to the YAML configuration file containing simulation parameters.
        """
        
        try:
            with open(config_file, "r") as file:
                self.parameters = yaml.safe_load(file)

            self.init_timeframe()
            self.initialize_vpplib_env()
            self.get_thermal_and_electrical_demand()
            self.get_heat_pump_cop()
            self.get_pv_generation()

            self.initialize_pypsa_network()
            self.total_cost = 0
        except Exception as e:
            print(f"An error occurred during initialization: {e}")

    def get_param(self, *keys):
        """
        Retrieves a parameter value from the configuration dictionary.

        Args:
            *keys: A variable length argument list of keys to navigate through the nested dictionary.

        Returns:
            The value of the specified parameter, or None if the parameter is not found.
        """
        param = self.parameters
        for key in keys:
            if key in param:
                param = param[key]
            else:
                return None
        return param

    def init_timeframe(self):
        """
        Initializes the simulation timeframe based on the specified start and end dates.

        Args:
            start (str): The start date for the simulation in the format "YYYY-MM-DD".
            end (str): The end date for the simulation in the format "YYYY-MM-DD".
        """
        start_yaml = pd.Timestamp(self.get_param("general", "start"))
        end_yaml = pd.Timestamp(self.get_param("general", "end"))

        range_of_starts = pd.date_range(start_yaml, end_yaml - pd.Timedelta(days=7), freq="D")
        random_start = np.random.choice(range_of_starts)

        self.start = pd.Timestamp(random_start).replace(hour=0, minute=0)
        self.end = self.start + pd.Timedelta(days=7) - pd.Timedelta(minutes=15)
        self.datetime_range = pd.date_range(start=self.start, end=self.end, freq="15min")
        self.num_time_steps = len(self.datetime_range)
        self.week_id = self.start.strftime("KW%U_%Y")

    def initialize_vpplib_env(self):
        """
        Initializes the VPP Lib components of the energy system to prepare the load shape and obtain COP values.

        This method sets up the environment, user profile, photovoltaic system, and heat pump
        components using parameters retrieved from the configuration file.
        """
        # Initialize Environment
        self.VppLib_env = Environment(
            timebase=self.get_param("general", "timebase"),
            start= self.start.strftime("%Y-%m-%d %H:%M:%S"), #"2015-01-01 00:00:00",   #self.get_param("general", "start"),
            end= self.end.strftime("%Y-%m-%d %H:%M:%S"), #"2015-12-31 23:45:00",       #self.get_param("general", "end"),
            year=self.get_param("general", "year"),
            time_freq="15 min",
        )
        
        self.VppLib_env.get_pv_data(file=self.get_param("general", "pv_data_file"))
        self.VppLib_env.get_dwd_mean_temp_hours(lat=self.get_param("general", "latitude"),lon=self.get_param("general", "longitude"))
        self.VppLib_env.get_dwd_mean_temp_days(lat=self.get_param("general", "latitude"),lon=self.get_param("general", "longitude"))

    def get_thermal_and_electrical_demand(self):
        """
        Retrieves the thermal and electrical demand data for the energy system simulation.

        This method calculates the thermal and electrical demand based on the user profile and
        environmental data obtained from the VPP Lib environment.

        Electrical demand is based on SLP data, while thermal demand is calculated using the user profile.
        """
        baseload = pd.read_csv(self.get_param("general", "slp_data_file"))
        baseload.drop(columns=["Time"], inplace=True)
        baseload.set_index(self.VppLib_env.pv_data.index, inplace=True)
        
        self.energy_demand = pd.DataFrame()

        demands = self.get_param("energy_demand")

        combined_electrical_baseload = None
        combined_thermal_energy_demand = None

        if demands is not None:
            for demand in demands:
                electrical_demand = baseload[demand["SLP_type"]] / 1e6 * demand["yearly_electrical_energy_demand"] # convert from 1000 MWh to 1 kWh and scale with yearly demand
                self.energy_demand[f"electrical_baseload_{demand['name']}"] = electrical_demand
                
                # Reload Userprofile without faulty HP paramaters
                self.user_profile = UserProfile(
                    identifier=self.get_param("general", "name"),
                    latitude=self.get_param("general", "latitude"),
                    longitude=self.get_param("general", "longitude"),
                    thermal_energy_demand_yearly= demand["yearly_thermal_energy_demand"],
                    building_type= demand["building_type"], 
                    comfort_factor= None,
                    t_0= self.get_param("general", "t_0"),
                )
                self.user_profile.get_thermal_energy_demand()

                thermal_energy_demand = self.user_profile.thermal_energy_demand
                self.energy_demand[f"thermal_energy_demand_{demand['name']}"] = thermal_energy_demand

                if combined_electrical_baseload is None:
                    combined_electrical_baseload = electrical_demand
                    combined_thermal_energy_demand = thermal_energy_demand
                else:
                    combined_electrical_baseload += electrical_demand
                    combined_thermal_energy_demand += thermal_energy_demand

        self.energy_demand["electrical_baseload"] = combined_electrical_baseload
        self.energy_demand["thermal_energy_demand"] = combined_thermal_energy_demand

        self.df_market_data = pd.read_csv("./input/market_data.csv")
        self.df_market_data["time"] = pd.to_datetime(self.df_market_data["time"])
        self.df_market_data.set_index("time", inplace=True)
        self.df_market_data = self.df_market_data.loc[self.start:self.end]

    def get_heat_pump_cop(self):
        """
        Retrieves the coefficient of performance (COP) for the heat pump.

        This method calculates the COP value for the heat pump based on the current temperature and
        other parameters, such as the heat pump type and system temperature.
        """
        self.heat_pumps = pd.DataFrame()
        # Initialize User Profile with mean_temp_hours for COP calculation
        user_profile = UserProfile(
            identifier=self.get_param("general", "name"),
            latitude=self.get_param("general", "latitude"),
            longitude=self.get_param("general", "longitude"),
            thermal_energy_demand_yearly= 10000, #self.get_param("energy_demand", "yearly_thermal_energy_demand"), # Dummy Value for COP
            mean_temp_days=self.VppLib_env.mean_temp_days,
            mean_temp_hours=self.VppLib_env.mean_temp_hours,
            mean_temp_quarter_hours=self.VppLib_env.mean_temp_hours.resample("15 Min").interpolate(),
            building_type= "DE_HEF33", #self.get_param("energy_demand", "building_type"), # Dummy Value for COP
            #comfort_factor= None,
            #t_0= self.get_param("energy_demand", "t_0"),
        )
        user_profile.get_thermal_energy_demand()

        hp_systems = self.get_param("heat_pumps")

        if hp_systems is not None:
            for hp in hp_systems:
                hp_identifier = f'{self.get_param("general", "name")}_{hp["name"]}'
                
                heatpump = HeatPump(
                    identifier=hp_identifier,
                    unit='kW',
                    environment=self.VppLib_env,
                    user_profile=user_profile,
                    el_power=hp["power_el"],
                    th_power=hp["power_th"],
                    ramp_up_time=1/15,
                    ramp_down_time=1/15,
                    min_runtime=1,
                    min_stop_time=2,
                    heat_pump_type=hp["type"],
                    heat_sys_temp=self.get_param("thermal_storage", "t_max_TES"),
                )

                if hp["power_el"] > 0:
                    heatpump.prepare_time_series()
                    self.heat_pumps[f"hp_cop_{hp['name']}"] = heatpump.timeseries.cop
                else:
                    self.heat_pumps[f"hp_cop_{hp['name']}"] = 0
        else:
            self.heat_pumps["hp_cop"] = 0

    def get_pv_generation(self):
        """
        Retrieves the photovoltaic (PV) generation data for the energy system simulation.

        This method calculates the PV generation based on the configured PV system parameters
        and the environmental data obtained from the VPP Lib environment.
        """
        # PV Generation
        pv_systems = self.get_param("pv_systems")
        self.pv_gen = pd.DataFrame()

        if pv_systems is not None:
            for pv in pv_systems:
                    PV = Photovoltaic(
                        unit='kW',
                        identifier=f'{self.get_param("general", "name")}_{pv["name"]}',
                        environment=self.VppLib_env,
                        user_profile=self.user_profile,
                        module_lib=pv["module_lib"],
                        module=pv["module"],
                        inverter_lib=pv["inverter_lib"],
                        inverter=pv["inverter"],
                        surface_tilt=pv["surface_tilt"],
                        surface_azimuth=pv["surface_azimuth"],
                        modules_per_string=pv["modules_per_string"],
                        strings_per_inverter=pv["strings_per_inverter"],
                        temp_lib=pv["temp_lib"],
                        temp_model=pv["temp_model"]
                    )
                    PV.prepare_time_series()
                    
                    self.pv_gen[f"pv_gen_{pv['name']}"] = PV.timeseries / 1.76  # Beispiel mit Skalierung
                    self.pv_gen[f"pv_gen_{pv['name']}"] = self.pv_gen[f"pv_gen_{pv['name']}"].clip(lower=0)
        else:
            self.pv_gen["pv_gen"] = 0

    def prepare_loadshape(self):
        """
        Prepares the load shape for the energy system simulation by combining various demand sources.

        This method loads baseload data, computes PV generation, retrieves heat pump coefficients,
        and sets the electricity price based on the selected pricing model.
        """

        self.loadshape = pd.DataFrame()

        demand = self.energy_demand.loc[self.start:self.end]
        pv_gen = self.pv_gen.loc[self.start:self.end]
        hp_cop = self.heat_pumps.loc[self.start:self.end]

        self.loadshape = pd.concat([demand, pv_gen, hp_cop], axis=1)

        self.get_electricity_demand_price()
        self.get_co2_intensity()
        self.get_electricity_feed_in_tariff()

        self.loadshape = self.loadshape.interpolate(method="linear", limit_direction="both")

    
    def calculate_cost(self):
        """
        Calculates the total cost for the energy system simulation based on the load shape.

        This method retrieves the relevant price data and computes the overall cost incurred 
        during the simulation period.
        """
        self.loadshape["cost"] = self.network.links_t.p0["grid_demand_link"] * self.electricity_demand_price #+ self.network.links_t.p0['grid_feed-in_link'] * self.electricity_feed_in_tariff #+ self.network.generators_t.p["natural_gas_generator"] * self.get_param("prices", "natural_gas") + self.network.generators_t.p["district_heating_generator"] * self.get_param("prices", "district_heat")
        self.total_cost = self.loadshape["cost"].sum()

    def calculate_co2(self):
        """
        Calculates the total CO2 emissions for the energy system simulation.

        This method is a placeholder for calculating CO2 emissions based on the load shape.
        The implementation is yet to be defined.
        """
        self.loadshape["co2"] = self.network.links_t.p0["grid_demand_link"] * self.co2_intensity #+ self.network.generators_t.p["natural_gas_generator"] * self.get_param("co2", "gas") + self.network.generators_t.p["district_heating_generator"] * self.get_param("co2", "district_heat")
        self.total_emissions = self.loadshape["co2"].sum() 

    def get_electricity_demand_price(self):
        """
        Retrieves the electricity demand price based on the configured pricing model.

        This method sets the electricity demand price for the simulation based on whether
        the pricing is dynamic or static.
        """
        if self.get_param("prices", "electricity_type")=="dynamic":
            self.loadshape["price_el"] = self.df_market_data["price_el"][self.start:self.end]
            grid_charges = self.loadshape["price_el"].index.to_series().apply(self.get_grid_charge_price_for_time)
            add_charges = self.get_param("prices", "kwk_aufschlag") + self.get_param("prices", "offshore_umlage") + self.get_param("prices", "konzessionsabgabe") + self.get_param("prices", "stromsteuer") 
            self.loadshape["price_el"] = (self.loadshape["price_el"] + grid_charges + add_charges) * (1+ self.get_param("prices", "mehrwertsteuer"))
        elif self.get_param("prices", "electricity_type")=="static":
            self.loadshape["price_el"] = self.get_param("prices", "electricity")

        self.electricity_demand_price = np.array(self.loadshape["price_el"].to_list())

    def get_grid_charge_price_for_time(self, timestamp):
        """
        Defines network charges for the given timestamp
        """
        hour, minute = timestamp.hour, timestamp.minute
        if (hour == 18 and minute >= 0) or (hour == 19) or (hour == 20 and minute < 15):
            return self.get_param("prices", "netzentgelt_ht")  # 18:00 - 20:15 Uhr
        elif (hour == 23 and minute >= 45) or (hour >= 0 and hour < 5) or (hour == 5 and minute < 45):
            return self.get_param("prices", "netzentgelt_nt")  # 23:45 - 5:45 Uhr
        else:
            return self.get_param("prices", "netzentgelt_st")  # Standard

    def get_co2_intensity(self):
        """
        Retrieves the co2 intensity based on the configured emission model.

        This method sets the co2 intensity for the simulation based on whether
        the pricing is dynamic or static.
        """
        if self.get_param("co2_emissions", "emission_type")=="dynamic":
            self.loadshape["co2_el"] = self.df_market_data["co2_el"][self.start:self.end]
        elif self.get_param("co2_emissions", "emission_type")=="static":
            self.loadshape["co2_el"] = self.get_param("co2_emissions", "electricity")

        self.co2_intensity = np.array(self.loadshape["co2_el"].to_list())

    def get_electricity_feed_in_tariff(self):
        """
        Retrieves the electricity feed-in tariff based on the configured pricing model.

        This method sets the feed-in tariff for exported electricity based on whether the
        pricing is dynamic or static.
        """
        if self.get_param("prices", "electricity_type")=="dynamic":
            self.electricity_feed_in_tariff = - np.array(self.loadshape["price_el"].to_list())
        elif self.get_param("prices", "electricity_type")=="static":
            self.electricity_feed_in_tariff = - self.get_param("prices", "feed_in_tariff")     


    def set_optimization_objective(self):
        """
        Sets the optimization objective for the energy system simulation.

        This method configures the optimization objective based on the defined parameters,
        which can be cost minimization or CO2 emissions reduction.
        """

        if self.get_param("general", "optimizer") == "cost":
            self.marginal_electricity_demand = self.electricity_demand_price
            self.marginal_electricity_feed_in = self.electricity_feed_in_tariff
            self.marginal_district_heat = self.get_param("prices", "district_heat")
            self.marginal_gas = self.get_param("prices", "natural_gas")
        
        elif self.get_param("general", "optimizer") == "co2":
            self.marginal_electricity_demand = self.co2_intensity
            self.marginal_electricity_feed_in = 0
            self.marginal_gas = self.get_param("co2_emissions", "gas")
            self.marginal_district_heat = self.get_param("co2_emissions", "district_heat")
        
    def add_buses(self):
        """
        Adds buses to the PyPSA network for managing electrical and thermal flows.

        This method creates necessary buses for various energy carriers, including electricity, thermal energy,
        grid demand, and other storage systems.
        """
        self.network.add('Carrier', name='AC')
        self.network.add('Carrier', name='heat')
        self.network.add('Carrier', name='AC_grid')
        self.network.add('Carrier', name='AC_grid_feed-in')
        self.network.add('Carrier', name='DC_BAT')
        self.network.add('Carrier', name='gas')
        self.network.add('Carrier', name='heat_DH')   

        self.network.add('Bus', name='electrical_bus', carrier='AC')
        self.network.add('Bus', name='thermal_bus', carrier='heat')
        self.network.add('Bus', name='grid_demand_bus', carrier='AC_grid')
        self.network.add('Bus', name='grid_feed-in_bus', carrier='AC_grid_feed-in')
        self.network.add('Bus', name='battery_bus', carrier='DC_BAT')
        self.network.add('Bus', name='natural_gas_bus', carrier='gas')
        self.network.add('Bus', name='district_heating_bus', carrier='heat_DH')     

    def add_loads(self):
        """
        Adds electrical and thermal loads to the PyPSA network.

        This method specifies the characteristics of loads based on the prepared load shape data.
        """
        self.network.add('Load', 'electrical_load', bus='electrical_bus', p_set=self.loadshape["electrical_baseload"]/4)
        self.network.add('Load', 'thermal_load', bus='thermal_bus', p_set=self.loadshape["thermal_energy_demand"]/4)

    def add_grid_connection(self):
        """
        Establishes a grid connection in the PyPSA network.

        This method adds a generator and links for grid demand and feed-in, facilitating interaction
        with external energy sources.
        """
        self.network.add('Generator', 'grid_generator', bus='grid_demand_bus', p_nom=self.get_param("grid_connection", "grid_connection_power")/4)
        self.network.add('Link', 'grid_demand_link', bus0='grid_demand_bus', bus1='electrical_bus', p_nom=self.get_param("grid_connection", "grid_connection_power")/4, efficiency=1, marginal_cost=self.marginal_electricity_demand)

        # Add Generators for Gas and District Heating for cost calculation
        self.network.add('Generator', 'natural_gas_generator', bus='natural_gas_bus', p_nom=1e6, marginal_cost=self.marginal_gas)
        self.network.add('Generator', 'district_heating_generator', bus='district_heating_bus', p_nom=self.get_param("district_heating", "power")/4, marginal_cost=self.marginal_district_heat)

    def add_pv_generator(self):
        """
        Adds a photovoltaic generator to the PyPSA network.

        This method configures a generator for the PV system based on the load shape and specified power capacity.
        """
        # TODO - Handle negative values
        pv_systems = self.get_param("pv_systems")
        if pv_systems is not None:
            self.network.add('Generator', 'grid_feed-in', bus='grid_feed-in_bus', p_nom=self.get_param("grid_connection", "grid_connection_power")/4, marginal_cost=self.marginal_electricity_feed_in, sign = -1)
            self.network.add('Link', 'grid_feed-in_link', bus0='electrical_bus', bus1='grid_feed-in_bus', p_nom=self.get_param("grid_connection", "grid_connection_power")/4, efficiency=1)
            for pv in pv_systems:
                self.network.add('Generator', f'{pv["name"]}_pv_generator', bus='grid_feed-in_bus', p_nom=pv["pv_power"]/4, p_max_pu=self.loadshape[f"pv_gen_{pv['name']}"]) # removed p_min_pu for better load balancing     p_min_pu=self.loadshape[f"pv_gen_{pv['name']}"],


    def add_battery_storage(self):
        """
        Integrates battery storage into the PyPSA network.

        This method adds a storage component for managing electrical energy, including charge and discharge links.
        """
        # TODO - Add marginal cost for Battery degradation
        self.network.add('Store', 'battery_storage', bus='battery_bus', e_nom=self.get_param("battery", "capacity"), e_initial=self.get_param("battery", "capacity")*self.get_param("battery", "initial_SoC") , p_nom=self.get_param("battery", "max_power")/4, standing_loss=0.000001, e_cyclic=False)
        self.network.add('Link', 'battery_charge', bus0='electrical_bus', bus1='battery_bus', p_nom=self.get_param("battery", "max_power")/4, efficiency=self.get_param("battery", "charge_efficiency"))
        self.network.add('Link', 'battery_discharge', bus0='battery_bus', bus1='electrical_bus', p_nom=self.get_param("battery", "max_power")/4, efficiency=self.get_param("battery", "discharge_efficiency"))

        #self.network.add('Store', 'battery_DRL_overflow', bus='battery_bus', e_nom=1e15, e_initial=1e10 , p_nom=1e10, marginal_cost=1e10)

    def add_thermal_storage(self):
        """
        Adds thermal storage to the PyPSA network.

        This method configures thermal energy storage based on specified parameters, ensuring efficient
        thermal energy management.
        """
        # e_min_TES for standing losses; Assumption: 18°C Room Temp
        t_min_TES = self.get_param("general", "t_0") - 18  # 18°C Room Temp
        delta_t = self.get_param("thermal_storage", "t_max_TES") - self.get_param("general", "t_0")                
        e_min_TES = (self.get_param("thermal_storage", "tes_mass") * 4.18 * t_min_TES) / 3600
        e_max_TES = (self.get_param("thermal_storage", "tes_mass") * 4.18 * (t_min_TES + delta_t)) / 3600 # 1.16 kg/l * 4.18 J/kgK * 20 K / 3600 J/kWh    18°C Room Temp
        p_nom_TES = self.get_param("thermal_storage", "heat_exchange_power")
        self.network.add('Store', 'thermal_storage', bus='thermal_bus', e_nom=e_max_TES, e_initial=e_min_TES*1.2, p_nom=p_nom_TES/4, efficiency=0.98, standing_loss=self.get_param("thermal_storage", "standing_loss"), e_min_pu= e_min_TES/e_max_TES, e_cyclic=True) # Standing loss ~ 13% per day

    def add_heat_pump(self):
        """
        Integrates a heat pump into the PyPSA network.

        This method adds a link for heat pump operation, allowing for the conversion of electrical energy 
        to thermal energy.
        """
        hp_systems = self.get_param("heat_pumps")  # Holt die Liste der Wärmepumpen

        if hp_systems is not None:
            for hp in hp_systems:
                self.network.add('Link', f'{hp["name"]}_heat_pump', bus0='electrical_bus', bus1='thermal_bus', p_nom=hp["power_el"]/4, efficiency=self.loadshape[f"hp_cop_{hp['name']}"], committable=True, p_min_pu=self.get_param("heat_pump", "p_min"))
        #self.network.add('Link', 'heat_pump', bus0='electrical_bus', bus1='thermal_bus', p_nom=self.get_param("heat_pump", "power_el")/4, efficiency=self.loadshape.hp_cop)

    def add_gas_boiler(self):
        """
        Adds a gas boiler to the PyPSA network.

        This method configures a gas boiler for thermal energy generation based on specified parameters.
        """
        self.network.add('Link', 'gas_boiler', bus0='natural_gas_bus', bus1='thermal_bus', p_nom=self.get_param("gas_boiler", "power")/4, efficiency=self.get_param("gas_boiler", "efficiency"), committable=True, p_min_pu=self.get_param("gas_boiler", "p_min"))

    def add_CHP(self):
        """
        Adds a combined heat and power (CHP) system to the PyPSA network.

        This method incorporates a generator and storage for managing both electrical and thermal output from the CHP.
        """
        self.network.add('Link', 'CHP', bus0='natural_gas_bus', bus1='thermal_bus', bus2='electrical_bus', p_nom=self.get_param("CHP", "power")/4, efficiency=self.get_param("CHP", "efficiency_th"), efficiency2=self.get_param("CHP", "efficiency_el"), committable=True, p_min_pu=self.get_param("CHP", "p_min"))

    def add_district_heating(self):
        """
        Integrates district heating components into the PyPSA network.

        This method adds thermal storage and links for managing district heating supplies and demands.
        """
        self.network.add('Link', 'district_heating', bus0='district_heating_bus', bus1='thermal_bus', p_nom=self.get_param("district_heating", "power")/4, efficiency=self.get_param("district_heating", "efficiency"), committable=True, p_min_pu=self.get_param("district_heating", "p_min"))

    def get_full_loadshape(self):
        """
        Calculates and retrieves the complete load shape for the energy system simulation.

        This method combines load shape data with the results of the simulation, including temperature
        of thermal energy storage and other relevant metrics.
        """
        # Calculate total cost
        self.calculate_cost()
        self.calculate_co2()
        # Calcualte TES Temperature
        #self.loadshape["TES_temp"] = (self.network.stores_t.e['thermal_storage'] * 3600)  / (self.get_param("thermal_storage", "tes_mass") * 4.180) + 18

        loadshape_df = self.loadshape

        dfs = [
            loadshape_df, 
            self.network.generators_t.p * 4, 
            self.network.loads_t.p * 4, 
            self.network.stores_t.p * 4, 
            self.network.links_t.p0 * 4, 
            self.network.links_t.p1 * 4, 
            self.network.stores_t.e
        ]

        suffixes = [
            "", 
            "_kW", 
            "_kW", 
            "_kW", 
            "_p0_kW", 
            "_p1_kW", 
            "_kWh"
        ]
        for df, suffix in zip(dfs, suffixes):
             df.columns = [f"{col}{suffix}" if not col.endswith(suffix) else col for col in df.columns]
            #df.columns = [f"{col}{suffix}" for col in df.columns]
        self.full_network_df = pd.concat(dfs, axis=1)

        self.start = pd.to_datetime(self.start)
        self.end = pd.to_datetime(self.end)
        self.full_network_df.index = self.datetime_range
        
    def initialize_pypsa_network(self):
        """
        Initializes the PyPSA network with defined parameters and components.

        This method sets up the network, configures the snapshots, and populates it with 
        buses, loads, and generation/storage components.
        """
        try:
            
            self.prepare_loadshape()
            self.loadshape.to_csv("./Output/loadshape_test.csv")

            # Initialize network
            self.loadshape = self.loadshape.loc[self.start:self.end]
            self.loadshape.index = range(self.num_time_steps)
            self.network = pypsa.Network() # Create network with custom attributes     override_component_attrs=override_component_attrs
            self.network.set_snapshots(range(self.num_time_steps)) # 672 15min steps = 7 days
            self.set_optimization_objective()

            # Must-Have Components for calculation without Errors
            self.add_buses()
            self.add_loads()
            self.add_grid_connection()
            #self.add_thermal_storage()
            
            # PV and Storage Components
            self.add_pv_generator()
            self.add_battery_storage()
            #self.add_BEV(V2G=False)
            #self.add_electrolyzer()
            #self.add_fuel_cell()

            # Heating Components
            self.add_heat_pump()
            #self.add_district_heating()
            #self.add_CHP()
            #self.add_gas_boiler()
            print("Initialized PyPsa network")
        
        except Exception as e:
            print(f"An error occurred during PyPSA network initialization: {e}")

    def simulate(self, DRL=True):
        """
        Executes the energy system simulation.

        This method optimizes the network based on configured parameters and retrieves the full load shape
        at the end of the simulation.
        """
        try:
            if DRL:
                self.prepare_DRL()

            # Optimize the network
            self.network.optimize()

            # Get the full load shape
            self.get_full_loadshape()
            
            self.calculate_reward()
            
        
        except Exception as e:
            self.reward = "Error"
            print(f"An error occurred during simulation: {e}")

    def prepare_DRL(self):
        """
        Prepares the network for Deep Reinforcement Learning (DRL).
        """
        self.network.add('Bus', name='DRL_overflow_bus')
        self.network.add('Store', 'DRL_overflow', bus='DRL_overflow_bus', e_nom=np.inf, e_initial=1e6)
        self.network.add('Link', 'DRL_overflow_link_bat_charge', bus0='battery_bus', bus1='DRL_overflow_bus', p_nom=1e6 , efficiency=1, marginal_cost=1e3)
        self.network.add('Link', 'DRL_overflow_link_bat_discharge', bus0='DRL_overflow_bus', bus1='battery_bus', p_nom=1e6, efficiency=1, marginal_cost=1e3)

    def calculate_reward(self):
        """
        Calculates the reward based on cost or CO₂ optimization.

        This method computes the reward based on the marginal 'cost' and any additional penalties or constraints.
        """
        grid_cost = (self.network.links_t.p0["grid_demand_link"] * self.marginal_electricity_demand).sum()
        gas_cost = (self.network.generators_t.p["natural_gas_generator"] * self.marginal_gas).sum()
        heating_cost = (self.network.generators_t.p["district_heating_generator"] * self.marginal_district_heat).sum()
        overflow_cost = (abs(self.network.stores_t.p["DRL_overflow"])).sum() * 10
        feed_in_revenue = (self.network.links_t.p0['grid_feed-in_link'] * self.marginal_electricity_feed_in).sum()

        baseline_grid_cost = (self.network.loads_t.p["electrical_load"] * self.marginal_electricity_demand).sum()
        baseline_gas_cost = gas_cost  # Placeholder
        baseline_heating_cost = heating_cost # Placeholder
        
        actual_cost = grid_cost + gas_cost + heating_cost + overflow_cost + feed_in_revenue
        baseline_cost = baseline_grid_cost + baseline_gas_cost + baseline_heating_cost 
        self.reward = 1 - (actual_cost / (baseline_cost + 1e-6))

 
        

# Example Usage
if __name__ == "__main__":

    #%% Test Cost Optimizer
    VPP = EnergySystemSimulation()
    VPP.simulate(DRL=True)
    VPP.full_network_df.to_csv("./Output/loadshape_cost.csv")    
    print(f'\n\nCost Optimizer | Total Cost: {VPP.total_cost/100:.2f} €\n\n')

    #%% Test Peak Shaving
    # # TODO - Test and validate - not working yet
    # VPP = EnergySystemSimulation()
    # VPP.add_peak_load_penalty(penalty_factor=1e6)
    # VPP.full_network_df.to_csv("./Output/loadshape_peak_shaving.csv")
    # print(f'\n\nPeak Shaving | Total Cost: {VPP.total_cost:.2f} €\n\n')


    #%% Test CO2 Optimizer
    # VPP = EnergySystemSimulation()
    # VPP.parameters["general"]["optimizer"] = "co2"    
    # VPP.initialize_pypsa_network()
    # VPP.simulate()
    # VPP.full_network_df.to_csv("./Output/loadshape_co2.csv") 
    # print(f'\n\nCO2 Optimizer | Total Emissions: {VPP.total_emissions/1000:.2f} kg CO2\n\n')
    #plt.plot(VPP.loadshape["time"], VPP.loadshape["co2_el"])
    #print(VPP.loadshape)
    #%% Test dynamic date and time handling
    # # Get current date and time and set start and end date for the simulation
    # # TODO - Date and Time handling; select by time index
    # from datetime import datetime, timedelta
    # current_datetime = datetime.now()
    # start_date = current_datetime.replace(year=2015, hour=0, minute=0, second=0, microsecond=0)
    # end_date = start_date + timedelta(days=7) - timedelta(minutes=15)
    # start_date_str = start_date.strftime("%Y-%m-%d %H:%M:%S")
    # end_date_str = end_date.strftime("%Y-%m-%d %H:%M:%S")



    #%% TODO
    # TODO - Add Function to optimize storage dimensioning
    # TODO - Add Function to create DF for WebUI
    # TODO - historische daten für dynamischen strompreis und CO2 nach Stromvertrag (Grünstrom, Graustrom, etc.)
    # TODO - Add Function to calculate CO2 emissions
    # TODO - Add Function: dynamische Netzentgelte und variable Einspeisevergütung


