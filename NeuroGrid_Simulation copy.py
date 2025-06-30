"""
Info
----

"""
import pandas as pd
import numpy as np
import yaml
import pypsa
import networkx as nx
import matplotlib.pyplot as plt
from vpplib.environment import Environment
from vpplib.user_profile import UserProfile
from vpplib.photovoltaic import Photovoltaic
from vpplib.heat_pump import HeatPump


class EnergySystemSimulation:
    
    def __init__(self, config_file="./config.yaml"):
        """
        Initializes the energy system simulation with parameters loaded from a configuration file.

        Args:
            config_file (str): The path to the YAML configuration file containing simulation parameters.
        """
        
        try:
            with open(config_file, "r") as file:
                self.config = yaml.safe_load(file)
                self.actor_lookup = {actor["name"]: actor for actor in self.config["actors"]} 
            self.init_timeframe()
            self.initialize_vpplib_env(self.config["general"])
            self.initialize_pypsa_network()
        except Exception as e:
            print(f"An error occurred during initialization: {e}")

    def init_timeframe(self):
        """
        Initializes the simulation timeframe based on the specified start and end dates.

        Args:
            start (str): The start date for the simulation in the format "YYYY-MM-DD".
            end (str): The end date for the simulation in the format "YYYY-MM-DD".
        """
        self.start = pd.Timestamp(self.config["general"]["start"])
        self.end = pd.Timestamp(self.config["general"]["end"])

        self.datetime_range = pd.date_range(start=self.start, end=self.end, freq="15min")
        self.num_time_steps = len(self.datetime_range)

    def random_timeframe(self):
        """
        Initializes the simulation timeframe based on the specified start and end dates.

        Args:
            start (str): The start date for the simulation in the format "YYYY-MM-DD".
            end (str): The end date for the simulation in the format "YYYY-MM-DD".
        """
        start_yaml = pd.Timestamp(self.config["general"]["start"])
        end_yaml = pd.Timestamp(self.config["general"]["end"])

        range_of_starts = pd.date_range(start_yaml, end_yaml - pd.Timedelta(days=7), freq="D")
        random_start = np.random.choice(range_of_starts)

        self.start = pd.Timestamp(random_start).replace(hour=0, minute=0)
        self.end = self.start + pd.Timedelta(days=7) - pd.Timedelta(minutes=15)
        self.datetime_range = pd.date_range(start=self.start, end=self.end, freq="15min")
        self.num_time_steps = len(self.datetime_range)
        self.week_id = self.start.strftime("KW%U_%Y")

        self.datetime_range = pd.date_range(start=self.start, end=self.end, freq="15min")
        self.num_time_steps = len(self.datetime_range)

    def initialize_vpplib_env(self, general_config):
        """
        Initializes the VPP Lib components of the energy system to prepare the load shape and obtain COP values.

        This method sets up the environment, user profile, photovoltaic system, and heat pump
        components using parameters retrieved from the configuration file.
        """
        # Initialize Environment
        self.VppLib_env = Environment(
            timebase= general_config.get("timebase"),
            start=self.start.strftime("%Y-%m-%d %H:%M:%S"), # "2015-01-01 00:00:00",  
            end=self.end.strftime("%Y-%m-%d %H:%M:%S"), # "2015-12-31 23:45:00",       
            year= general_config.get("year"),
            time_freq=general_config.get("time_freq"),
        )
        # CSV einlesen, erste Spalte ignorieren
        self.baseload = pd.read_csv(
            general_config.get("slp_data_file"),
            header=0,        # keine Kopfzeile verwenden
            usecols=range(1, 12)  # nur Spalten 1–11 behalten (ohne Zeit)
        )

        # neuen Zeitindex mit 15-Min-Auflösung über 2015 erzeugen
        date_index = pd.date_range(
            start="2015-01-01 00:00",
            end="2015-12-31 23:45",
            freq="15min"
        )
        
        # Index setzen
        self.baseload.index = date_index
        self.baseload = self.baseload.loc[self.start:self.end]
        self.baseload.reset_index(drop=True, inplace=True)

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

        print(self.electrical_demand.index)
        print(self.thermal_demand.index)
        print(self.pv_generation.index)
        print(self.heat_pump_cops.index)

    def get_electrical_demand(self, actor):
        """
        Retrieves the electrical demand data for the energy system simulation.

        This method calculates the electrical demand based on SLP data.
        """ 
        electrical_demand = self.baseload[actor.get("SLP_type")] / 1e6 * actor.get("yearly_electrical_energy_demand") # convert from 1000 MWh to 1 kWh and scale with yearly demand
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
        thermal_energy_demand = user_profile.get_thermal_energy_demand()
        thermal_energy_demand = thermal_energy_demand.loc[self.start:self.end]
        thermal_energy_demand = thermal_energy_demand.reset_index(drop=True)
        self.thermal_demand[f"thermal_demand_{actor.get('name')}"] = thermal_energy_demand

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
        heatpump_cop =  heatpump.timeseries.cop
        heatpump_cop = heatpump_cop.reset_index(drop=True)
        self.heat_pump_cops[f"hp_cop_{actor.get('name')}"] = heatpump_cop

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
        pv_generation = pv_generation.reset_index(drop=True)
        pv_generation = pv_generation.clip(lower=0)
        self.pv_generation[f"pv_gen_{actor.get('name')}"] = pv_generation

    def initialize_pypsa_network(self):
        self.heating_handler = {
            "HP_Air": lambda: self.add_heat_pump(actor, self.config["heat_pumps"]),
            "HP_Ground": lambda: self.add_heat_pump(actor, self.config["heat_pumps"]),
            "district": lambda: self.add_district_heating(actor, self.config["district_heating"]),  
            "gas": lambda: self.add_gas_heating(actor, self.config["gas_heating"])    
        }
        self.network = pypsa.Network()
        self.network.set_snapshots(range(self.num_time_steps))
        print(self.network.snapshots)
        self.network.add('Carrier', name='AC')
        self.network.add('Carrier', name='heat')
        self.network.add('Carrier', name='gas')

        self.network.add('Bus', name='grid_demand_bus', carrier='AC')
        self.network.add('Generator', name='grid power', bus='grid_demand_bus', carrier="AC", marginal_cost=0.3, p_nom = np.inf)

        for actor in self.config["actors"]:
            self.add_actor(actor)
        self.network.optimize(solver_name="gurobi")

    def add_actor(self, actor):
        if actor.get("yearly_electrical_energy_demand"):
            self.network.add("Bus", name=f"{actor.get('name')}_electrical_bus", carrier="AC")
            self.network.add("Load", name=f"{actor.get('name')}_electrical_load", bus=f"{actor.get('name')}_electrical_bus", p_set=self.electrical_demand[f"electrical_demand_{actor.get('name')}"]/4)
            self.network.add("Link", name=f"{actor.get('name')}_electrical_demand_link", bus0='grid_demand_bus', bus1=f"{actor.get('name')}_electrical_bus", p_nom = np.inf, efficiency=1)
        
        if actor.get("P_pv_nom"):
            self.add_pv_generator(actor,self.config["pv_systems"])
        if actor.get("E_bat_nom") and actor.get("P_bat_nom"):
            self.add_battery_storage(actor,self.config["battery"])
        if actor.get("quarter_grid") in self.actor_lookup:
            electricity_source_actor = self.actor_lookup[actor.get("quarter_grid")]
            self.add_quarter_grid(electricity_source_actor, actor)

        if actor.get("heating"):
            self.network.add("Bus", name=f"{actor.get('name')}_thermal_bus", carrier="heat")
            if actor.get("yearly_thermal_energy_demand"):
                self.network.add("Load", name=f"{actor.get('name')}_thermal_load", bus=f"{actor.get('name')}_thermal_bus", p_set=self.thermal_demand[f"thermal_demand_{actor.get('name')}"]/4)
            if actor.get("heating") in self.heating_handler:
                self.heating_handler[actor.get("heating")]() 
            elif actor.get("heating") in self.actor_lookup:
                heat_source_actor = self.actor_lookup[actor.get("heating")]
                self.add_local_heating(heat_source_actor, actor, self.config["local_heating"])
            else:
                raise ValueError(f"actor {actor.get('name')} has unknown heating type: {actor.get('heating')}")
            if actor.get("thermal_storage"):
                self.add_thermal_storage(actor,self.config["thermal_storage"])
        

    def add_pv_generator(self, actor, pv_config):
        """
        Adds a photovoltaic generator to the PyPSA network.

        This method configures a generator for the PV system based on the load shape and specified power capacity.
        """
        if f"{actor.get('name')}_electrical_bus" not in self.network.buses.index:
            self.network.add("Bus", name=f"{actor.get('name')}_electrical_bus", carrier="AC")
        P_pv_nom = actor.get('P_pv_nom')

        self.network.add('Bus', name=f"{actor.get('name')}_infeed_con", carrier='AC')
        self.network.add('Generator', name=f"{actor.get('name')}_grid_feed_in", bus=f"{actor.get('name')}_infeed_con", sign=-1, marginal_cost=-0.08, p_nom = np.inf)
        self.network.add('Generator', name=f"{actor.get('name')}_pv", bus=f"{actor.get('name')}_infeed_con", marginal_cost=0, p_nom = P_pv_nom, p_max_pu=self.pv_generation[f"pv_gen_{actor.get('name')}"]/P_pv_nom)

        self.network.add('Link', name=f"{actor.get('name')}_pv_link", bus0=f"{actor.get('name')}_infeed_con", bus1=f"{actor.get('name')}_electrical_bus", p_nom=P_pv_nom, efficiency=1)

    def add_battery_storage(self, actor, bat_config):
        """
        Integrates battery storage into the PyPSA network.

        This method adds a storage component for managing electrical energy, including charge and discharge links.
        """
        # TODO - Add marginal cost for Battery degradation  
        if f"{actor.get('name')}_electrical_bus" not in self.network.buses.index:
            self.network.add("Bus", name=f"{actor.get('name')}_electrical_bus", carrier="AC")
        max_hours_bat = actor.get("E_bat_nom")/actor.get("P_bat_nom")
        self.network.add('StorageUnit', name=f"{actor.get('name')}_battery", bus=f"{actor.get('name')}_electrical_bus", p_nom=actor.get("P_bat_nom"), max_hours=max_hours_bat, efficiency_store=bat_config.get('charge_efficiency'), efficiency_dispatch=bat_config.get('discharge_efficiency'))
        
    def add_quarter_grid(self, actor_source, actor_sink):
        """
        Adds electricity connection between actors to the PyPSA network.

        This method configures an electricity line between an actor with electricity generation and an actor with electricity demand.
        """
        self.network.add('Link', name=f"{actor_sink.get('name')}_local_grid", bus0=f"{actor_source.get('name')}_electrical_bus", bus1=f"{actor_sink.get('name')}_electrical_bus", p_nom=np.inf, efficiency=1)

    def add_heat_pump(self, actor, hp_config):
        """
        Integrates a heat pump into the PyPSA network.

        This method adds a link for heat pump operation, allowing for the conversion of electrical energy 
        to thermal energy.
        """
        hp_system = hp_config.get(actor.get("heating"), {})
        self.network.add("Link", name=f"{actor.get('name')}_heatpump", bus0=f"{actor.get('name')}_electrical_bus", bus1=f"{actor.get('name')}_thermal_bus", p_nom=np.inf, efficiency=self.heat_pump_cops[f"hp_cop_{actor.get('name')}"], p_min_pu=hp_system.get("p_min"))
        
    def add_district_heating(self, actor, dh_config):
        """
        Adds district heating to the PyPSA network.

        This method configures a district heating line for thermal energy generation based on specified parameters.
        """
        if "district_heating_bus" not in self.network.buses.index:
            self.network.add("Bus", name="district_heating_bus", carrier="heat")
            self.network.add("Generator", name="district_heat_supply", bus="district_heating_bus", carrier="heat", marginal_cost=dh_config.get("marginal_cost"), p_nom = np.inf)

        self.network.add("Link", name=f"{actor.get('name')}_district_heating_line", bus0="district_heating_bus", bus1=f"{actor.get('name')}_thermal_bus", p_nom=np.inf, efficiency=dh_config.get("efficiency"), p_min_pu=dh_config.get("p_min"))
    
    def add_gas_heating(self, actor, gas_config):
        """
        Adds gas heating to the PyPSA network.

        This method configures a gas boiler for thermal energy generation based on specified parameters.
        """
        if "natural_gas_bus" not in self.network.buses.index:
            self.network.add("Bus", name="natural_gas_bus", carrier="gas")
            self.network.add("Generator", name="gas_supply", bus="natural_gas_bus", carrier="gas", marginal_cost=gas_config.get("marginal_cost"), p_nom = np.inf)

        self.network.add('Link', name=f"{actor.get('name')}_gas_boiler", bus0='natural_gas_bus', bus1=f"{actor.get('name')}_thermal_bus", p_nom=np.inf, efficiency=gas_config.get("efficiency"), p_min_pu=gas_config.get("p_min"))

    def add_local_heating(self, actor_source, actor_sink, lh_config):
        """
        Adds local heating to the PyPSA network.

        This method configures a local heating line between an actor with heat generation and an actor with heat demand.
        """
        if f"{actor_source.get('name')}_thermal_bus" not in self.network.buses.index:
            self.add_actor(actor_source)
        self.network.add('Link', name=f"{actor_sink.get('name')}_local_heat", bus0=f"{actor_source.get('name')}_thermal_bus", bus1=f"{actor_sink.get('name')}_thermal_bus", p_nom=np.inf, efficiency=lh_config.get("efficiency"), p_min_pu=lh_config.get("p_min"))

    def add_thermal_storage(self, actor, ths_config):
        """
        Adds thermal storage to the PyPSA network.

        This method configures thermal energy storage based on specified parameters, ensuring efficient
        thermal energy management.
        """
        ths_system = ths_config.get(actor.get("heating"), {})
        
        c_water = ths_config.get("c_water", 4.18)
        t_room = ths_config.get("t_room", 18)
        t_min = ths_system.get("t_min", 30)
        t_max = ths_system.get("t_max", 55)

        delta_T = t_max - t_min
        usable_delta_T = max(t_min - t_room, 0)
        e_min_pu = min(usable_delta_T / delta_T, 1)
        thermal_peak_load = self.thermal_demand[f"thermal_demand_{actor.get('name')}"].max()
        mass =  thermal_peak_load * ths_system.get("ths_mass_per_kw", 100)
        e_nom = mass * c_water * delta_T / 3600  # in kWh
        p_nom = thermal_peak_load * ths_system.get("heat_exchange_power_per_kw", 1.0)
        standing_loss_per_kwh = ths_config.get("standing_loss_per_kwh", 0.0052)
        standing_loss = standing_loss_per_kwh * e_nom / 24

        self.network.add("Store", name=f"{actor.get('name')}_thermal_storage", bus=f"{actor.get('name')}_thermal_bus", e_nom=e_nom, p_nom=p_nom, e_cyclic=True, standing_loss=standing_loss, e_min_pu=e_min_pu)

if __name__ == "__main__":

    VPP = EnergySystemSimulation()
    operational_cost_1 = VPP.network.statistics()["Operational Expenditure"].sum()
    print("Wöchentliche Betriebskosten:", round(operational_cost_1,2), "€")
    
    # print("Alle Busse:\n", VPP.network.buses)
    # print("Alle Generatoren:\n", VPP.network.generators)
    # print("Alle Speicher:\n", VPP.network.stores)
    # print("Alle Links:\n", VPP.network.links)
    # print("Alle Loads:\n", VPP.network.loads)
    # with pd.ExcelWriter("netzwerk_export.xlsx") as writer:
    #     VPP.network.buses.to_excel(writer, sheet_name="Buses")
    #     VPP.network.loads.to_excel(writer, sheet_name="Loads")
    #     VPP.network.generators.to_excel(writer, sheet_name="Generators")
    #     VPP.network.links.to_excel(writer, sheet_name="Links")
    #     VPP.network.stores.to_excel(writer, sheet_name="Stores")

    G = VPP.network.graph()

    # Wähle ein Layout, z. B. Spring-Layout
    pos = nx.spring_layout(G)  # alternativ: nx.shell_layout(G), nx.kamada_kawai_layout(G)

    # Zeichne den Graph
    nx.draw(G, pos, with_labels=True, node_size=700, font_size=10)
    plt.show()