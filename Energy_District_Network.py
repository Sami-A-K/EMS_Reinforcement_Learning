"""
Info
----

"""
import pandas as pd
import numpy as np
import yaml
import pypsa

class EnergyDistrictNetwork:
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
            self.initialize_pypsa_network()
        except Exception as e:
            print(f"An error occurred during initialization: {e}")

    def initialize_pypsa_network(self):
        self.heating_handler = {
            "HP_Air": lambda: self.add_heat_pump(actor, self.config["heat_pumps"]),
            "HP_Ground": lambda: self.add_heat_pump(actor, self.config["heat_pumps"]),
            "district": lambda: self.add_district_heating(actor, self.config["district_heating"]),  
            "gas": lambda: self.add_gas_heating(actor, self.config["gas_heating"])    
        }
        self.network = pypsa.Network()

        self.network.add('Carrier', name='AC')
        self.network.add('Carrier', name='heat')
        self.network.add('Carrier', name='grid')
        self.network.add('Carrier', name='heat_grid')
        self.network.add('Carrier', name='grid_supply')
        self.network.add('Carrier', name='grid_feed_in')
        self.network.add('Carrier', name='el_load')
        self.network.add('Carrier', name='th_load')
        self.network.add('Carrier', name='solar')
        self.network.add('Carrier', name='battery')
        self.network.add('Carrier', name='heatpump')
        self.network.add('Carrier', name='thermal_storage')

        self.network.add('Bus', name='grid_demand_bus', carrier='AC')
        self.network.add('Generator', name='grid power', bus='grid_demand_bus', p_nom = np.inf, carrier='grid_supply')

        for actor in self.config["actors"]:
            self.add_actor(actor)
        
    def add_actor(self, actor):
        if actor.get("yearly_electrical_energy_demand"):
            self.network.add("Bus", name=f"{actor.get('name')}_electrical_bus", carrier="AC")
            self.network.add("Load", name=f"{actor.get('name')}_electrical_load", bus=f"{actor.get('name')}_electrical_bus", carrier="el_load")
            self.network.add("Link", name=f"{actor.get('name')}_electrical_demand_link", bus0='grid_demand_bus', bus1=f"{actor.get('name')}_electrical_bus", p_nom = np.inf, efficiency=1, carrier='grid')
        
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
                self.network.add("Load", name=f"{actor.get('name')}_thermal_load", bus=f"{actor.get('name')}_thermal_bus", carrier="th_load")
            if actor.get("heating") in self.heating_handler:
                self.heating_handler[actor.get("heating")]() 
            elif actor.get("heating") in self.actor_lookup:
                heat_source_actor = self.actor_lookup[actor.get("heating")]
                self.add_local_heating(heat_source_actor, actor, self.config["local_heating"])
            else:
                raise ValueError(f"actor {actor.get('name')} has unknown heating type: {actor.get('heating')}")
            if actor.get("E_th_nom"):
                self.add_thermal_storage(actor,self.config["thermal_storage"])
        
    def add_pv_generator(self, actor, pv_config):
        """
        Adds a photovoltaic generator to the PyPSA network.

        This method configures a generator for the PV system based on the load shape and specified power capacity.
        """
        if f"{actor.get('name')}_electrical_bus" not in self.network.buses.index:
            self.network.add("Bus", name=f"{actor.get('name')}_electrical_bus", carrier="AC")
        P_pv_nom = actor.get('P_pv_nom')
        
        self.network.add("Bus", name=f"{actor.get('name')}_infeed_con", carrier="AC")
        self.network.add('Generator', name=f"{actor.get('name')}_grid_feed_in", bus=f"{actor.get('name')}_infeed_con", sign=-1, p_nom = 1.5 * P_pv_nom, carrier="grid_feed_in")
        self.network.add('Generator', name=f"{actor.get('name')}_pv", bus=f"{actor.get('name')}_infeed_con", p_nom = P_pv_nom, control='PQ', carrier="solar")
        self.network.add('Link', name=f"{actor.get('name')}_pv_link", bus0=f"{actor.get('name')}_infeed_con", bus1=f"{actor.get('name')}_electrical_bus", p_nom=1.5 * P_pv_nom, efficiency=1, carrier='grid')

    def add_battery_storage(self, actor, bat_config):
        """
        Integrates battery storage into the PyPSA network.

        This method adds a storage component for managing electrical energy, including charge and discharge links.
        """
        # TODO - Add marginal cost for Battery degradation  
        # Eventuell cyclical = true
        if f"{actor.get('name')}_electrical_bus" not in self.network.buses.index:
            self.network.add("Bus", name=f"{actor.get('name')}_electrical_bus", carrier="AC")

        max_hours_bat = actor.get("E_bat_nom")/actor.get("P_bat_nom")
        self.network.add('StorageUnit', name=f"{actor.get('name')}_battery", bus=f"{actor.get('name')}_electrical_bus", p_nom=actor.get("P_bat_nom"), max_hours=max_hours_bat, efficiency_store=bat_config.get('charge_efficiency'), efficiency_dispatch=bat_config.get('discharge_efficiency'), carrier="battery")
        
    def add_quarter_grid(self, actor_source, actor_sink):
        """
        Adds electricity connection between actors to the PyPSA network.

        This method configures an electricity line between an actor with electricity generation and an actor with electricity demand.
        """
        self.network.add('Link', name=f"{actor_sink.get('name')}_local_grid", bus0=f"{actor_source.get('name')}_electrical_bus", bus1=f"{actor_sink.get('name')}_electrical_bus", p_nom=np.inf, efficiency=1, carrier='grid')

    def add_heat_pump(self, actor, hp_config):
        """
        Integrates a heat pump into the PyPSA network.

        This method adds a link for heat pump operation, allowing for the conversion of electrical energy 
        to thermal energy.
        """
        #TO-DO: p_min_pu from config 
        self.network.add("Link", name=f"{actor.get('name')}_heatpump", bus0=f"{actor.get('name')}_electrical_bus", bus1=f"{actor.get('name')}_thermal_bus", p_nom=actor.get("P_hp_nom"), p_nom_min=0.25*actor.get("P_hp_nom"), carrier="heatpump")         
    def add_local_heating(self, actor_source, actor_sink, lh_config):
        """
        Adds local heating to the PyPSA network.

        This method configures a local heating line between an actor with heat generation and an actor with heat demand.
        """
        if f"{actor_source.get('name')}_thermal_bus" not in self.network.buses.index:
            self.add_actor(actor_source)
        self.network.add('Link', name=f"{actor_sink.get('name')}_local_heat", bus0=f"{actor_source.get('name')}_thermal_bus", bus1=f"{actor_sink.get('name')}_thermal_bus", p_nom=np.inf, efficiency=lh_config.get("efficiency"), p_min_pu=lh_config.get("p_min"), carrier="heat_grid")

    def add_thermal_storage(self, actor, ths_config):
        """
        Adds thermal storage to the PyPSA network.

        This method configures thermal energy storage based on specified parameters, ensuring efficient
        thermal energy management.
        """
        # Eventuell cyclical = true
        e_nom = actor.get("E_th_nom")

        ths_system = ths_config.get(actor.get("heating"), {})
        standing_loss = ths_config.get("standing_loss_per_day")/(24*4) # Verlust pro 15 Minuten
        t_room = ths_config.get("t_room", 18)

        t_min = ths_system.get("t_min", 30)
        t_max = ths_system.get("t_max", 55)
        #standing_loss=standing_loss,
        e_min_pu = (t_min - t_room) / (t_max - t_room)
    
        self.network.add("Store", name=f"{actor.get('name')}_thermal_storage", bus=f"{actor.get('name')}_thermal_bus", e_nom=e_nom, e_min_pu=e_min_pu, carrier="thermal_storage")

    # def add_thermal_storage(self, actor, ths_config):
    #     """
    #     Adds thermal storage to the PyPSA network.

    #     This method configures thermal energy storage based on specified parameters, ensuring efficient
    #     thermal energy management.
    #     """
    #     ths_system = ths_config.get(actor.get("heating"), {})
        
    #     c_water = ths_config.get("c_water", 4.18)
    #     t_room = ths_config.get("t_room", 18)
    #     t_min = ths_system.get("t_min", 30)
    #     t_max = ths_system.get("t_max", 55)

    #     delta_T = t_max - t_min
    #     usable_delta_T = max(t_min - t_room, 0)
    #     e_min_pu = min(usable_delta_T / delta_T, 1)
    #     thermal_peak_load = actor.get('thermal_peak_load')
    #     mass =  thermal_peak_load * ths_system.get("ths_mass_per_kw", 100)
    #     e_nom = mass * c_water * delta_T / 3600  # in kWh
    #     p_nom = thermal_peak_load * ths_system.get("heat_exchange_power_per_kw", 1.0)
    #     standing_loss_per_kwh = ths_config.get("standing_loss_per_kwh", 0.0052)
    #     standing_loss = standing_loss_per_kwh * e_nom / 24
    #     #skalierung erst in der main klasse
    #     self.network.add("Store", name=f"{actor.get('name')}_thermal_storage", bus=f"{actor.get('name')}_thermal_bus", e_nom=e_nom, p_nom=p_nom, e_cyclic=True, standing_loss=standing_loss, e_min_pu=e_min_pu)