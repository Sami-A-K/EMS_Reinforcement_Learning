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
        }
        self.network = pypsa.Network()

        self.network.add('Carrier', name='AC')
        self.network.add('Carrier', name='heat')

        self.network.add('Bus', name='grid_connection', v_nom=230, carrier='AC')
        self.network.add('Generator', name='grid power', bus='grid_connection', p_nom = np.inf, control='Slack', carrier='AC')
        for actor in self.config["actors"]:
            self.add_actor(actor)
        
    def add_actor(self, actor):
        if actor.get("yearly_electrical_energy_demand"):
            self.network.add("Bus", name=f"{actor.get('name')}_electrical_bus", v_nom=230, carrier="AC")
            self.network.add("Load", name=f"{actor.get('name')}_electrical_load", bus=f"{actor.get('name')}_electrical_bus", carrier="el_load")
            self.network.add("Line", name=f"{actor.get('name')}_service_line", bus0='grid_connection', bus1=f"{actor.get('name')}_electrical_bus", s_nom = np.inf, r = 0.001, x = 0.01, carrier='AC')
        
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
                self.network.add("Load", name=f"{actor.get('name')}_thermal_load", bus=f"{actor.get('name')}_thermal_bus", carrier="heat")
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
        self.network.add('Generator', name=f"{actor.get('name')}_pv", bus=f"{actor.get('name')}_electrical_bus", p_nom=actor.get('P_pv_nom'), control='PQ', carrier="pv") 

    def add_battery_storage(self, actor, bat_config):
        """
        Integrates battery storage into the PyPSA network.

        This method adds a storage component for managing electrical energy, including charge and discharge links.
        """
        if f"{actor.get('name')}_electrical_bus" not in self.network.buses.index:
            self.network.add("Bus", name=f"{actor.get('name')}_electrical_bus", carrier="AC")

        max_hours_bat = actor.get("E_bat_nom")/actor.get("P_bat_nom")
        self.network.add('StorageUnit', name=f"{actor.get('name')}_battery", bus=f"{actor.get('name')}_electrical_bus", p_nom=actor.get("P_bat_nom"), max_hours=max_hours_bat, efficiency_store=bat_config.get('charge_efficiency'), efficiency_dispatch=bat_config.get('discharge_efficiency'), carrier="AC")
        
    def add_quarter_grid(self, actor_source, actor_sink):
        """
        Adds electricity connection between actors to the PyPSA network.

        This method configures an electricity line between an actor with electricity generation and an actor with electricity demand.
        """
        self.network.add('Link', name=f"{actor_sink.get('name')}_local_grid", bus0=f"{actor_source.get('name')}_electrical_bus", bus1=f"{actor_sink.get('name')}_electrical_bus", p_nom=np.inf, efficiency=1, carrier='AC')

    def add_heat_pump(self, actor, hp_config):
        """
        Integrates a heat pump into the PyPSA network.

        This method adds a link for heat pump operation, allowing for the conversion of electrical energy 
        to thermal energy.
        """
        #TO-DO: p_min_pu from config
        self.network.add("Generator", name=f"{actor.get('name')}_heatpump_el", bus=f"{actor.get('name')}_electrical_bus", p_nom=actor.get("P_hp_nom"), sign=-1, carrier="hp")
        self.network.add('Generator', name=f"{actor.get('name')}_heatpump_th", bus=f"{actor.get('name')}_thermal_bus", carrier="hp") 
        #self.network.add("Link", name=f"{actor.get('name')}_heatpump", bus0=f"{actor.get('name')}_electrical_bus", bus1=f"{actor.get('name')}_thermal_bus", p_nom=actor.get("P_hp_nom"), p_nom_min=0.25*actor.get("P_hp_nom"), carrier="heat")         
    
    def add_local_heating(self, actor_source, actor_sink, lh_config):
        """
        Adds local heating to the PyPSA network.

        This method configures a local heating line between an actor with heat generation and an actor with heat demand.
        """
        if f"{actor_source.get('name')}_thermal_bus" not in self.network.buses.index:
            self.add_actor(actor_source)
        self.network.add('Link', name=f"{actor_sink.get('name')}_local_heat", bus0=f"{actor_source.get('name')}_thermal_bus", bus1=f"{actor_sink.get('name')}_thermal_bus", p_nom=np.inf, efficiency=lh_config.get("efficiency"), p_min_pu=lh_config.get("p_min"), carrier="heat")

    def add_thermal_storage(self, actor, ths_config):
        """
        Adds thermal storage to the PyPSA network.

        This method configures thermal energy storage based on specified parameters, ensuring efficient
        thermal energy management.
        """
        # Eventuell cyclical = true
        e_max = actor.get("E_th_nom")

        ths_system = ths_config.get(actor.get("heating"), {})
        standing_loss_ths = ths_config.get("standing_loss_per_day")/(24*4) # Verlust pro 15 Minuten
        t_room = ths_config.get("t_room", 18)

        t_min = ths_system.get("t_min", 30)
        t_max = ths_system.get("t_max", 55)

        e_min_pu = (t_min - t_room) / (t_max - t_room)
        e_ths = e_max * (1 - e_min_pu) # Usable capacity of thermal storage
        max_hours_ths = 15/60 #TO-DO: Inverval from config
        p_ths = e_ths/max_hours_ths
        self.network.add('StorageUnit', name=f"{actor.get('name')}_thermal_storage", bus=f"{actor.get('name')}_thermal_bus", p_nom=p_ths, max_hours=max_hours_ths, standing_loss=standing_loss_ths, carrier="heat")
        