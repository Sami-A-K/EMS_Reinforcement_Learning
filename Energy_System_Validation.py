"""
Info
----

"""
import pandas as pd
import numpy as np
import yaml
from Energy_District_Data import EnergyDistrictData
from Energy_District_Network import EnergyDistrictNetwork

class EnergySystemTest():

    def __init__(self, config_file="./config.yaml"):
        """
        Initializes the energy system simulation with parameters loaded from a configuration file.

        Args:
            config_file (str): The path to the YAML configuration file containing simulation parameters.
        """

        with open(config_file, "r") as file:
            
            self.config = yaml.safe_load(file)
            start_yaml = pd.Timestamp(self.config["general"]["start"])
            end_yaml = pd.Timestamp(self.config["general"]["end"])
            self.training_range = pd.date_range(start_yaml, end_yaml, freq=self.config["general"]["time_freq"])
            self.min_per_interval = self.config["general"]["timebase"]
            self.num_steps_max = 96
    
            # VPP Data init from config.yaml
            district_data = EnergyDistrictData()
            self.temperature = district_data.temperature
            self.t_min = self.temperature['temperature'].min()
            self.t_max = self.temperature['temperature'].max()
            self.electrical_demand = district_data.electrical_demand
            self.el_demand_min = self.electrical_demand.min()
            self.el_demand_max = self.electrical_demand.max()
            self.thermal_demand = district_data.thermal_demand
            self.th_demand_min = self.thermal_demand.min()
            self.th_demand_max = self.thermal_demand.max()
            self.pv_generation_data = district_data.pv_generation
            self.heat_pump_cops = district_data.heat_pump_cops

            # Pypsa Netzwork init from config.yaml
            district_pypsa = EnergyDistrictNetwork()
            self.network = district_pypsa.network
            self.buses = self.network.buses.index
            self.grid_supply = self.network.generators[self.network.generators.carrier == "grid_supply"].index
            self.grid_feed_in = self.network.generators[self.network.generators.carrier == "grid_feed_in"].index
            self.el_load = self.network.loads[self.network.loads.carrier == "el_load"].index
            self.th_load = self.network.loads[self.network.loads.carrier == "th_load"].index
            self.el_grid = self.network.links[self.network.links.carrier == "grid"].index
            self.heatpumps = self.network.links[self.network.links.carrier == "heatpump"].index
            self.pv_generators = self.network.generators[self.network.generators.carrier == "solar"].index 
            self.batteries = self.network.storage_units[self.network.storage_units.carrier == "battery"].index
            self.thermal_storages = self.network.stores[self.network.stores.carrier == "thermal_storage"].index

            self.set_pypsa_network(self.training_range)
            self.network.optimize(solver_name="gurobi")
  

    def set_pypsa_network(self, timerange):
        self.network.set_snapshots(timerange)
        for actor in self.config["actors"]:
            if f"electrical_demand_{actor.get('name')}" in self.electrical_demand.columns:
                self.network.loads_t.p_set[f"{actor.get('name')}_electrical_load"] = self.electrical_demand.loc[timerange, f"electrical_demand_{actor.get('name')}"]
            if f"thermal_demand_{actor.get('name')}" in self.thermal_demand.columns:
                self.network.loads_t.p_set[f"{actor.get('name')}_thermal_load"] = self.thermal_demand.loc[timerange, f"thermal_demand_{actor.get('name')}"]
            if f"pv_gen_{actor.get('name')}" in self.pv_generation_data.columns:
                self.network.generators_t.p_max_pu[f"{actor.get('name')}_pv"] = self.pv_generation_data.loc[timerange, f"pv_gen_{actor.get('name')}"]/actor.get('P_pv_nom')
            if f"hp_cop_{actor.get('name')}" in self.heat_pump_cops.columns:
                self.network.links_t.efficiency[f"{actor.get('name')}_heatpump"] = self.heat_pump_cops.loc[timerange, f"hp_cop_{actor.get('name')}"]

        for name in self.batteries:
                p_nom_bat = self.network.storage_units.at[name, 'p_nom']
                e_nom_bat = p_nom_bat * self.network.storage_units.at[name, 'max_hours']
                soc_fraction = 0.5#np.random.uniform(0.1, 0.9) # initialer State of Charge zwischen 10 und 90 %
                self.network.storage_units.at[name, 'state_of_charge_initial'] = e_nom_bat * soc_fraction

        for name in self.thermal_storages:
                e_nom_tes = self.network.stores.at[name, 'e_nom']
                e_min_pu = self.network.stores.at[name, 'e_min_pu']
                soc_fraction = 0.5 #np.random.uniform(e_min_pu, 0.9) # initialer State of Charge zwischen e_min_pu und 90 %
                self.network.stores.at[name, 'e_initial'] = e_nom_tes * soc_fraction
        
        #for name in self.grid_supply:
        self.network.generators.at['grid power', 'marginal_cost'] = 0.28


if __name__ == "__main__":

    env = EnergySystemTest()
    