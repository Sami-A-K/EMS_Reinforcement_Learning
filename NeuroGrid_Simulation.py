"""
Info
----

"""
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
from Energy_District_Data import EnergyDistrictData
from Energy_District_Network import EnergyDistrictNetwork
import networkx as nx
import matplotlib.pyplot as plt

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
                self.start = pd.Timestamp(self.config["general"]["start"]) + pd.Timedelta(days=107) - pd.Timedelta(minutes=15)
                self.end = self.start + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)
                snapshots = pd.date_range(start=self.start, end=self.end, freq="15min")
                self.init_district_data()
                self.init_pypsa_network()
                self.set_pypsa_network(snapshots)
                self.solve_pypsa_network()
        except Exception as e:
            print(f"An error occurred during initialization: {e}")

    def init_district_data(self):
        district_data = EnergyDistrictData()
        self.electrical_demand = district_data.electrical_demand
        self.thermal_demand = district_data.thermal_demand
        self.pv_generation = district_data.pv_generation
        self.heat_pump_cops = district_data.heat_pump_cops

    def init_pypsa_network(self):
        district_pypsa = EnergyDistrictNetwork()
        self.network = district_pypsa.network 

    def solve_pypsa_network(self):
        self.network.optimize(solver_name="gurobi")

    def set_pypsa_network(self, timerange):
        self.network.set_snapshots(timerange)
        for actor in self.config["actors"]:
            if f"electrical_demand_{actor.get('name')}" in self.electrical_demand.columns:
                self.network.loads_t.p_set[f"{actor.get('name')}_electrical_load"] = self.electrical_demand.loc[timerange, f"electrical_demand_{actor.get('name')}"]
            if f"thermal_demand_{actor.get('name')}" in self.thermal_demand.columns:
                self.network.loads_t.p_set[f"{actor.get('name')}_thermal_load"] = self.thermal_demand.loc[timerange, f"thermal_demand_{actor.get('name')}"]
            if f"pv_gen_{actor.get('name')}" in self.pv_generation.columns:
                self.network.generators_t.p_max_pu[f"{actor.get('name')}_pv"] = self.pv_generation.loc[timerange, f"pv_gen_{actor.get('name')}"]/actor.get('P_pv_nom')
            if f"hp_cop_{actor.get('name')}" in self.heat_pump_cops.columns:
                self.network.links_t['efficiency'][f"{actor.get('name')}_heatpump"] = self.heat_pump_cops.loc[timerange, f"hp_cop_{actor.get('name')}"]

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


if __name__ == "__main__":

    VPP = EnergySystemSimulation()
    operational_cost = VPP.network.statistics()#["Operational Expenditure"]
    print(operational_cost)#"Wöchentliche Betriebskosten:", round(operational_cost_1,2), "€")

    #print(VPP.pv_generation.head(100))
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
