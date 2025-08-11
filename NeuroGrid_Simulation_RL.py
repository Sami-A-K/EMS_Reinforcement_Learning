"""
Info
----

"""
import pandas as pd
import numpy as np
import yaml
from Energy_District_Data import EnergyDistrictData
from Energy_District_Network import EnergyDistrictNetwork
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DDPG, SAC, TD3, PPO, DQN, HerReplayBuffer

class EnergySystemEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config_file="./config.yaml"):
        """
        Initializes the energy system simulation with parameters loaded from a configuration file.

        Args:
            config_file (str): The path to the YAML configuration file containing simulation parameters.
        """
        # try:
        with open(config_file, "r") as file:
            
            self.config = yaml.safe_load(file)
            start_yaml = pd.Timestamp(self.config["general"]["start"])
            end_yaml = pd.Timestamp(self.config["general"]["end"])
            self.training_range = pd.date_range(start_yaml, end_yaml, freq="15min")
            self.min_per_interval = 15
            self.num_steps_max = 96
    
            # VPP Data init with with seperate class. Data files from config.yaml
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
            
            # Pypsa network init with seperate class. Network structure from config.yaml
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
            print(self.batteries)
            self.thermal_storages = self.network.stores[self.network.stores.carrier == "thermal_storage"].index
            self.set_pypsa_network(self.training_range)
            self.network.optimize(solver_name="gurobi")
            self.optimal_generator_power = self.network.generators_t.p.loc[self.training_range, 'grid power']
            self.get_controllable_components()

            # Setting action- and observation-space 
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(len(self.controllables),), dtype=np.float32)
            self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(len(self.thermal_storages) + len(self.batteries) + self.num_steps_max * 3,), dtype=np.float32)


    def reset(self, seed=None, options=None):
        self.timestep = self.random_timestep()
        self.num_steps = 1
        print("Neue Episode, beginnend ab Zeitschritt:", self.timestep)
        self.start = self.timestep
        self.end = self.timestep + (self.num_steps_max) * pd.Timedelta(minutes=self.min_per_interval)
        timerange = pd.date_range(start=self.start, end=self.end, freq=f"{self.min_per_interval}min")
        self.set_pypsa_network(timerange)

        obs = self.get_obs(self.timestep)    
        info = {}  
        return obs, info

        #optimal_generator_power = self.network.generators_t.p
        #print(self.network)
        #marginal_cost = self.network.generators.marginal_cost
        #links = self.network.links_t.p0
        #self.step()
        #agent_generator_power = self.network.generators_t.p_set
        # comparison = pd.DataFrame({
        #     "agent": agent_generator_power["grid power"],
        #     "optimal": optimal_generator_power["grid power"],
        #     "diff": (agent_generator_power["grid power"] - optimal_generator_power["grid power"]).abs()
        # })
        # print(comparison[comparison["diff"] > 1e-5])

    def step(self, action):
        terminated = False 
        truncated = False 
        dt = self.min_per_interval/60
        penalty = 0
        p_el_supply_timestep = 0

        for i, component in enumerate(self.controllables):
            if component["type"] == "heatpump":
                p_hp_nom= component["p_hp_nom"]
                p_min_pu= component["p_min_pu"]
                cop_timestep = self.network.links_t.efficiency.loc[self.timestep, component["name"]]
                p_thermal_load_timestep = self.network.loads_t.p_set.loc[self.timestep, component["connected_loads"]].sum()
                e_th_max= component["e_th_nom"]  
                e_th_min= e_th_max * self.network.stores.at[component["thermal_storage_name"], 'e_min_pu']
                if self.num_steps > 1:
                    soc_init = self.network.stores_t.e.loc[self.timestep - pd.Timedelta(minutes=self.min_per_interval), component["thermal_storage_name"]]
                else:
                    soc_init = self.network.stores.at[component["thermal_storage_name"], 'e_initial']

                # Agent setzt Leistung der Wärmepumpe
                p_hp_timestep = (0.5 * (action[i] + 1) * (1 - p_min_pu) + p_min_pu) * p_hp_nom 
                self.network.links_t.p_set.loc[self.timestep, component["name"]] = p_hp_timestep

                # Resultierender Speicherstand:
                p_thermal_storage_timestep = p_hp_timestep * cop_timestep - p_thermal_load_timestep
                soc_timestep = soc_init + p_thermal_storage_timestep * dt 
                soc_timestep_clipped = np.clip(soc_timestep, e_th_min, e_th_max)
                self.network.stores_t.e.loc[self.timestep, component["thermal_storage_name"]] = soc_timestep_clipped
                
                penalty += abs(soc_timestep_clipped - soc_timestep)
                p_el_supply_timestep += p_hp_timestep

            elif component["type"] == "battery":
                p_bat_nom = component["p_nom"]
                e_bat_nom = component["e_nom"]
                eff_bat_charge = component["eff_charge"]
                eff_bat_discharge = component["eff_discharge"]
                p_el_load_timestep = self.network.loads_t.p_set.loc[self.timestep, component["connected_loads"]].sum()

                if self.num_steps > 1:
                    soc_init = self.network.storage_units_t.state_of_charge_set.loc[self.timestep - pd.Timedelta(minutes=self.min_per_interval), component["name"]]
                else:
                    soc_init = self.network.storage_units.at[component["name"], 'state_of_charge_initial']

                # Agent setzt Leistung der Batterie
                p_bat_timestep = action[i] * p_bat_nom
                if p_bat_timestep > 0:
                    soc_timestep = soc_init + p_bat_timestep * dt * eff_bat_discharge
                else:
                    soc_timestep = soc_init + p_bat_timestep * dt * eff_bat_charge
                soc_timestep_clipped = np.clip(soc_timestep, 0, e_bat_nom)
                self.network.storage_units_t.state_of_charge_set.loc[self.timestep, component["name"]] = soc_timestep_clipped
                penalty += abs(soc_timestep_clipped - soc_timestep)
                p_el_supply_timestep += p_el_load_timestep - p_bat_timestep 


        self.network.generators_t.p_set.loc[self.timestep, 'grid power'] = p_el_supply_timestep
        p_opt_supply = self.optimal_generator_power.loc[self.timestep]
        reward = 1 - abs(p_el_supply_timestep - p_opt_supply) / (abs(p_opt_supply) + 1e-6) - penalty

        info = {
            "iteration": self.num_steps,
            "timestep": self.timestep,
            "reward": reward,
        }

        self.num_steps += 1
        self.timestep += pd.Timedelta(minutes=self.min_per_interval)
        obs = self.get_obs(self.timestep) 
        if self.num_steps > self.num_steps_max:
            terminated = True

        return obs, reward, terminated, truncated, info


        
        # for hp_name in self.heatpumps:
        #     thermal_bus = self.network.links.at[hp_name, 'bus1']
        #     loads_at_bus = self.network.loads[self.network.loads.bus == thermal_bus].index
        #     thermal_load_timestep = self.network.loads_t.p_set.loc[self.timestep, loads_at_bus].sum()
            
        #     cop_timestep = self.network.links_t.efficiency.loc[self.timestep, hp_name]
        #     if cop_timestep is None or cop_timestep == 0 or np.isnan(cop_timestep):
        #         p_hp_timestep = 0
        #     else:
        #         p_hp_timestep = thermal_load_timestep / cop_timestep

        #     self.network.links_t.p_set.loc[self.timestep, hp_name] = p_hp_timestep
        # for pv_name in self.pv_generators:
        #     pv_bus = self.network.generators.at[pv_name, 'bus']
        #     print(pv_bus)
        #     connected_links = self.network.links.at[pv_bus, 'bus0']
        #     print(connected_links)
        #     feed_in = self.network.generators[self.network.generators.bus == pv_bus and self.network.generators.carrier == "grid_feed_in"].index
        #     #p_pv self.network.generators_t.p_set.loc[self.timestep, self.pv_generators]

        # for bat_name in self.batteries:
        #     soc_bat = self.network.storage_units_t.state_of_charge.loc[self.timestep, bat_name]

        # p_loads_sum = self.network.loads_t.p_set.loc[self.timestep, self.el_load].sum()
        # p_heatpumps_sum = self.network.links_t.p_set.loc[self.timestep, self.heatpumps].sum()
        # p_pv_sum = self.network.generators_t.p.loc[self.timestep, self.pv_generators].sum()
        # p_el_feed_in_sum = self.network.generators_t.p.loc[self.timestep, self.grid_feed_in].sum()
        # p_el_supply= p_loads_sum + p_heatpumps_sum + p_el_feed_in_sum - p_pv_sum
        # self.network.generators_t.p_set.loc[self.timestep, self.grid_supply] = p_el_supply

        # self.timestep += pd.Timedelta(minutes=self.min_per_interval)
        # self.num_steps += 1
        # truncated = self.num_steps > self.num_steps_max

        # if not truncated:
        #     self.step()

    # Hilfsfunktionen
    def get_obs(self, timestep):
        """
        Gibt den aktuellen observation space zurück.
        """    
        timerange = pd.date_range(start=timestep, end=(timestep + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)), freq="15min")
        sin_time, cos_time = self.encode_time_range(timerange)
        temp = self.temperature.loc[timerange, 'temperature'].to_numpy()
        # Skalierung der Temperatur auf [-1,1]
        temp = 2 * (temp - self.t_min) / (self.t_max - self.t_min) - 1
        # Absicherung bei Fehlerhaften Größen
        temp = np.clip(temp, -1, 1)
        socs = []

        for battery in self.batteries:
            if self.num_steps > 1:
                soc = self.network.storage_units_t.state_of_charge_set.loc[self.timestep - pd.Timedelta(minutes=self.min_per_interval), battery]
            else:
                soc = self.network.storage_units.at[battery, "state_of_charge_initial"]
            e_nom = self.network.storage_units.at[battery, "max_hours"] * self.network.storage_units.at[battery, "p_nom"]
            socs.append(soc / e_nom)

        for th_storage in self.thermal_storages:
            if self.num_steps > 1:
                    soc = self.network.stores_t.e.loc[self.timestep - pd.Timedelta(minutes=self.min_per_interval), th_storage]
            else:
                    soc = self.network.stores.at[th_storage, 'e_initial']
            e_nom = self.network.stores.at[th_storage, "e_nom"]
            socs.append(soc / e_nom)

        socs = np.array(socs)
        obs = np.concatenate([socs, sin_time, cos_time, temp])
        return obs

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
                soc_fraction = np.random.uniform(0.1, 0.9) # initialer State of Charge zwischen 10 und 90 %
                self.network.storage_units.at[name, 'state_of_charge_initial'] = e_nom_bat * soc_fraction

        for name in self.thermal_storages:
                e_nom_tes = self.network.stores.at[name, 'e_nom']
                e_min_pu = self.network.stores.at[name, 'e_min_pu']
                soc_fraction = np.random.uniform(e_min_pu, 0.9) 
                self.network.stores.at[name, 'e_initial'] = e_nom_tes * soc_fraction
        
        #for name in self.grid_supply:
        self.network.generators.at['grid power', 'marginal_cost'] = 0.28

    def get_controllable_components(self):
        self.controllables = []

        for actor in self.config["actors"]:
            actor_name = actor.get("name")

            if actor.get("E_bat_nom") and actor.get("P_bat_nom"):
                self.controllables.append({
                    "name": f"{actor_name}_battery",
                    "type": "battery",
                    "p_nom": actor["P_bat_nom"],
                    "e_nom": actor["E_bat_nom"],
                    "eff_charge": 0.98,
                    "eff_discharge": 0.98,  #TO-DO: Value from config
                    "connected_loads": self.get_connected_el_loads(actor_name)
                })

            if actor.get("P_hp_nom") and actor.get("E_th_nom"):
                self.controllables.append({
                    "name": f"{actor_name}_heatpump",
                    "type": "heatpump",
                    "p_hp_nom": actor["P_hp_nom"],
                    "p_min_pu": 0.25, #TO-DO: Value from config
                    "connected_loads": self.get_connected_th_loads(actor_name),
                    "thermal_storage_name": f"{actor_name}_thermal_storage",
                    "e_th_nom": actor["E_th_nom"]
                })
        print(self.controllables)
            # if actor.get("P_pv_nom"):
            #     self.controllables.append({
            #         "name": f"{actor_name}_pv",
            #         "type": "pv",
            #         "p_nom": actor["P_pv_nom"],
            #         "grid_feed_in": f"{actor_name}_infeed_con",
            #         "pv_link": f"{actor.get('name')}_pv_link"
            #     })


    def get_connected_el_loads(self, actor_name):
        connected_loads = []
        for actor in self.config["actors"]:
            if actor_name == actor.get("name") and actor.get("yearly_electrical_energy_demand") is not None:
                connected_loads.append(f"{actor.get('name')}_electrical_load")
            elif actor_name == actor.get("quarter_grid") and actor.get("yearly_electrical_energy_demand") is not None:
                connected_loads.append(f"{actor.get('name')}_electrical_load")
        return connected_loads
    
    def get_connected_th_loads(self, actor_name):
        connected_loads = []
        for actor in self.config["actors"]:
            if actor_name == actor.get("name") and actor.get("yearly_thermal_energy_demand") is not None:
                connected_loads.append(f"{actor.get('name')}_thermal_load")
            elif actor_name == actor.get("heating") and actor.get("yearly_thermal_energy_demand") is not None:
                connected_loads.append(f"{actor.get('name')}_thermal_load")
        return connected_loads
    
    def random_timestep(self):
        """
        Wählt zufällig einen Tageszeitraum (96 x 15min) aus dem Trainingszeitraum aus.
        """
        max_start = self.training_range[-1] - pd.Timedelta(days=2)
        possible_starts = self.training_range[(self.training_range >= self.training_range[0]) & (self.training_range <= max_start)]

        start = np.random.choice(possible_starts)
        start = pd.Timestamp(start) 
        return start

    def encode_time_range(self, timerange):
        seconds_in_day = 24 * 60 * 60
        seconds = timerange.hour * 3600 + timerange.minute * 60
        sin_time = np.sin(2 * np.pi * seconds / seconds_in_day)
        cos_time = np.cos(2 * np.pi * seconds / seconds_in_day)
        return sin_time, cos_time



def train_simple_agent(env, total_timesteps=10000):
    """
    Train a simple PPO agent in the given environment.

    Args:
        env: Dein Gym-Environment
        total_timesteps: Anzahl Trainingsschritte
    
    Returns:
        model: Das trainierte Modell
    """
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    return model


if __name__ == "__main__":

    env = EnergySystemEnvironment()
    # print("Action space shape:", env.action_space.shape)
    # print("Observation space shape:", env.observation_space.shape)

    model = train_simple_agent(env, total_timesteps=100000)
    obs, _ = env.reset()
    terminated = False
    truncated = False
    total_reward = 0

    episode_data = []

    while not (terminated or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        episode_data.append({
            "action": action,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
        })

    print("Testlauf total reward:", total_reward)
    df = pd.DataFrame(episode_data)
    df.to_csv("testlauf_episode.csv", index=False)