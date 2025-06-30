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
import gymnasium as gym
from gymnasium import spaces
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import DDPG, SAC, TD3, PPO, DQN, HerReplayBuffer
from sb3_contrib import TQC, RecurrentPPO
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
import joblib
import logging
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm


class EnergySystemSimulation(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, config_file="./config.yaml"):
        """
        Initializes the energy system simulation with parameters loaded from a configuration file.

        Args:
            config_file (str): The path to the YAML configuration file containing simulation parameters.
        """
        try:
            with open(config_file, "r") as file:
                self.config = yaml.safe_load(file)
                start_yaml = pd.Timestamp(self.config["general"]["start"])
                end_yaml = pd.Timestamp(self.config["general"]["end"])
                self.init_district_data()
                self.training_range = pd.date_range(start_yaml, end_yaml, freq="15min")
                self.max_controllables = 10     # Maximum Number of controlled components in the system. If a network has less it gets filled with 0 
                self.max_storage_units = 15     # Maximum Number of storage units in the system. If a network has less it gets filled with 0 
                self.max_steps = 10000
                self.timesteps = 96
                # Action-Space definieren
                self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.max_controllables * self.timesteps,), dtype=np.float32)
                # # Observation-Space konfigurieren
                self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.max_storage_units + self.timesteps * 3,), dtype=np.float32)

        except Exception as e:
            print(f"An error occurred during initialization: {e}")

    def reset(self, seed=None, options=None):
        self.init_pypsa_network()
        self.set_pypsa_network(self.training_range)
        self.network.optimize(solver_name="gurobi")
        self.optimal_generator_power = self.network.generators_t.p.copy()
        self.optimal_marginal_cost = self.network.generators.marginal_cost.copy()
        self.optimal_storage_soc = self.network.stores_t.e.copy()

        self.get_controllable_components()
        self.num_steps = 0
        self.random_timeframe()
        return self.get_obs(self.timerange), {}
        
    def step(self, action):
        self.random_timeframe()
        self.set_pypsa_network(self.timerange)
        obs = self.get_obs(self.timerange)

        start_soc = self.optimal_storage_soc.loc[self.timestep]
        for store_name, soc in start_soc.items():
            if store_name in self.network.stores.index:
                self.network.stores.at[store_name, 'e_init'] = soc
        self.set_controllable_components(action)
        self.network.optimize(solver_name="gurobi")

        gen_p_agent = self.network.generators_t.p.loc[self.timerange] 
        gen_cost = self.network.generators.marginal_cost

        agent_cost = (gen_p_agent * gen_cost).sum(axis=1).sum()

        gen_p_opt = self.optimal_generator_power.loc[self.timerange]
        optimal_cost = (gen_p_opt * gen_cost).sum(axis=1).sum()

        if np.isnan(agent_cost) or np.isinf(agent_cost) or agent_cost == 0:
            reward = -10
        else:
            reward = 1 - (agent_cost - optimal_cost) / (optimal_cost + 1e-6)
            reward = np.clip(reward, -10, 2)

        self.num_steps += 1
        truncated = self.num_steps >= self.max_steps
        terminated = False
        info = {
            "iteration": self.num_steps,
            "timestep": self.timestep,
            "reward": reward,
        }
        print(info)
        return obs, reward, terminated, truncated, info


    # Hilfsfunktionen
    def get_obs(self, timerange):
        """
        Gibt den aktuellen observation space zurück.
        """    
        sin_time, cos_time = self.encode_time_range(timerange)

        t_min = self.temperature['temperature'].min()
        t_max = self.temperature['temperature'].max()
        temp = self.temperature.loc[timerange, 'temperature']
        temp = temp.to_numpy()
        temp = 2 * (temp - t_min) / (t_max - t_min + 1e-6) - 1
        temp = np.clip(temp, -1, 1)
        socs = []
        for actor in self.config["actors"]:
            battery_name = f"{actor['name']}_battery"
            if battery_name in self.network.stores_t.e.columns:
                soc = self.network.stores_t.e.loc[self.timestep, battery_name]
                e_nom = self.network.stores.at[battery_name, "e_nom"]
                socs.append(soc / e_nom)

            thermal_name = f"{actor['name']}_thermal_storage"
            if thermal_name in self.network.stores_t.e.columns:
                soc = self.network.stores_t.e.loc[self.timestep, thermal_name]
                e_nom = self.network.stores.at[thermal_name, "e_nom"]
                socs.append(soc / e_nom)

        while len(socs) < self.max_storage_units:
            socs.append(0.0)
        socs = np.array(socs)

        obs = np.concatenate([socs, sin_time, cos_time, temp])
        return obs

    def get_controllable_components(self):
        self.controllables = []

        for actor in self.config["actors"]:
            actor_name = actor.get("name")

            battery_name = f"{actor_name}_battery"
            if battery_name in self.network.storage_units.index:
                self.controllables.append({
                    "name": battery_name,
                    "type": "storage_unit",
                    "p_min": actor["P_bat_nom"],
                    "p_max": actor["P_bat_nom"],
                    "component": self.network.storage_units.loc[battery_name]
                })

            hp_name = f"{actor_name}_heatpump"
            if hp_name in self.network.links.index:
                self.controllables.append({
                    "name": hp_name,
                    "type": "link",
                    "p_min": 0,
                    "p_max": actor["P_hp_nom"],
                    "component": self.network.links.loc[hp_name]
                })

    def set_controllable_components(self, action):
        action = action.reshape(self.max_controllables, self.timesteps)
        max_controllables = action.shape[0]
        num_controllables = len(self.controllables)

        assert action.shape[1] == 96, "Expected 96 time steps in action"
        assert max_controllables >= num_controllables, "Action has weniger Komponenten als Steuerbare"

        for i, component in enumerate(self.controllables):
            name = component["name"]
            values = action[i]  # 96 Zeitschritte

            if component["type"] == "storage_unit":
                p_nom = self.network.storage_units.at[name, "p_nom"]
                scaled_values = np.clip(values, -1, 1) * p_nom
                self.network.storage_units_t.p_set.loc[self.timerange, name] = scaled_values

            elif component["type"] == "link":
                p_nom = self.network.links.at[name, "p_nom"]
                scaled_values = np.clip(values, 0, 1) * p_nom
                self.network.links_t.p_set.loc[self.timerange, name] = scaled_values

    def init_district_data(self):
        district_data = EnergyDistrictData()
        self.temperature = district_data.temperature
        self.electrical_demand = district_data.electrical_demand
        self.thermal_demand = district_data.thermal_demand
        self.pv_generation = district_data.pv_generation
        self.heat_pump_cops = district_data.heat_pump_cops

    def init_pypsa_network(self):
        district_pypsa = EnergyDistrictNetwork()
        self.network = district_pypsa.network 

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

    # def reset_pypsa_network(self):
    #     for actor in self.config["actors"]:
    #         name = actor.get("name")
    #         # Elektrische Last
    #         load_name = f"{name}_electrical_load"
    #         if load_name in self.network.loads_t.p_set.columns:
    #             self.network.loads_t.p_set[load_name] = 0.0
    #         # Thermische Last
    #         tload_name = f"{name}_thermal_load"
    #         if tload_name in self.network.loads_t.p_set.columns:
    #             self.network.loads_t.p_set[tload_name] = 0.0
    #         # PV-Generator
    #         pv_name = f"{name}_pv"
    #         if pv_name in self.network.generators_t.p_set.columns:
    #             self.network.generators_t.p_set[pv_name] = 0.0
    #         # COP der Wärmepumpe
    #         hp_name = f"{name}_heatpump"
    #         if hp_name in self.network.links_t.efficiency.index:
    #             self.network.links_t.at[hp_name, "efficiency"] = 1.0  
    #         bat_name = f"{name}_battery"
    #         if bat_name in self.network.stores_t.e.index:
    #             self.network.storage_units.at[bat_name, "state_of_charge_initial"] = 0.5

    def random_timeframe(self):
        """
        Wählt zufällig einen Tageszeitraum (96 x 15min) aus dem Trainingszeitraum aus.
        """
        max_start = self.training_range[-1] - pd.Timedelta(days=1)

        possible_starts = self.training_range[(self.training_range >= self.training_range[0]) & (self.training_range <= max_start)]
        possible_starts = possible_starts[possible_starts.hour == 0]
        start = np.random.choice(possible_starts)
        start = pd.Timestamp(start) 
        end = start + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)
        self.timestep = start
        self.timerange = pd.date_range(start=start, end=end, freq="15min")
        return self.timerange


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

import pandas as pd

if __name__ == "__main__":

    env = EnergySystemSimulation()
    print("Action space shape:", env.action_space.shape)
    print("Observation space shape:", env.observation_space.shape)

    model = train_simple_agent(env, total_timesteps=10000)
    
    obs = env.reset()
    done = False
    total_reward = 0

    episode_data = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        episode_data.append({
            "observation": obs,
            "action": action,
            "reward": reward,
            "done": done,
            "info": info,
        })

    print("Testlauf total reward:", total_reward)
    df = pd.DataFrame(episode_data)
    df.to_csv("testlauf_episode.csv", index=False)

