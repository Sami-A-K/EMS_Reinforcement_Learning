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
import logging


class EnergyDistrictEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config_file="./config.yaml"):
        """
        Initializes the energy system simulation with parameters loaded from a configuration file.

        Args:
            config_file (str): The path to the YAML configuration file containing simulation parameters.
        """
        
        with open(config_file, "r") as file:
            print("EnergyDistrictEnvironment wird initialisiert")
            
            self.config = yaml.safe_load(file)
            start_yaml = pd.Timestamp(self.config["general"]["start"])
            end_yaml = pd.Timestamp(self.config["general"]["end"])
            self.training_range = pd.date_range(start_yaml, end_yaml, freq="15min")
            self.min_per_interval = 15
            self.num_steps_max = 96
            self.dt = self.min_per_interval / 60
    
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
            self.signal_values = district_data.signal_values
            
            # Pypsa network init with seperate class. Network structure from config.yaml
            district_pypsa = EnergyDistrictNetwork()
            self.network = district_pypsa.network
            self.set_pypsa_network(self.training_range)
            logging.getLogger("pypsa.pf").setLevel(logging.WARNING)

            # Calculation of baseline costs. No p_set of storages equals no storage use. 
            self.network.lpf() 
            service_lines = self.network.lines.index[self.network.lines["bus0"] == "grid_connection"]
            self.baseline_power = self.network.lines_t.p1.loc[self.training_range, service_lines]
            # Initilasation for Agent-Training
            self.get_controllable_components()
            dim_action_space = len(self.controllables)  # Eine Aktion pro zu steuernder Komponente
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(dim_action_space,), dtype=np.float32)
            dim_observation_space = 3*96 + len(self.controllables)*2  # Zeit sin/cos und Temperatur Vorhersage für 24h + PV Vorhersage für 24h +Steuerbare Komponenten * (SOC (verknüpfter) Speicher + Last im Zeitpunkt)
            self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(dim_observation_space,), dtype=np.float32)


    def reset(self, seed=None, options=None):
        self.timestep = self.random_timestep()
        self.num_steps = 1
        print("Neue Episode, beginnend ab Zeitschritt:", self.timestep)
        self.start = self.timestep

        # Initial SOCs zufällig setzen
        for name in self.network.storage_units.index:
            p_storage = self.network.storage_units.at[name, 'p_nom']
            e_max_storage = p_storage * self.network.storage_units.at[name, 'max_hours']
            soc_fraction = np.random.uniform(0.1, 0.9)
            self.network.storage_units_t.state_of_charge_set.loc[self.timestep - pd.Timedelta(minutes=self.min_per_interval), name] = e_max_storage * soc_fraction

        obs = self.get_obs(self.timestep)
        info = {}
        return obs, info


    def step(self, action):
        terminated = False
        truncated = False

        # Actions setzen
        for i, controllable in enumerate(self.controllables):
            if controllable["type"] == "battery":
                p_bat_nom = controllable["p_bat_nom"]
                e_bat_nom = controllable["e_bat_max"]
                soc_init = self.network.storage_units_t.state_of_charge_set.loc[self.timestep - pd.Timedelta(minutes=self.min_per_interval), controllable["name"]]

                p_bat_charge_max = max(-p_bat_nom, -(e_bat_nom - soc_init) / self.dt)
                p_bat_discharge_max = min(p_bat_nom, soc_init / self.dt)

                p_bat_action = p_bat_charge_max + ((action[i] + 1) / 2) * (p_bat_discharge_max - p_bat_charge_max)
                self.network.storage_units_t.p_set.loc[self.timestep, controllable["name"]] = p_bat_action

            elif controllable["type"] == "heatpump":
                p_hp_nom= controllable["p_hp_nom"]
                p_min_pu= controllable["p_min_pu"]
                cop = max(float(self.network.generators_t.efficiency.loc[self.timestep, controllable["name_th"]]), 1e-6)
                soc_ths_max = controllable["e_ths_max"]
                soc_ths_init =  self.network.storage_units_t.state_of_charge_set.loc[self.timestep - pd.Timedelta(minutes=self.min_per_interval), controllable["con_thermal_storage"]] 
                p_thermal_load = self.network.loads_t.p_set.loc[self.timestep, controllable["con_th_load"]]

                # Technische Grenzen Wärmepumpe
                p_hp_min = p_min_pu * p_hp_nom
                p_hp_max = p_hp_nom
                # Mögliche Ein-/Ausspeicherleistung
                discharge_ths_max = soc_ths_init / self.dt
                charge_ths_max = (soc_ths_max - soc_ths_init) / self.dt
                # Bedingte Heizleistung in kW_el
                p_heat_min = max(0.0, p_thermal_load - discharge_ths_max)/cop
                p_heat_max = (p_thermal_load + charge_ths_max)/cop
                # Effektive geltende Grenzen
                if discharge_ths_max >= p_thermal_load:
                    p_hp_min_eff = 0  
                else:
                    p_hp_min_eff = max(p_hp_min, p_heat_min)
                p_hp_max_eff = min(p_hp_max, p_heat_max)

                # WP kann ausgeschaltet werden, wenn Speicher die Last deckt
                if p_heat_min == 0 and action[i] == -1:
                    p_hp_action = 0.0
                else:
                    p_hp_action = p_hp_min_eff + ((action[i] + 1) / 2) * (p_hp_max_eff - p_hp_min_eff)

                self.network.generators_t.p_set.loc[self.timestep, controllable["name_el"]] = -p_hp_action
                self.network.generators_t.p_set.loc[self.timestep, controllable["name_th"]] = p_hp_action * cop
                p_ths = p_thermal_load - p_hp_action*cop
                self.network.storage_units_t.p_set.loc[self.timestep, controllable["con_thermal_storage"]] = p_ths

                
        # Linear Power Flow Berechnung durch PyPSA. Wie viel Leistung wird aus dem Stromnetz benötigt. 
        self.network.lpf(self.timestep)
        # print("Generators:",self.network.generators_t.p.loc[self.timestep])
        # print("Storage:" ,self.network.storage_units_t.p.loc[self.timestep])
        # print("Loads:",self.network.loads_t.p.loc[self.timestep])

        # SOC Update 
        for storage in self.network.storage_units.index:
            soc_init = self.network.storage_units_t.state_of_charge_set.loc[self.timestep - pd.Timedelta(minutes=self.min_per_interval), storage]
            p_storage = self.network.storage_units_t.p_set.loc[self.timestep, storage]
            self.network.storage_units_t.state_of_charge_set.loc[self.timestep, storage] = soc_init - p_storage * self.dt

            #print("State of Charge:", self.network.storage_units_t.state_of_charge_set.loc[self.timestep, storage])
        # Reward-Berechnung
        agent_cost = 0.0
        baseline_cost = 0.0

        for actor in self.config["actors"]:
            actor_name = actor["name"]
            agent_power = self.network.lines_t.p1.loc[self.timestep, f"{actor['name']}_service_line"]
            baseline_power = self.baseline_power.loc[self.timestep, f"{actor['name']}_service_line"]
            if agent_power < 0:
                agent_cost += -1 * agent_power * 0.28 * self.dt
            else:
                agent_cost += -1 * agent_power * 0.08 * self.dt
            if baseline_power < 0:
                baseline_cost += -1 * baseline_power * 0.28 * self.dt
            else:
                baseline_cost += -1 * baseline_power * 0.08 * self.dt
        eps = 1e-3
        rel_saving = (baseline_cost - agent_cost) / (abs(baseline_cost) + eps)
        reward = float(np.clip(rel_saving, -5, 5))

        # print("Baseline ", baseline_cost)
        # print("Agent ", agent_cost)
        # print("Reward ",reward)

        if np.isnan(reward):
            raise ValueError("Reward is NaN, aborting training")

        info = {
            "iteration": self.num_steps,
            "timestep": self.timestep,
            "baseline": baseline_cost,
            "agent": agent_cost,
            "saving": baseline_cost - agent_cost,
            "reward": reward
        }

        self.num_steps += 1
        self.timestep += pd.Timedelta(minutes=self.min_per_interval)
        obs = self.get_obs(self.timestep)

        if self.num_steps > self.num_steps_max:
            terminated = True

        return obs, reward, terminated, truncated, info

    # Hilfsfunktionen
    def get_obs(self, timestep):
        timerange = pd.date_range(start=timestep,end=(timestep + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)),freq="15min")

        # Zeit-Features
        sin_time, cos_time = self.encode_time_range(timerange)

        # Temperatur-Forecast (Grundwerte aus Daten)
        temp_true = self.temperature.loc[timerange, 'temperature'].to_numpy()
        # Forecast mit Noise. +/-10 % am Ende des Horizonts, linear wachsend
        temp_forecast = self.add_forecast_noise(temp_true, deviation=0.1, mode="linear")
        # Normierung auf [-1,1]
        temp_norm = 2 * (temp_forecast - self.t_min) / (self.t_max - self.t_min) - 1
        temp_norm = np.clip(temp_norm, -1, 1)


        # Zustände im Zusammenhang mit einem Controllable, Blockweise im Obs-Space
        obs_controllables = []
        for controllable in self.controllables:
            if controllable["type"] == "battery":
                soc = (self.network.storage_units_t.state_of_charge_set.loc[timestep - pd.Timedelta(minutes=self.min_per_interval),controllable["name"]] / controllable["e_bat_max"])
                soc = np.clip(soc, 0, 1)

                # Aktuelle PV und Last, falls mit Speicher verbunden
                pv = 0.0
                if controllable.get("p_pv_nom") and controllable.get("con_pv"):
                    pv = float(self.network.generators_t.p_set.loc[timestep, controllable["con_pv"]])
                el_load = 0.0
                if controllable.get("max_el_load") and controllable.get("con_el_load"):
                    el_load = float(self.network.loads_t.p_set.loc[timestep, controllable["con_el_load"]])
                net_load = pv - el_load
                # Normierung auf [-1,1]
                if net_load >= 0:
                    net_load_norm = np.clip(net_load / controllable["p_pv_nom"], 0, 1)
                elif net_load < 0:
                    net_load_norm = np.clip(net_load / controllable["max_el_load"], -1, 0)
                else:
                    net_load_norm = 0

                block = np.array([soc, net_load_norm], dtype=np.float32)

            elif controllable["type"] == "heatpump":
                soc = (self.network.storage_units_t.state_of_charge_set.loc[timestep - pd.Timedelta(minutes=self.min_per_interval),controllable["con_thermal_storage"]] / controllable["e_ths_max"])
                soc = np.clip(soc, 0, 1)

                # Aktuelle thermische Last, falls mit WP verbunden
                th_load_norm = 0.0
                if controllable.get("max_th_load") and controllable.get("con_th_load"):
                    th_load = float(-self.network.loads_t.p_set.loc[timestep, controllable["con_th_load"]])
                    th_load_norm = np.clip(th_load / controllable["max_th_load"], -1, 0)

                block = np.array([soc, th_load_norm], dtype=np.float32)

            obs_controllables.append(block)

        obs = np.concatenate([sin_time, cos_time, temp_norm] + obs_controllables).astype(np.float32)
        return obs


    def set_pypsa_network(self, timerange):
        self.network.set_snapshots(timerange)

        for actor in self.config["actors"]:
            if f"electrical_demand_{actor.get('name')}" in self.electrical_demand.columns:
                self.network.loads_t.p_set[f"{actor.get('name')}_electrical_load"] = self.electrical_demand.loc[timerange, f"electrical_demand_{actor.get('name')}"] * self.dt # Energy in interval (kWh) * interval_minutes/hour (1/h) = power in interval (kW)
            if f"thermal_demand_{actor.get('name')}" in self.thermal_demand.columns:
                self.network.loads_t.p_set[f"{actor.get('name')}_thermal_load"] = self.thermal_demand.loc[timerange, f"thermal_demand_{actor.get('name')}"] * self.dt # Energy in interval (kWh) * interval_minutes/hour (1/h) = power in interval (kW)
            if f"hp_cop_{actor.get('name')}" in self.heat_pump_cops.columns:
                self.network.generators_t.efficiency[f"{actor.get('name')}_heatpump_th"] = self.heat_pump_cops.loc[timerange, f"hp_cop_{actor.get('name')}"]
                self.network.generators_t.p_set[f"{actor.get('name')}_heatpump_el"] = (self.thermal_demand.loc[timerange, f"thermal_demand_{actor.get('name')}"]*self.dt) /self.heat_pump_cops.loc[timerange, f"hp_cop_{actor.get('name')}"]
            if f"pv_gen_{actor.get('name')}" in self.pv_generation_data.columns:
                self.network.generators_t.p_set[f"{actor.get('name')}_pv"] = self.pv_generation_data.loc[timerange, f"pv_gen_{actor.get('name')}"] * self.dt # Energy in interval (kWh) * interval_minutes/hour (1/h) = power in interval (kW)


    def get_controllable_components(self):
        self.controllables = []

        for actor in self.config["actors"]:
            actor_name = actor["name"]

            # Batterie
            if actor.get("E_bat_nom") and actor.get("P_bat_nom"):
                bat_name = f"{actor_name}_battery"
                pv_name = f"{actor_name}_pv"
                load_name = f"{actor_name}_electrical_load"

                p_bat_nom = self.network.storage_units.at[bat_name, "p_nom"]
                e_bat_max = p_bat_nom * self.network.storage_units.at[bat_name, "max_hours"]

                pv_nom = None
                if pv_name in self.network.generators.index:
                    pv_nom = self.network.generators.at[pv_name, "p_nom"]

                max_el_load = None
                if load_name in self.network.loads.index:
                    max_el_load = self.network.loads_t.p_set[load_name].max()

                self.controllables.append({
                    "type": "battery",
                    "name": bat_name,
                    "p_bat_nom": p_bat_nom,
                    "e_bat_max": e_bat_max,
                    "con_pv": pv_name if pv_nom else None,
                    "con_el_load": load_name if max_el_load else None,
                    "p_pv_nom": pv_nom,
                    "max_el_load": max_el_load,
                })

            # Wärmepumpe mit Pufferspeicher
            if actor.get("P_hp_nom") and actor.get("E_th_nom"):
                hp_name_el = f"{actor_name}_heatpump_el"
                hp_name_th = f"{actor_name}_heatpump_th"
                ths_name = f"{actor_name}_thermal_storage"
                thl_name = f"{actor_name}_thermal_load"
                
                p_hp_nom = self.network.generators.at[hp_name_el, "p_nom"]
                p_ths_nom = self.network.storage_units.at[ths_name, "p_nom"]
                e_ths_max = p_ths_nom * self.network.storage_units.at[ths_name, "max_hours"]

                max_th_load = None
                if thl_name in self.network.loads.index:
                    max_th_load = self.network.loads_t.p_set[thl_name].max()

                self.controllables.append({
                    "type": "heatpump",
                    "name_el": hp_name_el,
                    "name_th": hp_name_th,
                    "p_hp_nom": p_hp_nom,
                    "p_min_pu": 0.25,
                    "con_thermal_storage": ths_name,
                    "e_ths_max": e_ths_max,
                    "con_th_load": thl_name if max_th_load else None,
                    "max_th_load": max_th_load
                })


    def add_forecast_noise(self, true_values: np.ndarray, deviation: float, mode: str = "linear") -> np.ndarray:
        """
        Fügt Rauschen zu Forecastwerten hinzu, dessen Std. am letzten Schritt = deviation ist.
        - true_values: z.B. 96 Punkte
        - deviation: Relative Abweichung am Ende des Forecasts
        - mode: 'linear' | 'sqrt' | 'quadratic' (Wachstumsprofil)
        """
        n_steps = len(true_values)
        if n_steps == 0:
            return true_values

        # Skala 0..1 über Forecast-Länge
        steps = np.arange(n_steps)
        w = steps / (n_steps - 1)

        if mode == "linear":
            f = w
        elif mode == "sqrt":
            f = np.sqrt(w)
        elif mode == "quadratic":
            f = w**2
        else:
            raise ValueError("mode must be 'linear', 'sqrt', or 'quadratic'")

        sigma = deviation * np.abs(true_values) * f

        noise = np.random.normal(loc=0.0, scale=sigma, size=n_steps)
        forecast_values = true_values + noise
        return forecast_values


    def random_timestep(self):
        """
        Wählt zufällig einen Tageszeitraum (96 x 15min) aus dem Trainingszeitraum aus.
        Darf nicht Zeitpunkte unter 48h vor Datensatzende wählen, da das Ende des Zeitraums keinen Observation Space mehr hat, der im Datensatz liegt. 
        """
        max_start = self.training_range[-1] - pd.Timedelta(days=2)
        possible_starts = self.training_range[(self.training_range > self.training_range[0]) & (self.training_range <= max_start)]

        start = np.random.choice(possible_starts)
        start = pd.Timestamp(start) 
        return start


    def encode_time_range(self, timerange):
        seconds_in_day = 24 * 60 * 60
        seconds = timerange.hour * 3600 + timerange.minute * 60
        sin_time = np.sin(2 * np.pi * seconds / seconds_in_day)
        cos_time = np.cos(2 * np.pi * seconds / seconds_in_day)
        return sin_time, cos_time