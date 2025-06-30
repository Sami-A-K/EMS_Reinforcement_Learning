import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import DDPG, SAC, TD3, PPO, DQN, HerReplayBuffer
from sb3_contrib import TQC, RecurrentPPO
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
import joblib
import logging
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
import os
import time

from NeuroGrid_Simulation import EnergySystemSimulation


class EnergySystemEnvironment(gym.Env):
    """
    Custom Gym environment for the energy system simulation.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()

        self.energy_system = EnergySystemSimulation()

        # Action-Space definieren
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.energy_system.num_time_steps,),
            dtype=np.float16
        )

        # Observation-Space konfigurieren
        self.history_length = 96  # 1 Tag
        self.scaler = None
        self.update_observation_space()

        # Platzhalter für State
        self.state = np.zeros(self.observation_space.shape, dtype=np.float32) 

        # Initialwerte für Variablen
        self.total_cost = 0
        self.best_cost = -1e6
        self.best_reward = -1e6
        self.last_action = np.zeros(self.energy_system.num_time_steps, dtype=np.float16)
        self.episode_rewards = []
        self.total_week_reward = []
        self.all_rewards = []
        self.start_time = time.time()

    def reset(self, seed=None, options=None):
        self.time_step = 0
        self.episode_rewards.clear()
        self.total_week_reward.clear()
        self.energy_system.init_timeframe()
        self.energy_system.simulate(DRL=True)

        try:
            self.e_bat = float(self.energy_system.network.stores_t.e['battery_storage_kWh'].iloc[0])
        except Exception as e:
            print(f"[Fehler beim Auslesen von battery_storage_kWh in reset()]: {e}")
            self.e_bat = 0.0

        self.baseline_df = self.get_state(self.energy_system.loadshape)
        self.state = self.baseline_df

        self.best_reward = self.energy_system.reward
        self.total_cost = 0

        return self.state, {}

    def step(self, action):
        truncated = False

        charging_action, discharging_action = self.split_battery_action(action)

        self.energy_system.network.links_t.p_min_pu['battery_charge'] = charging_action
        self.energy_system.network.links_t.p_max_pu['battery_charge'] = charging_action
        self.energy_system.network.links_t.p_min_pu['battery_discharge'] = -discharging_action
        self.energy_system.network.links_t.p_max_pu['battery_discharge'] = -discharging_action

        try:
            self.energy_system.simulate(DRL=True)
            reward = self.energy_system.reward
            self.total_week_reward.append(self.energy_system.full_network_df['cost'].iloc[0])
        except Exception as e:
            print(f"[FEHLER in simulate()]: {e}")
            reward = -1e2
            truncated = True
            self.total_cost = 1e5

        self.episode_rewards.append(reward)
        self.episode_rewards = self.episode_rewards[-self.energy_system.num_time_steps:]
        self.total_week_reward = self.total_week_reward[-self.energy_system.num_time_steps:]
        avg_reward = np.mean(self.episode_rewards)

        try:
            e_series = self.energy_system.network.stores_t.e.get('battery_storage_kWh')
            if e_series is not None and not e_series.isna().iloc[0]:
                self.e_bat = float(e_series.iloc[0])
            else:
                print("[INFO] e_bat ist leer oder NaN – fallback auf 0.0")
                self.e_bat = 0.0
        except Exception as e:
            print(f"[FEHLER beim Auslesen von self.e_bat]: {e}")
            self.e_bat = 0.0

        # Reward-Offset zur Stabilisierung
        try:
            reward -= np.mean(self.total_week_reward) * 100
        except Exception as e:
            print(f"[WARNUNG] Fehler in reward-adjustment: {e}")

        if reward == "Error" or not np.isfinite(reward):
            print("[WARNUNG] Reward ist ungültig – fallback auf -1e2")
            self.total_cost = 1e2
            reward = -1e2
            truncated = True
        else:
            self.total_cost = self.energy_system.total_cost

        terminated = self.time_step >= self.energy_system.num_time_steps
        self.time_step += 1

        self.energy_system.start += pd.Timedelta(minutes=15)
        self.energy_system.end += pd.Timedelta(minutes=15)

        self.energy_system.initialize_pypsa_network() # warum nicht in reset

        try:
            if np.isfinite(self.e_bat):
                self.energy_system.network.stores.e_initial.loc['battery_storage'] = float(self.e_bat)
            else:
                print("[WARNUNG] self.e_bat war nicht finite - wird auf 0.0 gesetzt")
                self.energy_system.network.stores.e_initial.loc['battery_storage'] = 0.0
        except Exception as e:
            print(f"[Fehler beim Setzen von e_initial]: {e}")
            self.energy_system.network.stores.e_initial.loc['battery_storage'] = 0.0

        try:
            self.state = self.get_state(self.energy_system.loadshape)
        except Exception as e:
            print(f"[FEHLER beim get_state()]: {e}")
            self.state = np.zeros((self.history_length,))
            truncated = True

        if self.total_cost < self.best_cost:
            self.best_cost = self.total_cost

        if terminated:
            self.energy_system.full_network_df.to_csv('network_RL_last.csv', index=False)
            all_rewards_df = pd.DataFrame(self.all_rewards, columns=['time', 'reward','week_cost'])
            all_rewards_df.to_csv('rewards_RL_last.csv', index=False)

        info = {
            'total_cost': self.total_cost,
            'best_cost': self.best_cost,
            'reward': avg_reward
        }

        self.state = np.nan_to_num(self.state, nan=0.0)

        return self.state, reward, terminated, truncated, info


    def render(self, mode='human'):
        """
        Render the environment. This method is called when the environment is visualized.
        :param mode: The mode in which to render.
        """
        pass


    # Hilfsfunktionen

    def update_observation_space(self):
        """
        Setzt den Observation-Space auf eine Zeitreihe von Zeitschritten mit mehreren Features.
        """
        num_features = 7

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.history_length, num_features),
            dtype=np.float32
        )

    def split_battery_action(self, battery_action):
        """
        Splits the battery action into charging and discharging actions.
        The charging action contains only positive values, and the discharging action contains only negative values.
        All other values are set to 0 in each action.

        :param battery_action: The original action
        :return: Two arrays, one for charging and one for discharging, both of size 
        """
        # Initialisiere zwei Arrays mit Nullen
        charging_action = np.zeros_like(battery_action)
        discharging_action = np.zeros_like(battery_action)

        # Setze positive Werte für das Laden
        charging_action[battery_action > 0] = battery_action[battery_action > 0]

        # Setze negative Werte für das Entladen
        discharging_action[battery_action < 0] = battery_action[battery_action < 0]

        return charging_action, discharging_action

    def get_state(self, loadshape_df):
        """
        Erstellt den Beobachtungszustand für das DRL-Agenten-Interface.
        Fügt robuste Verarbeitung hinzu, um NaNs und fehlende Werte zu vermeiden.
        """
        selected_columns = [
            "electrical_baseload",
            "thermal_energy_demand",
            "pv_gen_PV_1", 
            "hp_cop_HP_Ground", "hp_cop_HP_Air",
            "price_el"
        ]

        for col in selected_columns:
            if col not in loadshape_df.columns:
                print(f"[WARNUNG] Spalte fehlt: {col} - wird mit 0 gefüllt")
                loadshape_df[col] = 0

        if not hasattr(self, 'e_bat') or not np.isfinite(self.e_bat):
            print("[WARNUNG] self.e_bat ist undefiniert oder NaN - fallback auf 0.0")
            self.e_bat = 0.0
        loadshape_df["initial_battery_storage"] = float(self.e_bat)
        selected_columns.append("initial_battery_storage")

        state_df = loadshape_df[selected_columns].iloc[:self.history_length]
        if len(state_df) < self.history_length:
            pad = pd.DataFrame(0, index=range(self.history_length - len(state_df)), columns=state_df.columns)
            state_df = pd.concat([pad, state_df], ignore_index=True)

        state_df = state_df.apply(pd.to_numeric, errors="coerce").fillna(0)

        if self.scaler is None:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            self.scaler.fit(state_df)

        state = self.scaler.transform(state_df)
        state = np.nan_to_num(state, nan=0.0)

        return state.astype(np.float32)



    def fill_with_zeros(self, partial_loadshape):
        """
        Fill a partial load shape DataFrame with zeros to match the observation space shape.
        """
        obs_rows, obs_columns = self.observation_space.shape

        if partial_loadshape.empty:
            partial_loadshape = pd.DataFrame(columns=range(obs_columns))

        for col in range(obs_columns):
            if col not in partial_loadshape.columns:
                partial_loadshape[col] = 0

        if len(partial_loadshape) < obs_rows:
            additional_rows = obs_rows - len(partial_loadshape)
            additional_data = pd.DataFrame(0, index=range(additional_rows), columns=partial_loadshape.columns)
            partial_loadshape = pd.concat([partial_loadshape, additional_data], ignore_index=True)
        return partial_loadshape.iloc[:obs_rows, :obs_columns]

    def save_scaler(self, filepath):
        """
        Save the fitted standard scaler to a file.
        :param filepath: File path where the scaler should be saved
        """
        joblib.dump(self.scaler, filepath)

    def load_scaler(self, filepath):
        """
        Load the standard scaler from a file.
        :param filepath: File path from where the scaler should be loaded
        """
        self.scaler = joblib.load(filepath)


class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps: int, verbose=0):
        super(ProgressBarCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.progress_bar = None

    def _on_training_start(self) -> None:
        # Initialize progress bar
        self.progress_bar = tqdm(total=self.total_timesteps, desc='Training Progress')

    def _on_step(self) -> bool:
        # Update progress bar
        self.progress_bar.update(1)

        # Retrieve total_cost and best_cost from info if available
        if 'infos' in self.locals and len(self.locals['infos']) > 0:
            latest_info = self.locals['infos'][-1]
            total_cost = latest_info.get('total_cost', None)
            best_cost = latest_info.get('best_cost', None)
            reward = latest_info.get('reward', None)
            
            # Update the progress bar display with total_cost and best_cost
            if total_cost is not None and best_cost is not None:
                self.progress_bar.set_postfix(total_cost=total_cost, avg_reward=reward)

        return True

    def _on_training_end(self) -> None:
        # Close progress bar
        self.progress_bar.close()


class AdaptiveActionNoise:
    def __init__(self, mean, sigma, decay_rate=0.99, min_sigma=0.1):
        self.mean = mean
        self.sigma = sigma
        self.initial_sigma = sigma  # Speichert das Anfangssigma
        self.decay_rate = decay_rate
        self.min_sigma = min_sigma
        self.state = np.zeros_like(mean)  # Initialisiert den Zustand (Rauschen)
        
    def reset(self):
        self.sigma = self.initial_sigma  # Setzt sigma zurück auf den Startwert
        self.state = np.zeros_like(self.mean)  # Setzt den Zustand auf den Anfangszustand
    
    def __call__(self):
        # Berechne das Rauschen (Ornstein-Uhlenbeck-Noise)
        dx = 0.1 * np.random.randn(*self.mean.shape) + self.state * 0.99
        self.state = dx
        self.sigma = max(self.min_sigma, self.sigma * self.decay_rate)  # Reduziere sigma
        return dx
    
def train_energy_system_agent(name, total_timesteps):
    """
    Train a DDPG agent on the custom energy system environment to minimize energy costs.
    Save the trained model and the scaler used for normalization.

    :param total_timesteps: The total number of training timesteps
    """
    env = EnergySystemEnvironment()

    # Define DDPG parameters including action noise
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    
    #action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.5 * np.ones(n_actions))
    #action_noise = AdaptiveActionNoise(mean=np.zeros(n_actions), sigma=0.1, decay_rate=0.995, min_sigma=0.05)

    # policy_kwargs = dict(
    #     net_arch=[256, 256] #[512, 256, 128]
    # )
    # # Initialize DDPG agent
    # model = SAC(
    #     "MlpPolicy", 
    #     env,
    #     action_noise=action_noise, 
    #     ent_coef="auto_0.1",
    #     policy_kwargs=policy_kwargs,
    #     learning_rate=lambda f: 1e-3 * f,
    #     batch_size=128,
    #     #train_freq=10,
    #     #gradient_steps=10,
    #     verbose=0
    # )
    #model = DDPG("MlpPolicy", env, buffer_size=100_000, device="cpu", verbose=0)
    model = SAC("MlpPolicy", env, device="cpu", verbose=0)
    #model = RecurrentPPO("MlpLstmPolicy", env, device="cpu", verbose=0)
    #model = TD3("MlpPolicy", env, device="cpu", verbose=0)
    # model = DDPG(
    #     "MlpPolicy",
    #     env,
    #     buffer_size=10000,
    #     batch_size=16,
    #     action_noise=action_noise,
    #     learning_rate = 1e-4,
    #     tau=0.005,
    #     policy_kwargs=policy_kwargs,
    #     device="cuda",
    #     verbose=0
    # )

    # Train the agent with progress bar callback
    model.learn(total_timesteps=total_timesteps, callback=ProgressBarCallback(total_timesteps=total_timesteps))

    # Save the trained model and the scaler
    model.save(f"models/{name}_VPP")
    env.save_scaler(f"models/{name}_VPP_scaler.pkl")
    return model, env

def train_and_evaluate_rl_models(total_timesteps=100, save_path="models/"):
    """
    Train and evaluate multiple RL models on the EnergySystemEnvironment.

    :param total_timesteps: Number of training timesteps for each model
    :param save_path: Directory to save trained models and scalers
    """
    os.makedirs(save_path, exist_ok=True)

    # Initialisiere die Umgebung
    env = EnergySystemEnvironment()
    n_actions = env.action_space.shape[-1]

    # Setze eine starke Exploration durch Ornstein-Uhlenbeck-Noise (für DDPG, TD3)
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.5 * np.ones(n_actions))

    # Liste der Algorithmen mit jeweiligen Parametern
    models = {
        #"DDPG": DDPG("MlpPolicy", env, buffer_size=100_000, action_noise=action_noise, tau=0.005, device="cpu", verbose=0),
        #"SAC": SAC("MlpPolicy", env, buffer_size=100_000, tau=0.005, device="cpu", verbose=0),
        #"TD3": TD3("MlpPolicy", env, buffer_size=100_000, action_noise=action_noise, tau=0.005, device="cpu", verbose=0),
        #"PPO": PPO("MlpPolicy", env, n_steps=2048, batch_size=64, ent_coef=0.01, device="cpu", verbose=0),
        "TQC": TQC("MlpPolicy", env, top_quantiles_to_drop_per_net=2, verbose=0, device="cpu"),
        #"DQN": DQN("MlpPolicy", env, verbose=0, device="cpu")
    }

    results = {}

    for name, model in models.items():
        print(f"Training {name} for {total_timesteps} timesteps...")
        
        # Trainieren mit Fortschrittsanzeige
        model.learn(total_timesteps=total_timesteps, callback=ProgressBarCallback(total_timesteps=total_timesteps))

        # Speichern des Modells
        model_path = os.path.join(save_path, f"{name}_energy_system.zip")
        model.save(model_path)
        
        # Speichern des Scalers, falls vorhanden
        scaler_path = os.path.join(save_path, f"{name}_scaler.pkl")
        env.save_scaler(scaler_path)

        print(f"{name} trained and saved at {model_path}")

        # Teste die durchschnittliche Belohnung nach Training
        avg_reward = evaluate_model(model, env, num_episodes=10)
        results[name] = avg_reward
        print(f"Average reward for {name}: {avg_reward:.2f}")

    return results

def evaluate_model(model, env, num_episodes=10):
    """
    Evaluate a trained model by running multiple episodes and averaging the rewards.

    :param model: Trained RL model
    :param env: Environment to test the model
    :param num_episodes: Number of test episodes
    :return: Average reward over episodes
    """
    total_reward = 0.0

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            episode_reward += reward

        total_reward += episode_reward / 672

    return total_reward / num_episodes

def train_and_evaluate(name, total_timesteps=672, iterations=10, eval_episodes=5, log_file="training_log.csv"):
    """
    Trains, evaluates, and continues training a reinforcement learning model.
    Saves average rewards with timestamps in a CSV file, overwriting previous results.

    :param name: Name of the RL algorithm (e.g., "SAC", "PPO").
    :param total_timesteps: Timesteps per training iteration.
    :param iterations: Number of train-evaluate cycles.
    :param eval_episodes: Number of episodes per evaluation.
    :param log_file: Path to CSV file for logging results.
    :return: Trained model
    """
    model, train_env = train_energy_system_agent(name=name, total_timesteps=total_timesteps)

    log_data = []  # Zwischenspeicher für Logging

    for i in range(iterations):
        avg_reward = evaluate_model(model, train_env, num_episodes=eval_episodes)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")  # Aktuelle Zeit als String
        print(f"Iteration {i+1}/{iterations}: Average reward = {avg_reward:.2f}")

        log_data.append([timestamp, avg_reward])  # Daten speichern

        # CSV-Datei mit den aktuellen Logs überschreiben
        df = pd.DataFrame(log_data, columns=["timestamp", "average_reward"])
        df.to_csv(log_file, index=False)

        model.learn(total_timesteps=total_timesteps)

    return model



# Execute training
model, train_env = train_energy_system_agent(name="SAC" , total_timesteps=5000)

#train_and_evaluate(name="DDPG", total_timesteps=672, iterations=100, eval_episodes=3, log_file="training_log_vpp.csv")
#train_and_evaluate_rl_models(total_timesteps=192, save_path="models/")