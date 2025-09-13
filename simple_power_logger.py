"""
Info
----
Power Data Logger für Agent und Baseline Vergleich

Speichert Power-Daten, Kosten und Steuerbare Komponenten-Daten in CSV-Format.
"""

import pandas as pd
import numpy as np
import os


class SimplePowerLogger:
    """
    Logger für Agent und Baseline Power-Daten, Kosten und Steuerbare Komponenten.
    """
    
    def __init__(self, log_file: str = "power_data.csv"):
        """
        Initializes the logger for power data logging.

        Args:
            log_file (str): The path to the CSV file for data storage.
        """
        self.log_file = log_file
        self.data = []
        
    def log_step(self, timestep, agent_power_data, baseline_power_data, costs=None, controllable_data=None):
        """
        Logs data for a single timestep including power, costs and controllable components.

        Args:
            timestep: Current timestep timestamp
            agent_power_data (dict): Agent power data per actor
            baseline_power_data (dict): Baseline power data per actor
            costs (dict, optional): Cost information including agent and baseline costs
            controllable_data (dict, optional): Heat pump and battery data
        """
        record = {
            "timestep": timestep,
            "step": len(self.data) + 1
        }
        
        # Power data for each actor
        for actor_name in agent_power_data.keys():
            record[f"agent_power_{actor_name}"] = agent_power_data[actor_name]
            record[f"baseline_power_{actor_name}"] = baseline_power_data.get(actor_name, 0.0)
            record[f"power_diff_{actor_name}"] = agent_power_data[actor_name] - baseline_power_data.get(actor_name, 0.0)
        
        # Heat pump and battery data
        if controllable_data:
            for comp_name, comp_data in controllable_data.items():
                for data_key, data_value in comp_data.items():
                    record[f"{comp_name}_{data_key}"] = data_value
        
        # Cost data if available
        if costs:
            record.update({
                "agent_cost": costs.get("agent", 0.0),
                "baseline_cost": costs.get("baseline", 0.0),
                "cost_saving": costs.get("saving", 0.0),
                "reward": costs.get("reward", 0.0)
            })
        
        self.data.append(record)
    
    def save_to_csv(self):
        """
        Saves all logged data to CSV file and displays summary statistics.
        """
        if not self.data:
            print("No data available for saving!")
            return
        
        df = pd.DataFrame(self.data)
        df["timestep"] = pd.to_datetime(df["timestep"])
        df = df.set_index("timestep")
        
        df.to_csv(self.log_file)
        print(f"Data saved: {self.log_file}")
        print(f"Number of timesteps: {len(df)}")
        
        # Display summary
        print("\n=== SUMMARY ===")
        
        # Power statistics
        agent_cols = [col for col in df.columns if col.startswith("agent_power_")]
        baseline_cols = [col for col in df.columns if col.startswith("baseline_power_")]
        
        print("Power data:")
        for agent_col in agent_cols:
            actor_name = agent_col.replace("agent_power_", "")
            baseline_col = f"baseline_power_{actor_name}"
            
            if baseline_col in df.columns:
                agent_avg = df[agent_col].mean()
                baseline_avg = df[baseline_col].mean()
                diff_avg = df[f"power_diff_{actor_name}"].mean()
                
                print(f"  {actor_name}:")
                print(f"    Agent average: {agent_avg:.2f} kW")
                print(f"    Baseline average: {baseline_avg:.2f} kW")
                print(f"    Average difference: {diff_avg:.2f} kW")
        
        # Cost statistics
        if "agent_cost" in df.columns:
            total_agent_cost = df["agent_cost"].sum()
            total_baseline_cost = df["baseline_cost"].sum()
            total_saving = df["cost_saving"].sum()
            
            print(f"\nCosts:")
            print(f"  Agent total costs: {total_agent_cost:.3f} €")
            print(f"  Baseline total costs: {total_baseline_cost:.3f} €")
            print(f"  Total savings: {total_saving:.3f} €")
            
            if total_baseline_cost != 0:
                saving_percent = (total_saving / total_baseline_cost) * 100
                print(f"  Savings percentage: {saving_percent:.2f} %")


def extract_power_data_from_environment(env):
    """
    Extracts power data from the energy district environment.

    Args:
        env: EnergyDistrictEnvironment instance
        
    Returns:
        tuple: (agent_power_data, baseline_power_data) dictionaries
    """
    agent_power_data = {}
    baseline_power_data = {}
    
    for actor in env.config["actors"]:
        actor_name = actor["name"]
        
        # Agent power data
        agent_power = env.network.lines_t.p1.loc[env.timestep, f"{actor_name}_service_line"]
        agent_power_data[actor_name] = float(agent_power)
        
        # Baseline power data  
        baseline_power = env.baseline_power.loc[env.timestep, f"{actor_name}_service_line"]
        baseline_power_data[actor_name] = float(baseline_power)
    
    return agent_power_data, baseline_power_data


def extract_controllable_data_from_environment(env):
    """
    Extracts heat pump and battery data from the energy district environment.
    
    Args:
        env: EnergyDistrictEnvironment instance
        
    Returns:
        dict: Dictionary containing heat pump and battery data for all controllable components
    """
    controllable_data = {}
    
    for controllable in env.controllables:
        comp_name = controllable["name"]
        controllable_data[comp_name] = {}
        
        if controllable["type"] == "battery":
            # Battery power
            p_bat = env.network.storage_units_t.p_set.loc[env.timestep, comp_name]
            controllable_data[comp_name]["P_bat"] = float(p_bat)
            
            # State of Charge (SOC)
            soc = env.network.storage_units_t.state_of_charge_set.loc[env.timestep, comp_name]
            controllable_data[comp_name]["SOC"] = float(soc / controllable["e_bat_max"])
            
        elif controllable["type"] == "heatpump":
            # Heat pump electrical power
            p_hp_el = env.network.generators_t.p_set.loc[env.timestep, controllable["name_el"]]
            controllable_data[comp_name]["P_hp_el"] = float(p_hp_el)
            
            # Heat pump thermal power
            p_hp_th = env.network.generators_t.p_set.loc[env.timestep, controllable["name_th"]]
            controllable_data[comp_name]["P_hp_th"] = float(p_hp_th)
            
            # Thermal storage SOC
            soc_th = env.network.storage_units_t.state_of_charge_set.loc[env.timestep, controllable["con_thermal_storage"]]
            controllable_data[comp_name]["SOC_th"] = float(soc_th / controllable["e_ths_max"])
            
            # Coefficient of Performance (COP)
            cop = env.network.generators_t.efficiency.loc[env.timestep, controllable["name_th"]]
            controllable_data[comp_name]["COP"] = float(cop)
    
    return controllable_data


def run_simple_agent_test(model_path="./models/ppo_energy_district", 
                         csv_file="power_data.csv", 
                         seed=42,
                         num_episodes=7):
    """
    Runs a simple agent test and logs all data to CSV file.
    
    Args:
        model_path (str): Path to the trained PPO model
        csv_file (str): Name of the CSV output file
        seed (int): Random seed for reproducible results
        num_episodes (int): Number of episodes to run (default: 7 = 1 week)
    """
    from stable_baselines3 import PPO
    from Energy_District_Gym_Environment import EnergyDistrictEnvironment
    
    print("=== SIMPLE AGENT TEST ===")
    
    # Load model
    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)
    
    # Initialize environment
    env = EnergyDistrictEnvironment()
    
    # Initialize logger
    logger = SimplePowerLogger(csv_file)
    
    total_steps = 0
    total_savings = 0.0
    
    for episode in range(num_episodes):
        print(f"\n=== EPISODE {episode + 1}/{num_episodes} ===")
        obs, info = env.reset(seed=seed + episode)
        
        print(f"Starting episode {episode + 1}")
        terminated, truncated = False, False
        episode_steps = 0
        episode_savings = 0.0
        
        while not (terminated or truncated):
            # Agent action
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Extract power data
            agent_power_data, baseline_power_data = extract_power_data_from_environment(env)
            
            # Extract controllable component data
            controllable_data = extract_controllable_data_from_environment(env)
            
            # Cost data
            costs = {
                "agent": float(info["agent"]),
                "baseline": float(info["baseline"]),
                "saving": float(info["saving"]),
                "reward": float(reward)
            }
            
            # Log step
            logger.log_step(
                timestep=info["timestep"],
                agent_power_data=agent_power_data,
                baseline_power_data=baseline_power_data,
                costs=costs,
                controllable_data=controllable_data
            )
            
            episode_steps += 1
            episode_savings += costs["saving"]
            
            # Progress display
            if episode_steps % 24 == 0:  # Every 6 hours
                print(f"  Episode {episode + 1}, Step {episode_steps}, Savings: {costs['saving']:.3f}€")
        
        total_steps += episode_steps
        total_savings += episode_savings
        print(f"Episode {episode + 1} completed: {episode_steps} steps, Total savings: {episode_savings:.3f}€")
    
    # Save data
    logger.save_to_csv()
    
    print(f"\n=== TEST SUMMARY ===")
    print(f"Total episodes: {num_episodes}")
    print(f"Total steps: {total_steps}")
    print(f"Total savings: {total_savings:.3f}€")
    print(f"Average savings per episode: {total_savings/num_episodes:.3f}€")
    print(f"CSV file: {csv_file}")


def run_extended_test(model_path="./models/ppo_energy_district", 
                     csv_file="power_data_extended.csv", 
                     seed=42,
                     test_type="week"):
    """
    Runs extended tests with different durations.
    
    Args:
        model_path (str): Path to the trained PPO model
        csv_file (str): Name of the CSV output file
        seed (int): Random seed for reproducible results
        test_type (str): Type of test - "day", "week", "month", "year"
    """
    test_configs = {
        "day": 1,
        "week": 7,
        "month": 30,
        "year": 365
    }
    
    if test_type not in test_configs:
        raise ValueError(f"test_type must be one of: {list(test_configs.keys())}")
    
    num_episodes = test_configs[test_type]
    print(f"Running {test_type} test with {num_episodes} episodes ({num_episodes * 96} total steps)")
    
    run_simple_agent_test(
        model_path=model_path,
        csv_file=csv_file,
        seed=seed,
        num_episodes=num_episodes
    )


if __name__ == "__main__":
    # Choose test type:
    # run_simple_agent_test()  # 1 day (96 steps)
    # run_extended_test(test_type="day")    # 1 day
    # run_extended_test(test_type="week")   # 1 week  
    # run_extended_test(test_type="month")  # 1 month
    run_extended_test(test_type="week")     # 1 week (recommended)
