import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from Energy_District_Gym_Environment import EnergyDistrictEnvironment


# Modell laden
model = PPO.load("./models/ppo_energy_district")

# Test-Environment
env = EnergyDistrictEnvironment()
obs, info = env.reset()

# Episode aufzeichnen
records = []
terminated, truncated = False, False
while not (terminated or truncated):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    row = {"timestep": info["timestep"], "reward": float(reward),
           "baseline_cost": float(info["baseline"]), "agent_cost": float(info["agent"]),
           "clip_pen": float(info["clip_pen"])}
    # Reale Leistungen & SOCs loggen
    for c in env.controllables:
        
        if c["type"] == "battery":
            row[f"{c['name']}_P"] = float(env.network.storage_units_t.p_set.loc[info["timestep"], c["name"]])
            row[f"{c['con_pv']}_P"] = float(env.network.generators_t.p_set.loc[info["timestep"], c["con_pv"]])
            row[f"{c['con_el_load']}_P"] = float(env.network.loads_t.p_set.loc[info["timestep"], c["con_el_load"]])
            soc = float(env.network.storage_units_t.state_of_charge_set.loc[info["timestep"], c["name"]])
            row[f"{c['name']}_SOC"] = soc / c["e_bat_max"]
        elif c["type"] == "heatpump":
            row[f"{c['name_el']}_P"] = float(env.network.generators_t.p_set.loc[info["timestep"], c["name_el"]])
            soc = float(env.network.storage_units_t.state_of_charge_set.loc[info["timestep"], c["con_thermal_storage"]])
            row[f"{c['con_thermal_storage']}_SOC_th"] = soc / c["e_ths_max"]
    records.append(row)
# DataFrame bauen
df = pd.DataFrame(records)
df["timestep"] = pd.to_datetime(df["timestep"])
df = df.set_index("timestep").sort_index()

print(f"Geloggte Schritte: {len(df)}")
df.to_csv("episode_log.csv")
print("CSV gespeichert: episode_log.csv")

# Plot
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
# 1) Leistungen
for col in df.filter(like="_P").columns:
    axes[0].plot(df.index, df[col], label=col)
axes[0].set_ylabel("Leistung [kW]")
axes[0].legend(loc="upper right", ncol=2)
axes[0].grid(True)

# 2) SOCs
for col in df.filter(like="_SOC").columns:
    axes[1].plot(df.index, df[col], label=col)
axes[1].set_ylabel("SOC [0..1]")
axes[1].legend(loc="upper right", ncol=2)
axes[1].grid(True)

plt.tight_layout()
plt.show()
