import gym
from gym import spaces
import numpy as np


class SimpleEnergyEnv(gym.Env):
    """
    Eine einfache Gym-Umgebung zur Simulation eines vereinfachten Energiesystems eines Einfamilienhauses.

    Systemkomponenten:
      - PV-Anlage (10 kWp) mit hinterlegtem Erzeugungsprofil (pro 5-Minuten-Schritt)
      - Batteriespeicher (5 kWh)
      - Wärmepumpe (10 kW elektrische Leistung, pro Schritt ca. 0,83 kWh, COP = 3)
      - Wärmespeicher (20 kWh ~350l)
      - Zufällig generierte Haushaltslast- und Wärmebedarfsprofile, die typische Anwesenheitszeiten simulieren

    Ziel:
      Der RL-Agent soll lernen, den Betrieb (insbesondere den Einsatz von PV-Strom, Batterie und Wärmepumpe) so zu steuern,
      dass die Energiekosten minimiert werden. Dabei gelten folgende Annahmen:
        - PV: 6 ct/kWh
        - Batterie: 18 ct/kWh
        - Netz: 35 ct/kWh
    """

    def __init__(self):
        super(SimpleEnergyEnv, self).__init__()

        # Simulationsparameter: 5-Minuten-Schritte eines Tages -> 24h * 12 = 288 Schritte
        self.episode_length = 288
        self.current_step = 0

        # Systemkapazitäten
        self.battery_capacity = 5.0  # kWh
        self.thermal_storage_capacity = 20.0  # kWh (realistischer Wert, ca. 350l Pufferspeicher)

        # Initialzustände
        self.battery_state = self.battery_capacity * np.random.random() * 0.5
        self.thermal_storage_state = self.thermal_storage_capacity * np.random.random() * 0.5

        # Kosten (in ct pro kWh)
        self.cost_PV = 0.06  # PV-Energie
        self.cost_battery = 0.18  # Batterieentladung
        self.cost_grid = 0.35  # Netzbezug

        # Wärmepumpenparameter
        # 10 kW elektrische Leistung -> pro 5-Minuten-Schritt: 10 * (5/60) ≈ 0.83 kWh
        self.heatpump_max_power = 10.0 * (5 / 60)
        self.heatpump_COP = 3.0  # Wirkungsgrad

        # Zeitvektor in Stunden für jeden Schritt
        time_hours = np.arange(self.episode_length) * (5 / 60)

        # PV-Profil: Sinusförmig, Peak um 6h (Anpassung: Zeit in Stunden), Umrechnung in kWh pro 5 Minuten
        # Maximaler Ertrag pro Schritt ≈ 10 kW * (5/60) = 0.83 kWh
        self.pv_profile = 10 * np.maximum(0, np.sin((np.pi / 12) * (time_hours - 6))) * (5 / 60)

        # Zufällig generiertes Haushaltslastprofil:
        # Höhere Last in den Morgen- (6-9 Uhr) und Abendstunden (17-21 Uhr), niedriger sonst.
        base_house_load = np.where(((time_hours >= 6) & (time_hours < 9)) | ((time_hours >= 17) & (time_hours < 21)),
                                   0.1, 0.01)
        noise_house = np.random.uniform(0.25, 2, size=self.episode_length)
        self.house_load_profile = np.clip(base_house_load * noise_house, 0, None)

        # Zufällig generiertes Wärmebedarfprofil:
        # Höherer Bedarf in den gleichen Zeitfenstern (6-9 Uhr und 17-21 Uhr), niedriger sonst.
        # Dabei entspricht 0.25 kWh pro Schritt ca. 3 kWh pro Stunde, 0.0833 kWh pro Schritt ca. 1 kWh pro Stunde.
        base_heat_demand = np.where(((time_hours >= 6) & (time_hours < 9)) | ((time_hours >= 17) & (time_hours < 21)),
                                    0.25, 0.0833)
        noise_heat = np.random.uniform(0.75, 1.25, size=self.episode_length)
        self.heat_demand_profile = np.clip(base_heat_demand * noise_heat, 0, None)

        # Definition des Aktionsraums:
        # Aktionen: [battery_discharge, heatpump_electric_input]
        # battery_discharge: gewünschte Entladung in kWh (0 bis max. Batteriekapazität)
        # heatpump_electric_input: elektrische Energie, die der Wärmepumpe zugeführt wird (0 bis heatpump_max_power)
        self.action_space = spaces.Box(low=np.array([0.0, 0.0]),
                                       high=np.array([self.battery_capacity, self.heatpump_max_power]),
                                       dtype=np.float32)

        # Definition des Beobachtungsraums:
        # Beobachtung: [battery_state, thermal_storage_state, aktueller PV-Ertrag, Haushaltslast, Wärmebedarf, normalisierte Zeit]
        low_obs = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # Obergrenzen: Batterie (5 kWh), Wärmespeicher (20 kWh), PV (ca. 1 kWh), Haushaltslast (max. ca. 0.5 kWh), Wärmebedarf (max. ca. 0.5 kWh), Zeit [0,1]
        high_obs = np.array([self.battery_capacity, self.thermal_storage_capacity, 1.0, 0.5, 0.5, 1.0])
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

    def reset(self):
        """
        Setzt die Umgebung auf den Anfangszustand zurück.
        """
        time_hours = np.arange(self.episode_length) * (5 / 60)
        self.current_step = 0
        self.battery_state = self.battery_capacity * 0.5
        self.thermal_storage_state = self.thermal_storage_capacity * 0.5
        base_house_load = np.where(((time_hours >= 6) & (time_hours < 9)) | ((time_hours >= 17) & (time_hours < 21)),
                                   0.1, 0.01)
        noise_house = np.random.uniform(0.25, 2, size=self.episode_length)
        self.house_load_profile = np.clip(base_house_load * noise_house, 0, None)
        base_heat_demand = np.where(((time_hours >= 6) & (time_hours < 9)) | ((time_hours >= 17) & (time_hours < 21)),
                                    0.25, 0.0833)
        noise_heat = np.random.uniform(0.75, 1.25, size=self.episode_length)
        self.heat_demand_profile = np.clip(base_heat_demand * noise_heat, 0, None)
        return self._get_obs()

    def _get_obs(self):
        """
        Gibt den aktuellen state zurück.
        """
        pv = self.pv_profile[self.current_step]
        house_load = self.house_load_profile[self.current_step]
        heat_demand = self.heat_demand_profile[self.current_step]
        time_norm = self.current_step / (self.episode_length - 1)
        return np.array([self.battery_state, self.thermal_storage_state, pv, house_load, heat_demand, time_norm],
                        dtype=np.float32)

    def step(self, action):
        """
        Führt einen 5-Minuten-Zeitschritt in der Umgebung aus.

        Parameter:
          - action: numpy Array mit den Werten [battery_discharge, heatpump_electric_input]

        Ablauf:
          1. Ermittlung des elektrischen Gesamtbedarfs: Haushaltslast + Wärmepumpeneinsatz.
          2. Zunächst wird der PV-Ertrag genutzt.
          3. Bei Bedarf erfolgt die Batterieentladung, bevor fehlende Energie aus dem Netz bezogen wird.
          4. Überschüssiger PV-Ertrag lädt den Batteriespeicher automatisch auf.
          5. Der Wärmepumpen-Einsatz (mit COP) produziert Wärme, die den Wärmebedarf deckt.
             Überschüsse werden im Wärmespeicher gespeichert, Defizite – soweit möglich – aus diesem entnommen.
          6. Nicht gedeckter Wärmebedarf wird bestraft.
          7. Es werden die Kosten (in ct) berechnet, der Reward entspricht negativen Gesamtkosten.

        Rückgabe:
          - obs: neuer state
          - reward: Reward (negative Gesamtkosten)
          - done: Boolean, ob die Episode beendet ist
          - info: Dictionary mit zusätzlichen Informationen
        """
        # Extrahiere und begrenze die Aktionswerte
        # Hier müsst ihr anpassen, je nach RL Methodik, die Implementierung hier ist "continuous control",
        # viele Algorithmen brauchen aber diskrete action spaces! Dann könnte man zB ein Mapping machen und
        # in 10% Schritten vorgehen
        battery_discharge_requested = np.clip(action[0], 0.0, self.battery_capacity)
        heatpump_input = np.clip(action[1], 0.0, self.heatpump_max_power)

        # Aktuelle Profilwerte für diesen 5-Minuten-Schritt
        pv = self.pv_profile[self.current_step]
        house_load = self.house_load_profile[self.current_step]
        heat_demand = self.heat_demand_profile[self.current_step]

        # --- Elektrischer Teil ---
        # Gesamtbedarf: Haushaltslast + Wärmepumpeneinsatz (alle Werte in kWh pro Schritt)
        total_electric_load = house_load + heatpump_input

        # PV wird primär zur Deckung des Bedarfs genutzt
        pv_used_for_load = min(pv, total_electric_load)
        remaining_load = total_electric_load - pv_used_for_load

        # Batterieentladung: Der Agent bestimmt die gewünschte Entladung, begrenzt durch den Batteriezustand und den Bedarf
        battery_discharge = min(battery_discharge_requested, self.battery_state, remaining_load)
        remaining_load -= battery_discharge

        # Fehlende Energie wird aus dem Netz bezogen
        grid_usage = remaining_load

        # Aktualisiere den Batteriezustand
        self.battery_state -= battery_discharge

        # Überschüssige PV-Energie lädt den Batteriespeicher automatisch (falls Kapazität vorhanden)
        if pv > total_electric_load:
            surplus = pv - total_electric_load
            battery_charge = min(surplus, self.battery_capacity - self.battery_state)
            self.battery_state += battery_charge
        else:
            battery_charge = 0.0

        # Berechnung der elektrischen Kosten (in ct)
        cost_electric = (pv_used_for_load * self.cost_PV +
                         battery_discharge * self.cost_battery +
                         grid_usage * self.cost_grid)

        # --- Thermischer Teil (Wärmepumpe und Wärmespeicher) ---
        # Wärmepumpe produziert Wärme (mit COP)
        heat_produced = self.heatpump_COP * heatpump_input

        # Vergleiche produzierte Wärme mit Wärmebedarf
        if heat_produced >= heat_demand:
            surplus_heat = heat_produced - heat_demand
            # Überschüssige Wärme wird im Wärmespeicher gespeichert (sofern Platz vorhanden)
            heat_stored = min(surplus_heat, self.thermal_storage_capacity - self.thermal_storage_state)
            self.thermal_storage_state += heat_stored
            unmet_heat = 0.0
        else:
            deficit = heat_demand - heat_produced
            # Nutze den Wärmespeicher, um den Fehlbetrag zu decken – dies kann über mehrere Stunden den Speicher entladen
            heat_from_storage = min(self.thermal_storage_state, deficit)
            unmet_heat = deficit - heat_from_storage
            self.thermal_storage_state -= heat_from_storage

        # Strafkosten für nicht gedeckten Wärmebedarf (50,00 ct pro kWh Fehlbedarf)
        heat_penalty = unmet_heat * 50.00

        # Gesamtkosten sind die Summe aus elektrischen Kosten und Wärme-Penalty
        total_cost = cost_electric + heat_penalty

        # Reward ist negativ, da das Ziel Kostenminimierung ist
        reward = -total_cost

        # Erhöhe den Zeitschritt
        self.current_step += 1
        done = self.current_step >= self.episode_length

        # Aktualisiere den Beobachtungsvektor
        obs = self._get_obs() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)

        info = {
            "cost_electric": cost_electric,
            "heat_penalty": heat_penalty,
            "grid_usage": grid_usage,
            "battery_discharge": battery_discharge,
            "battery_charge": battery_charge,
            "heat_produced": heat_produced,
            "unmet_heat": unmet_heat
        }

        return obs, reward, done, info

    def render(self, mode='human'):
        """
        Gibt den aktuellen Zustand der Umgebung aus.
        """
        print(f"Step: {self.current_step}")
        print(f"Battery: {self.battery_state:.2f} kWh, Thermal Storage: {self.thermal_storage_state:.2f} kWh")
        # Falls current_step == 0, verwende den ersten Wert im PV-Profil
        current_pv = self.pv_profile[self.current_step - 1] if self.current_step > 0 else self.pv_profile[0]
        print(f"Aktueller PV-Ertrag: {current_pv:.2f} kWh")


if __name__ == "__main__":
    # Beispielhafte Nutzung der Umgebung
    env = SimpleEnergyEnv()
    obs = env.reset()
    print("Initial Observation:", obs)

    # Beispielhafte Schleife über einige 5-Minuten-Schritte
    for step in range(10):
        action = env.action_space.sample()  # Zufällige Aktion als Beispiel
        obs, reward, done, info = env.step(action)
        print(f"\nStep: {step + 1}")
        print(f"Aktion: {action}")
        print(f"Beobachtung: {obs}")
        print(f"Reward: {reward}")
        print(f"Info: {info}")
        if done:
            break