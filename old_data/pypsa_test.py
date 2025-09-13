import pypsa

n = pypsa.Network()
n.set_snapshots([0])

# Elektrische Busse mit v_nom
#n.add("Bus", "grid", v_nom=230)
n.add("Bus", "bus_th", v_nom=230)  

n.add("Generator", "heatpump", bus="bus_th", p_nom=10, control='PQ')
n.add("Load", "hp_el_load", bus="bus_th", p_set=10)
n.add("StorageUnit", "THS", bus="bus_th", p_nom=100, max_hours=1, p_set=10)
n.pf()
print(n.statistics())
# # Genau EIN Slack im Subnetz
# n.add("Generator", "slack_elec", bus="bus_slack", p_nom=1000, control='Slack')

# # AC-Leitung mit nicht-Null-Impedanz und s_nom
# n.add("Line", "cable",
#       bus0="bus_slack", bus1="bus_e",
#       r=0.01, x=0.1, s_nom=1000)

# # WICHTIG: Wenn du hier eine *thermische* Last meintest, gehört sie NICHT in den pf().
# # Für den AC-PF brauchst du die *elektrische* Aufnahme der WP als PQ-Last:
# COP = 3.0
# Q_heat = 400.0            # kW Wärmebedarf (dein Wert)
# P_el = Q_heat / COP       # ~133.33 kW elektrische Last

# n.add("Load", "hp_el_load", bus="bus_e", p_set=P_el)
# n.add("StorageUnit", "Bat", bus="bus_e", p_nom=100, max_hours=1, p_set=10)
# # Power Flow
# n.pf()
# print(n.statistics())
# print("Generator outputs (MW):")
# print(n.generators_t.p)
# print("\nLine flow p0 (MW):")
# print(n.lines_t.p0)
# print("\nLoads (MW):")
#print(n.loads_t.p)
