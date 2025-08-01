import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Parameters
N = 100
K = 0.6
dt = 0.0125
Tmax = 60
timesteps = int(Tmax / dt)
edges_max = N * (N - 1) / 2
num_topologies = 2
fixed_edges = 2000  # adjust this to fix cost
output_dir = "kuramoto_batch_output"
os.makedirs(output_dir, exist_ok=True)

omega = 1.0 + 0.1 * np.random.randn(N)
theta0 = 2 * np.pi * np.random.rand(N)

# Simulation

def run_kuramoto(G, omega, theta0):
    N = G.number_of_nodes()
    theta = theta0.copy()
    A = nx.to_numpy_array(G)
    r_series = []

    for _ in range(timesteps):
        r_series.append(np.abs(np.mean(np.exp(1j * theta))))
        theta_dot = np.zeros(N)
        for i in range(N):
            sum_sin = sum(np.sin(theta[j] - theta[i]) for j in range(N) if A[i, j])
            theta_dot[i] = omega[i] + (K / N) * sum_sin
        theta += theta_dot * dt

    return np.array(r_series)

# Generate graph with fixed number of edges

def generate_graph_fixed_edges(seed, target_edges):
    np.random.seed(seed)
    max_attempts = 1000
    for attempt in range(max_attempts):
        G = nx.gnm_random_graph(N, target_edges, seed=seed + attempt)
        if nx.is_connected(G):
            return G
    raise ValueError("Failed to generate connected graph with fixed edges.")

# Global r(t)
G_global = nx.complete_graph(N)
r_global_t = run_kuramoto(G_global, omega, theta0)
r_global = r_global_t[-1]

results = []

# Multithreaded execution

def simulate_topology(i):
    seed = i
    G = generate_graph_fixed_edges(seed, fixed_edges)
    r_t = run_kuramoto(G, omega, theta0)
    final_r = r_t[-1]
    cost = G.number_of_edges() / edges_max
    efficiency = final_r / r_global
    return {
        "Topology_ID": i,
        "Edges": G.number_of_edges(),
        "Cost": cost,
        "Final_r(t)": final_r,
        "Efficiency": efficiency
    }

with ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(simulate_topology, i) for i in range(num_topologies)]
    for future in as_completed(futures):
        results.append(future.result())

# Save CSV

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
df = pd.DataFrame(results)
csv_out = f"{output_dir}/efficiency_hist_{timestamp}.csv"
df.to_csv(csv_out, index=False)

# Histogram

plt.figure(figsize=(10, 6))
plt.hist(df["Efficiency"], bins=20, color="skyblue", edgecolor="black")
plt.title(f"Efficiency Distribution (Fixed {fixed_edges} Edges)")
plt.xlabel("Efficiency")
plt.ylabel("Number of Graphs")
plt.tight_layout()
plt.savefig(f"{output_dir}/efficiency_hist_{timestamp}.png")
plt.show()

print(f"Saved results and histogram to {csv_out}")

best_row = df.loc[df["Efficiency"].idxmax()]
worst_row = df.loc[df["Efficiency"].idxmin()]

best_id = int(best_row["Topology_ID"])
worst_id = int(worst_row["Topology_ID"])

print(f"\nBest Topology ID: {best_id} → Efficiency: {best_row['Efficiency']:.4f}")
print(f"Worst Topology ID: {worst_id} → Efficiency: {worst_row['Efficiency']:.4f}")
3
# Regenerate the best and worst graphs
G_best = generate_graph_fixed_edges(best_id, fixed_edges)
G_worst = generate_graph_fixed_edges(worst_id, fixed_edges)
def export_connections_to_csv(G, topo_id, label):
    rows = []
    for node in G.nodes():
        for neighbor in G.neighbors(node):
            if node < neighbor:  # avoid duplicate edges
                rows.append({"Source": node, "Target": neighbor})
    df_edges = pd.DataFrame(rows)
    out_path = f"{output_dir}/topology_{label.lower()}_{topo_id}.csv"
    df_edges.to_csv(out_path, index=False)
    print(f"Saved {label} topology edges to {out_path}")

# Export both
export_connections_to_csv(G_best, best_id, "Best")
export_connections_to_csv(G_worst, worst_id, "Worst")

