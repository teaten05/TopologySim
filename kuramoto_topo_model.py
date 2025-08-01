import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from copy import deepcopy
import pandas as pd
import json

# === Parameters ===
N = 8  # number of drones
POP_SIZE = 30
GENERATIONS = 50
MUTATION_RATE = 0.05
K = 0.6
dt = 0.0125
Tmax = 60
timesteps = int(Tmax / dt)

# === Kuramoto Simulator ===
def kuramoto_r(G, omega, theta0):
    theta = theta0.copy()
    A = nx.to_numpy_array(G)
    r_values = []

    for _ in range(timesteps):
        r = np.abs(np.mean(np.exp(1j * theta)))
        r_values.append(r)
        theta_dot = np.zeros(N)
        for i in range(N):
            sum_sin = 0
            for j in range(N):
                if i != j and A[i, j] == 1:
                    sum_sin += np.sin(theta[j] - theta[i])
            theta_dot[i] = omega[i] + (K / N) * sum_sin
        theta += theta_dot * dt

    return r_values[-1]

# === Individual / Adjacency Matrix Encoding ===
def random_adj_matrix():
    A = np.random.randint(0, 2, size=(N, N))
    A = np.triu(A, 1)
    A = A + A.T
    np.fill_diagonal(A, 0)
    return A

def mutate_adj(A):
    A_new = A.copy()
    for i in range(N):
        for j in range(i+1, N):
            if random.random() < MUTATION_RATE:
                A_new[i, j] = 1 - A_new[i, j]
                A_new[j, i] = A_new[i, j]
    return A_new

def crossover_adj(A1, A2):
    A_new = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(i+1, N):
            A_new[i, j] = A1[i, j] if random.random() < 0.5 else A2[i, j]
            A_new[j, i] = A_new[i, j]
    return A_new

def is_connected(A):
    G = nx.from_numpy_array(A)
    return nx.is_connected(G)

# === GA Logging ===
generation_log = []

# === Main GA Loop ===
population = [random_adj_matrix() for _ in range(POP_SIZE)]
fitness_history = []
best_graph = None
best_r = -1

for gen in range(GENERATIONS):
    fitness = []
    for A in population:
        if not is_connected(A):
            fitness.append(0)
            continue
        G = nx.from_numpy_array(A)
        omega = 1.0 + 0.1 * np.random.randn(N)
        theta0 = 2 * np.pi * np.random.rand(N)
        final_r = kuramoto_r(G, omega, theta0)
        fitness.append(final_r)
        if final_r > best_r:
            best_r = final_r
            best_graph = deepcopy(G)

    # Track fitness
    fitness_history.append(max(fitness))

    # Rank population by fitness
    ranked = sorted(zip(fitness, population), key=lambda x: x[0], reverse=True)

    gen_best_r = ranked[0][0]
    best_individual = ranked[0][1]
    best_edges = list(nx.from_numpy_array(best_individual).edges())

    print(f"Generation {gen+1}: Best r(t) = {gen_best_r:.4f}")

    # Log best of this generation
    generation_log.append({
        "Generation": gen + 1,
        "Best_r(t)": gen_best_r,
        "Edge_Count": len(best_edges),
        "Edges": best_edges
    })

    # Selection: top 50%
    parents = [deepcopy(ind) for _, ind in ranked[:POP_SIZE // 2]]

    # Generate new population
    new_population = []
    while len(new_population) < POP_SIZE:
        p1, p2 = random.sample(parents, 2)
        child = crossover_adj(p1, p2)
        child = mutate_adj(child)
        new_population.append(child)
    population = new_population

# === Plot Results ===
plt.figure()
plt.plot(fitness_history, 'b-o')
plt.xlabel("Generation")
plt.ylabel("Best r(t)")
plt.title("Best r(t) Over Generations")
plt.grid(True)
plt.tight_layout()
plt.savefig("fitness_evolution.png")
plt.show()

# === Plot Best Graph ===
plt.figure(figsize=(5, 5))
nx.draw_circular(best_graph, with_labels=True, node_color='skyblue', edge_color='gray')
plt.title(f"Best Topology (r(t) = {best_r:.4f})")
plt.savefig("best_topology.png")
plt.show()

# === Save Best Topology ===
A_best = nx.to_numpy_array(best_graph, dtype=int)
np.savetxt("best_topology_adj.csv", A_best, fmt="%d", delimiter=",")

with open("best_topology_edges.txt", "w") as f:
    for edge in best_graph.edges():
        f.write(f"{edge[0]},{edge[1]}\n")

data = nx.node_link_data(best_graph)
with open("best_topology.json", "w") as f:
    json.dump(data, f)

# === Save generation log to CSV ===
df_log = pd.DataFrame(generation_log)
# Convert list of edges to string for CSV readability
df_log["Edges"] = df_log["Edges"].apply(lambda x: "; ".join([f"{e[0]}-{e[1]}" for e in x]))
df_log.to_csv("generation_best_log.csv", index=False)

print("\nBest topology saved as:")
print("- best_topology_adj.csv (Adjacency Matrix)")
print("- best_topology_edges.txt (Edge List)")
print("- best_topology.json (Full Graph Structure)")
print("- generation_best_log.csv (Generation best r(t) and topology log)")
