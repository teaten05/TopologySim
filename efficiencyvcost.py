import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- PARAMETERS ---
csv_path = "kuramoto_results_20250715_111705.csv"
N = 100
edges_max = N * (N - 1) / 2

# --- Load Data ---
df = pd.read_csv(csv_path)

# --- Find r(t) of global network (alltoall) ---
#global_rows = df[df["Topology_Type"] == "alltoall"]
#if global_rows.empty:
 #   raise ValueError("No alltoall topology found in the CSV. You need at least one to define efficiency.")

r_global = 0.986

# --- Compute Cost and Efficiency ---
df["Cost"] = df["Edges"] / edges_max
df["Efficiency"] = df["Final_r(t)"] / r_global

# --- Save new CSV ---
output_csv = csv_path.replace(".csv", "_efficiency_analysis.csv")
df.to_csv(output_csv, index=False)

# --- Scatter Plot: Efficiency vs Cost ---
sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="Cost", y="Efficiency", hue="Topology_Type", alpha=0.7)
plt.title("Efficiency vs Cost")
plt.xlabel("Cost (Edges / Max Edges)")
plt.ylabel("Efficiency (r_random / r_global)")
plt.ylim(0, 1.05)
plt.xlim(0.15, 0.55)
plt.axhline(1.0, color="gray", linestyle="--", label="Global Reference")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("efficiency_vs_cost.png")
plt.show()
