# Gotta change this whole thing to fit the results of only one topology
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV
df = pd.read_csv("kuramoto_results_20250715_111705.csv")
sns.set(style="whitegrid")

# --- Top 10 Topologies by Final r(t) ---
top10 = df.sort_values("Final_r(t)", ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(data=top10, x="Final_r(t)", y="Topology_ID", hue="Topology_Type", dodge=False)
plt.title("Top 10 Topologies by Final r(t)")
plt.xlabel("Final Synchronization (r(t))")
plt.ylabel("Topology ID")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

# --- Scatter Plot: Final r(t) vs. Average Degree ---
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="Avg_Degree", y="Final_r(t)", hue="Topology_Type", alpha=0.7)
plt.title("Final r(t) vs. Average Degree")
plt.xlabel("Average Degree")
plt.ylabel("Final r(t)")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

# --- Average r(t) by Topology Type ---
mean_r_by_type = df.groupby("Topology_Type")["Final_r(t)"].mean().sort_values(ascending=False).reset_index()
plt.figure(figsize=(8, 6))
sns.barplot(data=mean_r_by_type, x="Final_r(t)", y="Topology_Type", palette="magma")
plt.title("Average Final r(t) by Topology Type")
plt.xlabel("Average r(t)")
plt.ylabel("Topology Type")
plt.tight_layout()
plt.show()

# --- Top Low-Connectivity Topologies (Avg Degree < 3) ---
low_conn_top = df[df["Avg_Degree"] < 3].sort_values("Final_r(t)", ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(data=low_conn_top, x="Final_r(t)", y="Topology_ID", hue="Topology_Type", dodge=False)
plt.title("Top Low-Connectivity Topologies (Avg Degree < 3)")
plt.xlabel("Final Synchronization (r(t))")
plt.ylabel("Topology ID")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()
