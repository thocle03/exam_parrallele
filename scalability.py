import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv("results.csv",
                 names=["MPI", "OMP", "Time"],
                 header=None)

# Remove header if duplicated
df = df[df["MPI"] != "mpi_ranks"]

df["MPI"] = df["MPI"].astype(int)
df["OMP"] = df["OMP"].astype(int)
df["Time"] = df["Time"].astype(float)

# Plot scalability
plt.figure(figsize=(8,5))

for mpi in sorted(df["MPI"].unique()):
    subset = df[df["MPI"] == mpi]
    plt.plot(subset["OMP"], subset["Time"], marker='o', label=f"MPI={mpi}")

plt.xlabel("Nombre de threads OpenMP")
plt.ylabel("Temps d'exécution (s)")
plt.title("Scalabilité hybride MPI / OpenMP")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
