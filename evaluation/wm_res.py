import matplotlib.pyplot as plt
import numpy as np

# Data
n = np.array([1, 10, 50, 100, 150, 200])

# Stage 1
mse_s1 = [0.0023, 0.0974, 2.0676, 2.6689, 2.8113, 2.9058]
cos_s1 = [0.0012, 0.0473, 0.7167, 0.9016, 0.9412, 0.9681]

# Stage 2
mse_s2 = [0.0075, 0.1269, 0.2728, 0.3642, 0.3824, 0.3654]
cos_s2 = [0.0027, 0.0401, 0.1268, 0.2233, 0.2311, 0.2461]

# Stage 3
mse_s3 = [0.0017, 0.0145, 0.0334, 0.0608, 0.0891, 0.1157]
cos_s3 = [0.0009, 0.0078, 0.0157, 0.0247, 0.0360, 0.0491]

# Full episode (Task 1)
full_mse = [2.9089, 0.4352, 0.1554]
full_cos = [0.9701, 0.3160, 0.0734]

stages = ["Stage 1", "Stage 2", "Stage 3"]

# -----------------------------
# Figure 1: MSE vs Horizon
# -----------------------------
plt.figure(figsize=(6,4))
plt.plot(n, mse_s1, 'o-', label='Stage 1')
plt.plot(n, mse_s2, 's-', label='Stage 2')
plt.plot(n, mse_s3, '^-', label='Stage 3')

plt.xlabel("Rollout Horizon (n)")
plt.ylabel("MSE ↓")
plt.title("Open-loop Latent Prediction (Task 2)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("mse_vs_horizon.pdf")
plt.show()

# -----------------------------
# Figure 2: Cosine Distance vs Horizon
# -----------------------------
plt.figure(figsize=(6,4))
plt.plot(n, cos_s1, 'o-', label='Stage 1')
plt.plot(n, cos_s2, 's-', label='Stage 2')
plt.plot(n, cos_s3, '^-', label='Stage 3')

plt.xlabel("Rollout Horizon (n)")
plt.ylabel("Cosine Distance ↓")
plt.title("Open-loop Latent Alignment (Task 2)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("cosine_vs_horizon.pdf")
plt.show()

# -----------------------------
# Figure 3: Full Episode Comparison
# -----------------------------
x = np.arange(len(stages))

plt.figure(figsize=(5,4))
plt.bar(x - 0.15, full_mse, width=0.3, label='MSE')
plt.bar(x + 0.15, full_cos, width=0.3, label='Cosine Distance')

plt.xticks(x, stages)
plt.ylabel("Error ↓")
plt.title("Full Episode Rollout (Task 1)")
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("full_episode.pdf")
plt.show()