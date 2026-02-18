import matplotlib.pyplot as plt

# -----------------------------
# FIGURE 2 LEFT — Initial Trust × Mobility
# -----------------------------
initial_trust = [0.4, 0.5, 0.6, 0.7, 0.8]
fig2_left = {
    1:  [0.000, 0.000, 0.120, 1.000, 1.000],
    2:  [0.000, 0.000, 0.560, 1.000, 1.000],
    5:  [0.000, 0.000, 0.990, 1.000, 1.000],
    10: [0.000, 0.000, 1.000, 1.000, 1.000],
}

# -----------------------------
# FIGURE 2 RIGHT — Share Trustworthy × Mobility
# -----------------------------
share_trustworthy = [0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4]
fig2_right = {
    1:  [1.000, 1.000, 1.000, 0.850, 0.100, 0.000, 0.000],
    2:  [1.000, 1.000, 1.000, 1.000, 0.600, 0.150, 0.000],
    5:  [1.000, 1.000, 1.000, 1.000, 1.000, 0.550, 0.050],
    10: [1.000, 1.000, 1.000, 1.000, 1.000, 0.950, 0.100],
}

# -----------------------------
# FIGURE 4 LEFT — Sensitivity × Mobility
# -----------------------------
sensitivity = [0.04, 0.06, 0.08, 0.10]
fig4_left = {
    1:  [0.650, 0.000, 0.000, 0.000],
    2:  [0.950, 0.250, 0.000, 0.000],
    5:  [1.000, 0.850, 0.200, 0.000],
    10: [1.000, 0.950, 0.350, 0.150],
}

# -----------------------------
# FIGURE 5 LEFT — Shocks × Untrustworthy × Mobility
# X-axis is share of UNtrustworthy agents
# -----------------------------
untrustworthy = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]

fig5_left = {
    1:  [1.000, 1.000, 0.200, 0.000, 0.000, 0.000, 0.000],
    2:  [1.000, 1.000, 0.750, 0.000, 0.000, 0.000, 0.000],
    5:  [1.000, 1.000, 1.000, 0.100, 0.000, 0.000, 0.000],
    10: [1.000, 1.000, 0.950, 0.150, 0.000, 0.000, 0.000],
}
# -----------------------------
# NEW FIGURE — CDEM Size × Mobility × Initial Trust
# -----------------------------

cdem_sizes = [5, 10, 15, 20]

cdem_results = {
    0.5: {   # initial trust = 0.5
        1:  [0.050, 0.150, 0.350, 0.400],
        2:  [0.000, 0.000, 0.050, 0.050],
        5:  [0.000, 0.000, 0.000, 0.000],
        10: [0.000, 0.000, 0.000, 0.000],
    },
    0.6: {   # initial trust = 0.6
        1:  [1.000, 1.000, 1.000, 1.000],
        2:  [1.000, 1.000, 1.000, 1.000],
        5:  [1.000, 1.000, 1.000, 1.000],
        10: [1.000, 1.000, 1.000, 1.000],
    }
}

ddem_sizes = [5, 10, 15, 20]

ddem_decay_results = {
    0.5: {   # initial trust = 0.5
        1:  [0.000, 0.000, 0.000, 0.000],
        2:  [0.000, 0.000, 0.000, 0.000],
        5:  [0.000, 0.000, 0.000, 0.000],
        10: [0.000, 0.000, 0.000, 0.000],
    },
    0.6: {   # initial trust = 0.6
        1:  [0.000, 0.000, 0.000, 0.000],
        2:  [0.400, 0.350, 0.350, 0.350],
        5:  [1.000, 1.000, 1.000, 1.000],
        10: [1.000, 1.000, 1.000, 1.000],
    }
}
# -----------------------------
# PLOTTING
# -----------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharey=True)

# ---- FIG 2 LEFT ----
for mobility, values in fig2_left.items():
    axes[0, 0].plot(initial_trust, values, marker='o', label=f"M={mobility}")
axes[0, 0].set_title("Fig 2 Left: Initial Trust × Mobility")
axes[0, 0].set_xlabel("Initial Trust Mean")
axes[0, 0].set_ylabel("Share of Trusting Outcomes")
axes[0, 0].set_ylim(-0.05, 1.05)
axes[0, 0].grid(True, linestyle='--', alpha=0.6)

# ---- FIG 2 RIGHT ----
for mobility, values in fig2_right.items():
    axes[0, 1].plot(share_trustworthy, values, marker='s', label=f"M={mobility}")
axes[0, 1].set_title("Fig 2 Right: Share Trustworthy × Mobility")
axes[0, 1].set_xlabel("Initial Share of Trustworthy Agents")
axes[0, 1].grid(True, linestyle='--', alpha=0.6)

# ---- FIG 4 LEFT ----
for mobility, values in fig4_left.items():
    axes[1, 0].plot(sensitivity, values, marker='^', label=f"M={mobility}")
axes[1, 0].set_title("Fig 4 Left: Sensitivity × Mobility")
axes[1, 0].set_xlabel("Sensitivity to New Information")
axes[1, 0].set_ylabel("Share of Trusting Outcomes")
axes[1, 0].grid(True, linestyle='--', alpha=0.6)

# ---- FIG 5 LEFT ----
for mobility, values in fig5_left.items():
    axes[1, 1].plot(untrustworthy, values, marker='D', label=f"M={mobility}")
axes[1, 1].set_title("Fig 5 Left: Shock Robustness")
axes[1, 1].set_xlabel("Initial Share of Untrustworthy Agents")
axes[1, 1].grid(True, linestyle='--', alpha=0.6)

# Shared legend
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4)

plt.suptitle("Trust ABM Verification — Figures 2, 4, and 5 Replication", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("trust_abm_verification.pdf", format="pdf", bbox_inches="tight", dpi=300)

# =============================
# FIGURE — CDEM RESULTS
# =============================

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for ax, init_trust in zip(axes, sorted(cdem_results.keys())):
    
    for mobility, values in cdem_results[init_trust].items():
        ax.plot(
            cdem_sizes,
            values,
            marker='o',
            linewidth=2,
            label=f"M={mobility}"
        )

    ax.set_title(f"CDEM Effects (Initial Trust = {init_trust})")
    ax.set_xlabel("CDEM Size")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.6)

axes[0].set_ylabel("Share of Trusting Outcomes")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4)

plt.suptitle("CDEM Size × Mobility × Initial Trust", fontsize=15)
plt.tight_layout(rect=[0, 0, 1, 0.90])

plt.savefig("c დემ_results.pdf", bbox_inches="tight", dpi=300)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for ax, init_trust in zip(axes, sorted(ddem_decay_results.keys())):
    
    for mobility, values in ddem_decay_results[init_trust].items():
        ax.plot(
            ddem_sizes,
            values,
            marker='s',
            linewidth=2,
            label=f"M={mobility}"
        )

    ax.set_title(f"DDEM + Decay (Initial Trust = {init_trust})")
    ax.set_xlabel("DDEM Size")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.6)

axes[0].set_ylabel("Share of Trusting Outcomes")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4)

plt.suptitle("DDEM Decay × Mobility × Initial Trust", fontsize=15)
plt.tight_layout(rect=[0, 0, 1, 0.90])

plt.savefig("ddem_decay_results.pdf", bbox_inches="tight", dpi=300)
plt.show()