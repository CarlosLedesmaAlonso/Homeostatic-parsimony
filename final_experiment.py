import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

# Intentar usar estilo seaborn para gráficos bonitos; si no, usar estilo default
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
except ImportError:
    plt.style.use("ggplot")

# --- CONFIGURACIÓN DEL EXPERIMENTO ---
SEEDS = 20           # Número de réplicas para robustez estadística
STEPS = 1000         # Duración del episodio
SHOCK_STEP = 500     # Momento del shock vital
HIDDEN_DIM = 32
LR_BASE = 0.01
TEMP_BASE = 1.0
RELIEF_COEF = 10.0   # Coeficiente de recompensa por alivio (Δdrive)

# Parámetros del criterio de recuperación
BASELINE_WINDOW = 100     # Ventana pre-shock para estimar baseline
RECOVERY_MARGIN = 0.10    # target = baseline + margin
RECOVERY_CAP = 0.35       # target = min(cap, baseline + margin)

# Configuración de Gráficos
OUTPUT_DIR = "."


# --- 1. ENTORNO (Micro-Mundo Homeostático) ---
class SocialEnv:
    def __init__(self):
        self.state = np.array([0.5, 0.5], dtype=np.float32)

    def reset(self):
        self.state = np.array([0.5, 0.5], dtype=np.float32)
        return self.state.copy()

    def step(self, action):
        reward = 0.0
        cost = np.zeros(2, dtype=np.float32)

        # 0=TRABAJAR (Gana recursos, pierde social/energía)
        if action == 0:
            reward = 1.0
            self.state[0] += 0.1
            self.state[1] -= 0.1
            cost = np.array([-0.1, 0.1], dtype=np.float32)

        # 1=SOCIALIZAR (Gasta recursos, gana social)
        elif action == 1:
            reward = 0.2
            self.state[0] -= 0.05
            self.state[1] += 0.15
            cost = np.array([-0.1, -0.2], dtype=np.float32)

        # 2=DESCANSAR (Recuperación fuerte)
        elif action == 2:
            reward = 0.0
            cost = np.array([0.30, 0.02], dtype=np.float32)

        self.state = np.clip(self.state, 0, 1)
        return self.state.copy(), reward, cost


# --- 2. AGENTE (Arquitectura Neuronal) ---
class HomeostaticAgent(nn.Module):
    def __init__(self):
        super().__init__()
        # Input 4D: [Recurso, Social, Energía_Interna, Integridad_Interna]
        self.net = nn.Sequential(
            nn.Linear(4, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 3)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=LR_BASE)

        # Estado interno y setpoint
        self.h_t = np.array([1.0, 1.0], dtype=np.float32)
        self.h_star = np.array([1.0, 1.0], dtype=np.float32)

    def update_physiology(self, cost):
        self.h_t[0] += cost[0]
        self.h_t[1] -= cost[1]
        self.h_t -= 0.01  # Metabolismo basal
        self.h_t = np.clip(self.h_t, 0, 1.0)

    def get_drive(self):
        # Drive = distancia media al setpoint
        # Drive = mean distance to setpoint
        return float(np.mean(np.abs(self.h_star - self.h_t)))


# --- 3. BUCLE DE SIMULACIÓN ---
groups = ["FULL", "REG-NO-MOD", "NO-REG"]

raw_drives = {g: np.zeros((SEEDS, STEPS), dtype=np.float32) for g in groups}
stats_summary = {g: {"rec_time": [], "auc": []} for g in groups}

print(f"Iniciando simulación ({SEEDS} semillas)...")
print(f"Criterio de recuperación: target = min({RECOVERY_CAP}, baseline + {RECOVERY_MARGIN}), "
      f"baseline=media de {BASELINE_WINDOW} pasos pre-shock.")

for seed in range(SEEDS):
    torch.manual_seed(seed)
    np.random.seed(seed)

    for group in groups:
        env = SocialEnv()
        agent = HomeostaticAgent()
        ext_state = env.reset()

        prev_drive = agent.get_drive()

        for step in range(STEPS):
            # A) INDUCCIÓN DEL SHOCK (solo para condiciones con cuerpo operativo)
            # Shock is applied only to embodied conditions
            if step == SHOCK_STEP and group in ["FULL", "REG-NO-MOD"]:
                agent.h_t = np.array([0.1, 0.1], dtype=np.float32)
                prev_drive = agent.get_drive()  # reset para Δdrive

            # B) PERCEPCIÓN (Interocepción)
            if group == "NO-REG":
                # Baseline externo: canal interno "clamped" y sin fisiología operativa
                # External-only baseline: internal channel clamped; no operative physiology
                int_p = np.array([1.0, 1.0], dtype=np.float32)
            else:
                int_p = agent.h_t.copy()

            state_t = torch.FloatTensor(np.concatenate([ext_state, int_p]))

            # C) MODULACIÓN
            D_t = agent.get_drive()
            g_t = np.clip(D_t, 0, 1) if group == "FULL" else 0.0

            curr_lr = LR_BASE
            curr_temp = TEMP_BASE
            logit_bias = torch.zeros(3)

            if group == "FULL" and g_t > 0.7:
                # MODO PÁNICO / RIGIDEZ (global modulation + crisis override)
                curr_temp = 0.1           # visión de túnel
                curr_lr = LR_BASE * 5.0   # plasticidad alta
                logit_bias[2] = 10.0      # sesgo fuerte hacia REST
            elif group == "FULL":
                # Yerkes-Dodson suave fuera de pánico
                mod_lr = 4 * g_t * (1 - g_t)
                curr_lr = LR_BASE * (1.0 + 5.0 * mod_lr)
                curr_temp = max(0.5, TEMP_BASE * (1.0 - 0.6 * g_t))

            for pg in agent.optimizer.param_groups:
                pg["lr"] = curr_lr

            # D) SELECCIÓN DE ACCIÓN
            logits = agent.net(state_t) + logit_bias
            probs = torch.softmax(logits / curr_temp, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            # E) EJECUCIÓN
            ext_state, r_ext, cost = env.step(action.item())
            if group != "NO-REG":
                agent.update_physiology(cost)

            # F) APRENDIZAJE (Gradiente de Alivio)
            curr_drive = agent.get_drive()

            if group == "NO-REG":
                # Baseline externo (sin criterio interno de valor)
                tot_r = r_ext
            else:
                # Recompensa = Externo + (Alivio * coef)
                # Alivio = D_{t-1} - D_t
                delta = prev_drive - curr_drive
                tot_r = r_ext + (RELIEF_COEF * delta)

            loss = -dist.log_prob(action) * tot_r
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()

            prev_drive = curr_drive
            raw_drives[group][seed, step] = curr_drive

        # --- MÉTRICAS POST-EPISODIO ---
        # Nota: NO-REG NO es comparable como "recuperación homeostática" en este script.
        if group == "NO-REG":
            stats_summary[group]["rec_time"].append(np.nan)
            stats_summary[group]["auc"].append(np.nan)
            continue

        d_post = raw_drives[group][seed, SHOCK_STEP:]

        # Baseline pre-shock
        if SHOCK_STEP > BASELINE_WINDOW:
            pre = raw_drives[group][seed, SHOCK_STEP-BASELINE_WINDOW:SHOCK_STEP]
            baseline = float(np.mean(pre))
        else:
            baseline = 0.0

        target = min(RECOVERY_CAP, baseline + RECOVERY_MARGIN)
        rec_idx = np.where(d_post < target)[0]

        if len(rec_idx) > 0:
            t = int(rec_idx[0])
            auc = float(np.sum(d_post[:t+1]))
        else:
            t = STEPS - SHOCK_STEP
            auc = float(np.sum(d_post))

        stats_summary[group]["rec_time"].append(t)
        stats_summary[group]["auc"].append(auc)

print("Simulación completa. Generando figuras...")

# --- 4. GRÁFICOS ---
colors = {
    "NO-REG": "#95a5a6",      # Gris
    "FULL": "#e74c3c",        # Rojo
    "REG-NO-MOD": "#2ecc71"   # Verde
}

labels = {
    "NO-REG": "External-only baseline (NO-REG; not comparable)",
    "FULL": "Global modulation + panic override (FULL)",
    "REG-NO-MOD": "Interoception + relief gradient (REG-NO-MOD)"
}

# FIGURA 1: series temporal (solo FULL vs REG-NO-MOD para evitar confusión)
plt.figure(figsize=(10, 6), dpi=300)
zoom_start, zoom_end = 400, 900
time_axis = np.arange(zoom_start, zoom_end)

for g in ["FULL", "REG-NO-MOD"]:
    mean_drive = np.mean(raw_drives[g], axis=0)[zoom_start:zoom_end]
    std_drive = np.std(raw_drives[g], axis=0)[zoom_start:zoom_end]
    plt.plot(time_axis, mean_drive, label=labels[g], color=colors[g], linewidth=2.5)
    plt.fill_between(time_axis, mean_drive - std_drive, mean_drive + std_drive, color=colors[g], alpha=0.2)

plt.axvline(x=SHOCK_STEP, color="black", linestyle="--", alpha=0.7, label=f"Critical shock (t={SHOCK_STEP})")
plt.title("Homeostatic Recovery Dynamics (mean ± SD, n=20 seeds)", fontsize=14, fontweight="bold")
plt.xlabel("Simulation steps", fontsize=12)
plt.ylabel("Drive (distress)  D(t) = mean(|h* - h(t)|)", fontsize=12)

plt.text(
    zoom_start + 5,
    plt.ylim()[1] * 0.92,
    f"Recovery threshold: target = min({RECOVERY_CAP}, baseline + {RECOVERY_MARGIN})",
    fontsize=9,
    alpha=0.8
)

plt.legend(loc="upper right", frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Figure1_Recovery_Dynamics.png"))
print("-> Figure 1 guardada: Figure1_Recovery_Dynamics.png")

# FIGURA 2: barras (recuperación)
plt.figure(figsize=(8, 6), dpi=300)

groups_plot = ["FULL", "REG-NO-MOD"]  # NO-REG no es comparable como recuperación homeostática aquí
metrics_mean = []
metrics_std = []

for g in groups_plot:
    vals = np.array(stats_summary[g]["rec_time"], dtype=np.float32)
    metrics_mean.append(float(np.nanmean(vals)))
    metrics_std.append(float(np.nanstd(vals)))

x_pos = np.arange(len(groups_plot))
bars = plt.bar(
    x_pos,
    metrics_mean,
    yerr=metrics_std,
    align="center",
    alpha=0.85,
    color=[colors[g] for g in groups_plot],
    capsize=10,
    width=0.6
)

plt.xticks(x_pos, [labels[g] for g in groups_plot], fontsize=10, rotation=10, ha="right")
plt.ylabel("Time to recovery (steps)", fontsize=12)
plt.title("Recovery Efficiency After Critical Shock (mean ± SD)", fontsize=14, fontweight="bold")

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 5,
        f"{height:.1f}",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold"
    )

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Figure2_Recovery_Time.png"))
print("-> Figure 2 guardada: Figure2_Recovery_Time.png")

# --- RESUMEN FINAL ---
print("\n" + "=" * 90)
print(f"{'DATOS FINALES PARA EL MANUSCRITO (mean ± SD over seeds)':^90}")
print("=" * 90)

for g in groups:
    if g == "NO-REG":
        print(f"{g:<12} | Rec: N/A (baseline) | AUC: N/A (baseline)")
        continue

    rec_vals = np.array(stats_summary[g]["rec_time"], dtype=np.float32)
    auc_vals = np.array(stats_summary[g]["auc"], dtype=np.float32)

    rec_m = float(np.nanmean(rec_vals))
    rec_s = float(np.nanstd(rec_vals))
    auc_m = float(np.nanmean(auc_vals))
    auc_s = float(np.nanstd(auc_vals))

    print(f"{g:<12} | Rec: {rec_m:.1f} ± {rec_s:.1f} | AUC: {auc_m:.1f} ± {auc_s:.1f}")

print("=" * 90)
print("Nota: NO-REG es un baseline externo en este script; no se interpreta como recuperación homeostática.")

