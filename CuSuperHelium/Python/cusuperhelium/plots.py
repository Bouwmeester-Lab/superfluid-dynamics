import numpy as np

def plot_phase_diagram(values_y, T, base_time, ax_pd, ax_ts, label="Detuning", n_arrows=5, lw=0.8, arrow_step=5):
    ydot = np.gradient(values_y, T * base_time)
    line, = ax_pd.plot(values_y, ydot, label=f"{label}", lw=0.8)
    color = line.get_color()
    idx = np.linspace(0, len(values_y) - arrow_step - 1, n_arrows, dtype=int)
    for i in idx:
        ax_pd.annotate(
            "",
            xy=(values_y[i + arrow_step], ydot[i + arrow_step]),
            xytext=(values_y[i], ydot[i]),
            arrowprops=dict(
                arrowstyle="->",
                color=color,
                lw=lw,
                shrinkA=0,
                shrinkB=0,
            ),
        )
    ax_ts.plot(T * base_time * 1e6, values_y, label=f"{label}", lw=lw)