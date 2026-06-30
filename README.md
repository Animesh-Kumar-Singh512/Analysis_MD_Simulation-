# Analysis_MD_Simulation-
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import re
import os
import pandas as pd
from natsort import natsorted

# ============================================================
# Origin Pro ExpDecay2 model (IDENTICAL mathematical form)
# y = y0 + A1exp(-(x-x0)/t1) + A2exp(-(x-x0)/t2)
# ============================================================
def exp_decay2_origin(x, y0, x0, A1, t1, A2, t2):
    return y0 + A1 * np.exp(-(x - x0) / t1) + A2 * np.exp(-(x - x0) / t2)

# ============================================================
# Load .xvg file (from gmx hbond -ac)
# ============================================================
def load_xvg(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            if not line.startswith(('#', '@')):
                parts = re.split(r'\s+', line.strip())
                if len(parts) >= 2:
                    data.append((float(parts[0]), float(parts[1])))
    return np.array(data)

# ============================================================
# Initial guesses (similar scale to Origin)
# ============================================================
initial_guesses = [0.001, -2.0, 1.5, 1.0, 0.7, 10.0]

# ============================================================
# Directory containing .xvg files
# ============================================================
xvg_dir = '/content/sample_data'
xvg_files = natsorted([f for f in os.listdir(xvg_dir) if f.endswith('.xvg')])

summary_data = []

if not xvg_files:
    print("No .xvg files found.")
else:
    for file in xvg_files:
        data = load_xvg(os.path.join(xvg_dir, file))

        # Remove t <= 0 (important for log scale and stability)
        data = data[data[:, 0] > 0]

        x_data, y_data = data[:, 0], data[:, 1]

        try:
            # ====================================================
            # Curve fitting (Levenberg–Marquardt)
            # ====================================================
            popt, pcov = curve_fit(
                exp_decay2_origin,
                x_data,
                y_data,
                p0=initial_guesses,
                maxfev=50000
            )

            perr = np.sqrt(np.diag(pcov))

            # ====================================================
            # Dense log-spaced curve (KEY to Origin-like tracing)
            # ====================================================
            x_fit = np.logspace(np.log10(x_data.min()),
                                np.log10(x_data.max()),
                                2000)
            y_fit = exp_decay2_origin(x_fit, *popt)

            # ====================================================
            # Plot
            # ====================================================
            plt.figure(figsize=(8, 6))
            plt.plot(x_data, y_data, 'o', markersize=4, label='Data')
            plt.plot(x_fit, y_fit, '-', linewidth=2, label='ExpDecay2 Fit')
            plt.xscale('log')
            plt.xlabel('Time')
            plt.ylabel('Autocorrelation')
            plt.title(f'ExpDecay2 Fit (Origin-equivalent): {file}')
            plt.legend()
            plt.grid(True, which='both', linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.show()

            # ====================================================
            # Store results
            # ====================================================
            summary_data.append({
                'File': file,
                'y0': popt[0], 'y0_err': perr[0],
                'x0': popt[1], 'x0_err': perr[1],
                'A1': popt[2], 'A1_err': perr[2],
                't1': popt[3], 't1_err': perr[3],
                'A2': popt[4], 'A2_err': perr[4],
                't2': popt[5], 't2_err': perr[5],
            })

        except RuntimeError as e:
            print(f"Fit failed for {file}: {e}")

# ============================================================
# Tabular summary (like Origin report)
# ============================================================
summary_df = pd.DataFrame(summary_data)

print("\n================ ORIGIN-STYLE FIT SUMMARY ================\n")
pd.set_option('display.float_format', lambda x: '%.6f' % x)

if not summary_df.empty:
    summary_df = summary_df.set_index('File')
    summary_df = summary_df.loc[natsorted(summary_df.index)].reset_index()
    print(summary_df.to_string(index=False))
else:
    print("No successful fits.")
