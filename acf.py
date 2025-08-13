##H-Bonds Autocorrelation function curve fitting
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import re
import os
import pandas as pd
from natsort import natsorted  # for natural sorting like 1,2,3..10

# Define the ExpDecay2 function
def exp_decay2(x, y0, x0, A1, t1, A2, t2):
    return y0 + A1 * np.exp(-(x - x0) / t1) + A2 * np.exp(-(x - x0) / t2)

# Function to load data from .xvg
def load_xvg(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            if not line.startswith(('#', '@')):
                parts = re.split(r'\s+', line.strip())
                if len(parts) >= 2:
                    data.append((float(parts[0]), float(parts[1])))
    return np.array(data)

# Initial guesses
initial_guesses = [0.00117, -2.44, 1.58, 1.75, 0.72, 13.36]

# Get all .xvg files in /content/
xvg_files = [f for f in os.listdir('/content') if f.endswith('.xvg')]

# Store results
summary_data = []

for file in xvg_files:
    data = load_xvg(f'/content/{file}')
    x_data, y_data = data[:,0], data[:,1]

    try:
        popt, pcov = curve_fit(exp_decay2, x_data, y_data, p0=initial_guesses, maxfev=10000)
        perr = np.sqrt(np.diag(pcov))

        # Plot
        x_fit = np.linspace(min(x_data), max(x_data), 500)
        y_fit = exp_decay2(x_fit, *popt)

        plt.figure(figsize=(8,6))
        plt.plot(x_data, y_data, label='Data', color='blue', linewidth=1.5)
        plt.plot(x_fit, y_fit, label='Fitted Curve', color='red', linestyle='--', linewidth=2)
        plt.xscale('log')
        plt.xlabel('Time (log scale)')
        plt.ylabel('Autocorrelation')
        plt.title(f'Double Exponential Fit: {file}')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.show()

        # Save the results
        summary_data.append({
            'File': file,
            'tau1': popt[3], 'tau1_err': perr[3],
            'tau2': popt[5], 'tau2_err': perr[5],
            'y0': popt[0], 'y0_err': perr[0],
            'x0': popt[1], 'x0_err': perr[1],
            'A1': popt[2], 'A1_err': perr[2],
            'A2': popt[4], 'A2_err': perr[4]
        })

    except RuntimeError as e:
        print(f"Fit could not converge for {file}: {e}")

# Create DataFrame for tabular display
summary_df = pd.DataFrame(summary_data)

# Sort DataFrame using natural sort on 'File'
summary_df = summary_df.set_index('File')
summary_df = summary_df.loc[natsorted(summary_df.index)].reset_index()

print("\n================ TABULAR SUMMARY OF FIT RESULTS ================\n")
pd.set_option('display.float_format', lambda x: '%.5f' % x)
print(summary_df.to_string(index=False))
