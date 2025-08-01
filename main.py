import numpy as np
import random
import pandas as pd
from io import StringIO

from numpy.polynomial import Polynomial
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import plotly.io as pio
from scipy.interpolate import interp1d

def main():
    print("Start PID Ki Optimization!")

if __name__ == "__main__":
    main()

raw_data = """
Ki,IAE,ISE,ITAE
1,48.2938,2407.98,21.646
5,19.8728,673.78,5.448
10,10.3745,337.6,1.642
20,5.1949,168.86,0.416
30,3.4616,112.61,0.185
40,2.5949,84.48,0.104
50,2.0749,67.61,0.066
60,1.7282,56.36,0.046
70,1.4806,48.32,0.034
80,1.2948,42.3,0.026
90,1.1504,37.61,0.02
100,1.0348,33.86,0.016
110,0.9403,30.8,0.0136
120,0.8615,28.24,0.0114
140,0.7377,24.22,0.00835
160,0.6448,21.21,0.00638
180,0.5726,18.86,0.00502
200,0.5148,16.98,0.00406
220,0.4675,15.45,0.00334
240,0.4281,14.17,0.0028
250,0.4108,13.61,0.00258
260,0.3948,13.09,0.00238
270,0.38,12.6,0.0022
280,0.3662,12.16,0.00205
290,0.3534,11.74,0.00191
300,0.3415,11.35,0.00178
310,0.3303,10.99,0.00166
320,0.3198,10.65,0.00156
330,0.31,10.33,0.00146
340,0.3007,10.03,0.00138
350,0.292,9.74,0.0013
360,0.2837,9.47,0.00123
370,0.2759,9.22,0.00116
380,0.2685,8.98,0.0011
390,0.2615,8.75,0.00104
400,0.2548,8.54,0.00099
500,0.2029,6.84,0.00062
600,0.1683,5.72,0.00043
700,0.1436,4.91,0.00031
800,0.125,4.31,0.00024
1000,0.099,3.46,0.00015
1200,0.0816,2.9,0.0001
1400,0.0692,2.5,7.2e-05
1600,0.0599,2.2,5.4e-05
2000,0.0469,1.78,3.3e-05
3000,0.0295,1.22,1.3e-05
4000,0.0209,0.96,6e-06
5000,0.0156,0.82,3e-06
6000,0.0122,0.75,1.7e-06
7000,0.0103,0.72,1.1e-06
8000,0.0113,0.73,1.5e-06
9000,0.0138,0.77,2.3e-06
"""

# Read the data, use StringIO to read plain text as CSV.
df = pd.read_csv(StringIO(raw_data))

# to check
print(df.head())

# Interpolation function
interp_func = interp1d(df["Ki"], df["ITAE"], kind='cubic', fill_value="extrapolate")

# --- COST FUNCTION ---
def cost_func(ki):
    if ki < df["Ki"].min() or ki > df["Ki"].max():
        return 1e6  # ceza
    return float(interp_func(ki))
# -----------------------------------------------------------------------------------------------------
# --- Particle Swarm Optimization 
# -----------------------------------------------------------------------------------------------------
# --- PSO PARAMETERS ---
n_particles = 30
n_iter = 100
w, c1, c2 = 0.5, 1.5, 1.5

# --- Output lists ---
all_Ki = []
all_ITAE = []

for run in range(10):
    
    positions = np.random.uniform(df["Ki"].min(), df["Ki"].max(), n_particles)
    velocities = np.zeros(n_particles)

    personal_best = positions.copy()
    personal_best_scores = np.array([cost_func(p) for p in positions])
    global_best = personal_best[np.argmin(personal_best_scores)]
    global_best_score = np.min(personal_best_scores)

    global_history = []
    cost_history_all = []

    for _ in range(n_iter):
        current_costs = []
        for i in range(n_particles):
            r1, r2 = np.random.rand(), np.random.rand()
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (personal_best[i] - positions[i])
                + c2 * r2 * (global_best - positions[i])
            )
            positions[i] += velocities[i]
            positions[i] = np.clip(positions[i], df["Ki"].min(), df["Ki"].max())
            score = cost_func(positions[i])
            current_costs.append(score)

            if score < personal_best_scores[i]:
                personal_best[i] = positions[i]
                personal_best_scores[i] = score

        if np.min(personal_best_scores) < global_best_score:
            global_best = personal_best[np.argmin(personal_best_scores)]
            global_best_score = np.min(personal_best_scores)

        global_history.append(global_best_score)
        cost_history_all.append(current_costs)

    # Record outputs
    optimal_Ki = global_best
    all_Ki.append(optimal_Ki)
    all_ITAE.append(global_best_score)

    print(f"Run {run+1}: Ki = {optimal_Ki}, ITAE = {global_best_score}")

# --- Mean values of Ki and ITAE ---
pso_ki = np.mean(all_Ki)
pso_itae = np.mean(all_ITAE)

# -----------------------------------------------------------------------------------------------------
# Random Search
# -----------------------------------------------------------------------------------------------------
# Parameters
n_trials = 200
n_runs = 10
Ki_range = (df["Ki"].min(), df["Ki"].max())

# outputs
all_rand_ki = []
all_rand_itae = []

for run in range(n_runs):
    best_Ki_ran = None
    best_ITAE_ran = float('inf')
    history_random = []

    for _ in range(n_trials):
        ki = random.uniform(*Ki_range)
        itae = float(interp_func(ki))
        history_random.append(itae)

        if itae < best_ITAE_ran:
            best_Ki_ran = ki
            best_ITAE_ran = itae

    all_rand_ki.append(best_Ki_ran)
    all_rand_itae.append(best_ITAE_ran)

    print(f"Run {run+1}: Ki = {best_Ki_ran}, ITAE = {best_ITAE_ran}")

# mean of the rand Ki and ITAE
mean_rand_ki = np.mean(all_rand_ki)
mean_rand_itae = np.mean(all_rand_itae)

rand_ki = best_Ki_ran
rand_itae = best_ITAE_ran

# -----------------------------------------------------------------------------------------------------
# Grid Search
# -----------------------------------------------------------------------------------------------------
# Parameters
n_runs = 10
n_points = 200
Ki_min, Ki_max = df["Ki"].min(), df["Ki"].max()

# Results
all_grid_ki = []
all_grid_itae = []

for run in range(n_runs):
    Ki_values = np.linspace(Ki_min, Ki_max, n_points)
    itae_values = [float(interp_func(ki)) for ki in Ki_values]

    min_index = np.argmin(itae_values)
    best_Ki_grid = Ki_values[min_index]
    best_ITAE_grid = itae_values[min_index]

    all_grid_ki.append(best_Ki_grid)
    all_grid_itae.append(best_ITAE_grid)

    print(f"Run {run+1}: Ki = {best_Ki_grid}, ITAE = {best_ITAE_grid}")

# mean of the results
mean_grid_ki = np.mean(all_grid_ki)
mean_grid_itae = np.mean(all_grid_itae)

# optimum outputs
grid_ki = best_Ki_grid
grid_itae = best_ITAE_grid

# -----------------------------------------------------------------------------------------------------
# Genetik Alghoritms
# -----------------------------------------------------------------------------------------------------
# Parameters
pop_sizes = [20, 30, 40, 50]
mutation_rates = [0.05, 0.1, 0.2, 0.3]
n_gen = 100
n_runs = 10

Ki_min, Ki_max = df["Ki"].min(), df["Ki"].max()
fitness = lambda ki: float(interp_func(ki)) if Ki_min <= ki <= Ki_max else 1e6

all_best_Ki = []
all_best_ITAE = []

for run in range(n_runs):
    run_results = []

    for pop_size in pop_sizes:
        for mutation_rate in mutation_rates:
            population = np.random.uniform(Ki_min, Ki_max, pop_size)
            best_ITAE = float("inf")
            best_Ki = None

            for gen in range(n_gen):
                scores = np.array([fitness(ind) for ind in population])

                min_idx = np.argmin(scores)
                if scores[min_idx] < best_ITAE:
                    best_Ki = population[min_idx]
                    best_ITAE = scores[min_idx]

                # Selection
                selected = []
                for _ in range(pop_size):
                    i1, i2 = np.random.randint(0, pop_size, 2)
                    winner = population[i1] if scores[i1] < scores[i2] else population[i2]
                    selected.append(winner)

                
                offspring = []
                for i in range(0, pop_size, 2):
                    if i + 1 < pop_size:
                        alpha = random.uniform(0, 1)
                        c1 = alpha * selected[i] + (1 - alpha) * selected[i + 1]
                        c2 = alpha * selected[i + 1] + (1 - alpha) * selected[i]
                        offspring.extend([c1, c2])
                    else:
                        offspring.append(selected[i])

                
                for i in range(pop_size):
                    if random.random() < mutation_rate:
                        offspring[i] += np.random.normal(0, 0.05 * (Ki_max - Ki_min))
                        offspring[i] = np.clip(offspring[i], Ki_min, Ki_max)

                population = np.array(offspring)

            run_results.append({
                "pop_size": pop_size,
                "mutation_rate": mutation_rate,
                "best_Ki": best_Ki,
                "best_ITAE": best_ITAE
            })

    
    best_in_run = min(run_results, key=lambda x: x["best_ITAE"])
    all_best_Ki.append(best_in_run["best_Ki"])
    all_best_ITAE.append(best_in_run["best_ITAE"])

    print(f"Run {run+1}: Ki = {best_in_run['best_Ki']}, ITAE = {best_in_run['best_ITAE']}")

# Mean of the Results
ga_ki = np.mean(all_best_Ki)
ga_itae = np.mean(all_best_ITAE)

# -----------------------------------------------------------------------------------------------------