import os
import xgi
from src.algorithms.sem import sem_functions
import numpy as np
import csv


sem = sem_functions()

s_intial = np.array([1, 2, 1, 2, 0.5, 0.5, 0.5, 0.5])
initial_rate = 0.01
constant = 0.001
iteration_limit = 8000
n_replicates = 20
n_nodes = 2000

base = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]

# x-axis (properties of the edge copy step)
weak_cop = [0.6, 0.4]
strong_cop = [0.9, 0.2]
none_high_cop = [0.9, 0.9]
none_low_cop = [0.1, 0.1]


# y-axis (properties of the ext and novel node addition steps)
null_ext = [0.5, 0.5]
weak_ext = [1, 0.8]
strong_ext = [2, 0.2]
none_high_ext = [2, 2]
none_low_ext = [0.1, 0.1]

null_nov = [0.5, 0.5]
weak_nov = [1, 0.8]
strong_nov = [2, 0.2]
none_high_nov = [2, 2]
none_low_nov = [0.1, 0.1]


# weak homophily in external node addition step
g1 = weak_cop + weak_ext + null_nov
g2 = strong_cop + weak_ext + null_nov
g3 = none_high_cop + weak_ext + null_nov
g4 = none_low_cop + weak_ext + null_nov

# strong homophily in external node addition step
g5 = weak_cop + strong_ext + null_nov
g6 = strong_cop + strong_ext + null_nov
g7 = none_high_cop + strong_ext + null_nov
g8 = none_low_cop + strong_ext + null_nov

# weak homophily in novel node addition step
g9 = weak_cop + null_ext + weak_nov
g10 = strong_cop + null_ext + weak_nov
g11 = none_high_cop + null_ext + weak_nov
g12 = none_low_cop + null_ext + weak_nov

# strong homophily in novel node addition step
g13 = weak_cop + null_ext + strong_nov
g14 = strong_cop + null_ext + strong_nov
g15 = none_high_cop + null_ext + strong_nov
g16 = none_low_cop + null_ext + strong_nov

# weak homophily in both steps
g17 = weak_cop + weak_ext + weak_nov
g18 = strong_cop + weak_ext + weak_nov
g19 = none_high_cop + weak_ext + weak_nov
g20 = none_low_cop + weak_ext + weak_nov

# strong homophily in both steps
g21 = weak_cop + strong_ext + strong_nov
g22 = strong_cop + strong_ext + strong_nov
g23 = none_high_cop + strong_ext + strong_nov
g24 = none_low_cop + strong_ext + strong_nov

# no homophily, high external node params
g25 = weak_cop + none_high_ext + null_nov
g26 = strong_cop + none_high_ext + null_nov
g27 = none_high_cop + none_high_ext + null_nov
g28 = none_low_cop + none_high_ext + null_nov

# no homophily, high novel node params
g29 = weak_cop + null_ext + none_high_nov
g30 = strong_cop + null_ext + none_high_nov
g31 = none_high_cop + null_ext + none_high_nov
g32 = none_low_cop + null_ext + none_high_nov

# no homophily, high ext and novel node params
g33 = weak_cop + none_high_ext + none_high_nov
g34 = strong_cop + none_high_ext + none_high_nov
g35 = none_high_cop + none_high_ext + none_high_nov
g36 = none_low_cop + none_high_ext + none_high_nov

# no homophily, low ext and novel node params
g37 = weak_cop + none_low_ext + none_low_nov
g38 = strong_cop + none_low_ext + none_low_nov
g39 = none_high_cop + none_low_ext + none_low_nov
g40 = none_low_cop + none_low_ext + none_low_nov


true_thetas = [g1, g2, g3, g4, g5,
               g6, g7, g8, g9, g10,
               g11, g12, g13, g14, g15,
               g16, g17, g18, g19, g20,
               g21, g22, g23, g24, g25,
               g26, g27, g28, g29, g30,
               g31, g32, g33, g34, g35,
               g36, g37, g38, g39, g40]


def run_one(i, true_theta):
    """Generate n_replicates independent hypergraphs for parameter set i,
    run SEM on each, and write each replicate's estimates to its own CSV."""
    print(f"Graph {i}")
    for rep in range(1, n_replicates + 1):
        print(f"  Replicate {rep}/{n_replicates}")
        GH = sem.generate_hypergraph(true_theta, n_nodes)
        estimates = sem.SEM_without_likelihood(
            GH, s_intial, iteration_limit, initial_rate, constant
        )
        out_path = f"throughput/synthetic_results/graph{i}_rep{rep}_sem_ests_extended.csv"
        with open(out_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(estimates)


if __name__ == "__main__":
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")

    if task_id is not None:
        # Running as one task of a SLURM job array: process only the
        # parameter set corresponding to this array index (1-based, matches
        # g1..g40 / --array=1-40).
        idx = int(task_id)
        true_theta = true_thetas[idx - 1]
        run_one(idx, true_theta)
    else:
        # Fallback for local/interactive runs: process every parameter set
        # sequentially (useful for testing, but slow — prefer the job array
        # on the HPCC).
        for i, true_theta in enumerate(true_thetas, start=1):
            run_one(i, true_theta)