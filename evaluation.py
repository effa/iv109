import numpy as np
import pandas as pd
from models import N_STUDENTS, proxy_objective_solved_tasks
from simulation import simulate_system


def evaluate_system(
        proxy_objective=proxy_objective_solved_tasks,
        n_iters_system=50,
        n_systems=10,
        n_students=50):
    n_anomalies = 0
    for _i_system in range(n_systems):
        for i_student in range(N_STUDENTS):
            results = simulate_system(
                proxy_objective=proxy_objective,
                i_student=i_student,
                n_iters=n_iters_system,
                n_students=n_students
            )
            n_anomalies += int(
                np.mean(np.abs(results.thresholds - 0.5) > 0.49) > 0.5
            )
    n_experiments = n_systems * N_STUDENTS
    return n_anomalies / n_experiments
