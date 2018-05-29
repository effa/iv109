from random import randint
import numpy as np
import pandas as pd
from models import get_nstudents, proxy_objective_solved_tasks
from simulation import simulate_system


def evaluate_system(
        proxy_objective=proxy_objective_solved_tasks,
        noise=0,
        n_iters_system=50,
        n_systems=10,
        n_students=50,
        patience=3):
    n_anomalies = 0
    for _i_system in range(n_systems):
        for i_student in range(get_nstudents()):
            results = simulate_system(
                proxy_objective=proxy_objective,
                i_student=i_student,
                n_iters=n_iters_system,
                n_students=n_students,
                noise=noise,
                patience=patience
            )
            n_anomalies += int(
                np.mean(np.abs(results.thresholds - 0.5) > 0.49) > 0.1
            )
    n_experiments = n_systems * get_nstudents()
    return n_anomalies / n_experiments


def evaluate_system_once(
        proxy_objective=proxy_objective_solved_tasks,
        noise=0,
        n_iters_system=50,
        n_students=50,
        patience=3):
    i_student = randint(0, get_nstudents() - 1)
    results = simulate_system(
        proxy_objective=proxy_objective,
        i_student=i_student,
        n_iters=n_iters_system,
        n_students=n_students,
        noise=noise,
        patience=patience
    )
    anomaly = int(np.mean(np.abs(results.thresholds - 0.5) > 0.49) > 0.1)
    return anomaly
