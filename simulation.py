from models import create_proxy_student, create_true_student
from models import LearningSystem
from models import proxy_objective_solved_tasks
import numpy as np
import pandas as pd

def simulate_student(i_student=0, threshold=0.9):
    als = LearningSystem(
        threshold=threshold, i_student=i_student)
    true_student = create_true_student(i_student)
    proxy_student = als.process_student(true_student)
    results = pd.DataFrame({
        'answers': proxy_student.history + [None],  # align length
        'proxy_skill': proxy_student.skills_history,
        'true_skill': true_student.skills_history,
        'threshold': [threshold] * len(true_student.skills_history),
    })
    return results


def simulate_system(
        proxy_objective=proxy_objective_solved_tasks,
        noise=0,
        i_student=0,
        n_iters=50,
        n_students=50):
    als = LearningSystem(i_student=i_student, proxy_objective=proxy_objective)
    thresholds = []
    objectivePlus, objectiveMinus = [], []
    mastery = []
    for i_iter in range(n_iters):
        true_students = create_students(n_students, i_student)
        als.do_iteration(true_students, noise=noise)
        thresholds.append(als.threshold)
        mastery.append(np.mean([s.n_answers < s.max_answers for s in true_students]))
        objectiveMinus.append(als.objectives[0])
        objectivePlus.append(als.objectives[1])
    results = pd.DataFrame({
        'thresholds': thresholds,
        'mastery': mastery,
        'objectiveMinus': objectiveMinus,
        'objectivePlus': objectivePlus})
    return results


def create_students(n_students, i_student):
    return [
        create_true_student(i_student)
        for _ in range(n_students)
    ]
