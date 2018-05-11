from models import create_proxy_student, create_true_student
from models import LearningSystem

# TODO: simulation for single student and for the whole system
# (returning DF with results)

def simulate_single_student(i_student=0, threshold=0.95):
    als = LearningSystem(
        threshold=threshold, i_student=i_student)
    true_student = create_true_student(i_student)
    proxy_student = als.process_student(true_student)
    print(proxy_student.history)
