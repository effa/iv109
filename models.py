from collections import OrderedDict, namedtuple
from random import random, randint, choice
import numpy as np


class ProxyStudent:
    def __init__(self, p_init, p_learn, p_good):
        """
        Args:
            p_learn = prabability of getting learned after 1 question
            p_good = [probability of good performance for skill=0, -||- skill=1]
        """
        self.p_learn = p_learn
        self.p_good = p_good
        self.skill = p_init
        self.skills_history = [self.skill]

    def update_skill(self, performance):
        """Called when student answers a question."""

        # Performance at skill: p_perf_at_skill[performance][skill]
        p_perf_at_skill = [list(map(lambda x: 1 - x, self.p_good)), self.p_good]

        # TODO: fix names ("likelihod" is only the second term)
        likelihood_good = self.skill * p_perf_at_skill[performance][1]
        likelihood_bad = (1 - self.skill) * p_perf_at_skill[performance][0]

        # New skill according to Bayes theorem
        self.skill = (likelihood_good / (likelihood_good + likelihood_bad))

        # Learning
        self.skill += (1 - self.skill) * self.p_learn

        self.skills_history.append(self.skill)


class TrueStudent:
    def __init__(self, p_init, p_learn, p_good, max_answers=20):
        """
        Args:
            p_learn = prabability of getting learned after 1 question
            p_good = [probability of good performance for skill=0, -||- skill=1]
        """
        self.p_init = p_init
        self.p_learn = p_learn
        self.p_good = p_good
        self.skill = int(random() < p_init)
        self.n_answers = 0
        self.max_answers = max_answers
        self.skills_history = [self.skill]
        self.answers_history = []

    def update_skill(self):
        """Called when student answers a question."""
        if self.skill == 0 and random() < self.p_learn:
            self.skill = 1
        self.skills_history.append(self.skill)

    def answer(self):
        """Return a performance for a new question"""
        self.n_answers += 1
        self.update_skill()
        p_good = self.p_good[self.skill]
        performance = int(random() < p_good)
        self.answers_history.append(performance)
        return performance

    def leave(self, mastery_decision):
        if sum(self.answers_history[-3:]) == 3:
            return True
        return mastery_decision or self.n_answers >= self.max_answers


def proxy_objective_solved_tasks(students):
    return sum([len(s.history) for s in students])


def proxy_objective_successful_tasks(students):
    return sum([sum(s.history) for s in students])


def proxy_objective_target_skill(students, target=0.8):
    return sum([-abs(s.skill - target) for s in students])


def proxy_objective_effective_tasks(students):
    return sum([s.history.count(0) - s.history.count(1) for s in students])


class LearningSystem:
    def __init__(self, i_student=0, threshold=0.5,
                 proxy_objective=None, threshold_delta=0.05):
        self.threshold = threshold
        self.objectives = None
        self.proxy_objective = proxy_objective
        self.threshold_delta = threshold_delta
        self.i_student = i_student  # we assume to know the population params

    def do_iteration(self, true_students, noise=0):
        """
        Performs the simulation for a single month.
        Will affect local attributes. (threshold)
        """
        assert self.proxy_objective is not None
        thresholds = [
            max(0, self.threshold - self.threshold_delta),
            min(1, self.threshold + self.threshold_delta)]
        groups = [true_students[:len(true_students) // 2],
                  true_students[len(true_students) // 2:]]

        objectives = []
        for group, t in zip(groups, thresholds):
            students = [self.process_student(s, t, noise=noise) for s in group]
            objectives.append(self.proxy_objective(students))

        if objectives[1] >= objectives[0]:
            self.threshold = min(1, self.threshold + self.threshold_delta)
        else:
            self.threshold = max(0, self.threshold - self.threshold_delta)
        self.objectives = objectives #max(objectives)

    def process_student(self, true_student, threshold=None, noise=0):
        threshold = threshold if threshold is not None else self.threshold
        # We assume optimistic scenario of knowing the BKT params the
        # population of students:
        proxy_student = create_proxy_student(self.i_student, noise=noise)
        mastery = False
        history = []

        while not true_student.leave(mastery):
            performance = true_student.answer()
            proxy_student.update_skill(performance)
            mastery = proxy_student.skill >= threshold

            history.append(performance)

        proxy_student.history = history
        return proxy_student


BktParams = namedtuple('BktParams', ['p_init', 'p_learn', 'p_good'])
BKT_PARAMS = [
    BktParams(p_init=0.1, p_learn=0.1, p_good=(0.1, 0.9)),
    BktParams(p_init=0.2, p_learn=0.2, p_good=(0.3, 0.8)),
    BktParams(p_init=0.4, p_learn=0.15, p_good=(0.5, 0.9)),
    BktParams(p_init=0.4, p_learn=0.55, p_good=(0.3, 0.7)),
    BktParams(p_init=0.5, p_learn=0.55, p_good=(0.1, 0.8)),
    BktParams(p_init=0.6, p_learn=0.15, p_good=(0.3, 0.7)),
]
N_STUDENTS = len(BKT_PARAMS)


def get_nstudents():
    return len(BKT_PARAMS)


#def create_proxy_student(true_student):
#    return ProxyStudent(
#        p_init=true_student.p_init,
#        p_learn=true_student.p_learn,
#        p_good=true_student.p_good)

def create_proxy_student(i, noise=0.2):
    params = {}
    params['p_init'] = np.mean([p.p_init for p in BKT_PARAMS])
    params['p_learn'] = np.mean([p.p_learn for p in BKT_PARAMS])
    params['p_good'] = (np.mean([p.p_good[0] for p in BKT_PARAMS]),
                        np.mean([p.p_good[1] for p in BKT_PARAMS]))
    #params = BKT_PARAMS[i]._asdict()
    params['p_init'] = min(params['p_init'] + noise, 0.99)
    params['p_learn'] = min(params['p_learn'] + noise, 0.99)
    params['p_good'] = (min(params['p_good'][0] + noise, 0.99),
                        max(params['p_good'][1] - noise, 0.01))
    #params['p_good'] = (max(params['p_good'][0] - noise, 0.01),
    #                    min(params['p_good'][1] + noise, 0.99))
    return ProxyStudent(**params)

def create_true_student(i):
    return TrueStudent(**BKT_PARAMS[i]._asdict())
