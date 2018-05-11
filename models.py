from random import random, randint, choice


class ProxyStudent:
    def __init__(self, p_learn, p_good, init_skill=0.1):
        """
        Args:
            p_learn = prabability of getting learned after 1 question
            p_good = [probability of good performance for skill=0, -||- skill=1]
        """
        self.p_learn = p_learn
        self.p_good = p_good
        self.skill = init_skill

    def update_skill(self, performance):
        """Called when student answers a question."""

        # Performance at skill: p_perf_at_skill[performance][skill]
        p_perf_at_skill = [list(map(lambda x: 1 - x, self.p_good)), self.p_good]

        likelihood_good = self.skill * p_perf_at_skill[performance][1]
        likelihood_bad = (1 - self.skill) * p_perf_at_skill[performance][0]

        # New skill according to Bayes theorem
        self.skill = (likelihood_good / (likelihood_good + likelihood_bad))

        # Learning
        self.skill += (1 - self.skill) * self.p_learn


class TrueStudent:
    def __init__(self, p_init, p_learn, p_good, max_answers=20):
        """
        Args:
            p_learn = prabability of getting learned after 1 question
            p_good = [probability of good performance for skill=0, -||- skill=1]
        """
        self.p_learn = p_learn
        self.p_good = p_good
        self.skill = int(random() < p_init)
        self.n_answers = 0
        self.max_answers = max_answers

    def update_skill(self):
        """Called when student answers a question."""
        if self.skill == 1:
            return
        if random() < self.p_learn:
            self.skill = 1

    def answer(self):
        """Return a performance for a new question"""
        self.n_answers += 1
        self.update_skill()
        p_good = self.p_good[self.skill]
        performance = int(random() < p_good)
        return performance

    def leave(self, mastery_decision):
        return mastery_decision or self.n_answers >= self.max_answers


class LearningSystem:
    def __init__(self, threshold, proxy_objective, threshold_delta=0.05):
        self.threshold = threshold
        self.objectives = None
        self.proxy_objective = proxy_objective
        self.threshold_delta = threshold_delta

    def do_iteration(self, true_students):
        """
        Performs the simulation for a single month.
        Will affect local attributes. (threshold)
        """
        thresholds = [self.threshold - self.threshold_delta,
                      self.threshold + self.threshold_delta]
        groups = [true_students[:len(true_students) // 2],
                  true_students[len(true_students) // 2:]]

        objectives = []
        for group, t in zip(groups, thresholds):
            students = [self.process_student(s, t) for s in group]
            objectives.append(self.proxy_objective(students))

        if objectives[1] >= objectives[0]:
            self.threshold = min(1 - self.threshold_delta, self.threshold + self.threshold_delta)
        else:
            self.threshold = max(self.threshold_delta, self.threshold - self.threshold_delta)
        self.objectives = objectives #max(objectives)

    def process_student(self, true_student, threshold=None):
        threshold = threshold if threshold is not None else self.threshold
        # Intentional, TODO: think about it again.
        proxy_student = ProxyStudent(true_student.p_learn, true_student.p_good)
        mastery = False
        history = []

        while not true_student.leave(mastery):
            performance = true_student.answer()
            proxy_student.update_skill(performance)
            mastery = proxy_student.skill >= threshold

            history.append(performance)

        proxy_student.history = history
        return proxy_student
