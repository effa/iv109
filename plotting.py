import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Global plotting setting.
sns.set()
sns.set_context('poster')
pd.options.display.float_format = '{:.2f}'.format


def plot_student(student_df):
    xticks = list(range(len(student_df)))
    ax = student_df[['proxy_skill', 'true_skill']].plot(
        xticks=xticks)
    ax = student_df.threshold.plot(color='k', linestyle='--')
    student_df.answers.plot(ax=ax, marker='o', linestyle='')


def plot_system(results):
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(8, 8), sharex=True)
    results.thresholds.plot(ax=ax1, ylim=(0, 1.01), legend=True)
    results[['objectiveMinus', 'objectivePlus']].plot(ax=ax2)
    results.mastery.plot(ax=ax3, ylim=(0, 1.01), legend=True)
