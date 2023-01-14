
#%%
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os


OUTPUT_DIR = 'plots/rllib-results/'
if not os.path.exists(OUTPUT_DIR):
   os.makedirs(OUTPUT_DIR)

CSV_PATH_1 = 'data/rllib-sample-results/progress1/progress.csv'
CSV_PATH_2 = 'data/rllib-sample-results/progress2/progress.csv'

# Configure matplotlib for LaTeX

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex", # make sure you have pdflatex installed or change the tex distribution
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

# Read file

df_rllib1 = pd.read_csv(CSV_PATH_1, sep=',')
df_rllib2 = pd.read_csv(CSV_PATH_2, sep=',')
df_rllib2 = df_rllib2.iloc[0:len(df_rllib1),:] # remove the last part to have equal size

# x axis: timesteps_total

# %% 
# Episode reward mean: episode_reward_mean

fig, ax = plt.subplots(1, 1)
ax.plot('timesteps_total', 'episode_reward_mean', data=df_rllib1, label='Model finetuned on Carla data')
ax.plot('timesteps_total', 'episode_reward_mean', data=df_rllib2, label='Model only trained on BDD100K data')

# set labels (LaTeX can be used)
#plt.title(r'\textbf{Cumulative Reward for Carla Experiment}', fontsize=11)
ax.set_xlabel(r'\textbf{Total Timesteps}', fontsize=11)
ax.set_ylabel(r'\textbf{Episode Reward Mean}', fontsize=11)
ax.grid(linestyle='--', linewidth=1)
ax.legend()
fig.savefig(OUTPUT_DIR + 'episode_reward_mean.png', dpi=400, bbox_inches="tight")



# %% 
# Episode len mean

fig, ax = plt.subplots(1, 1)
ax.plot('timesteps_total', 'episode_len_mean', data=df_rllib1, label='Model finetuned on Carla data')
ax.plot('timesteps_total', 'episode_len_mean', data=df_rllib2, label='Model only trained on BDD100K data')

# set labels (LaTeX can be used)
#plt.title(r'\textbf{Cumulative Reward for Carla Experiment}', fontsize=11)
ax.set_xlabel(r'\textbf{Total Timesteps}', fontsize=11)
ax.set_ylabel(r'\textbf{Episode Length Mean}', fontsize=11)
ax.grid(linestyle='--', linewidth=1)
ax.legend()
fig.savefig(OUTPUT_DIR + 'episode_len_mean.png', dpi=400, bbox_inches="tight")


# %% 
# info/learner/default_policy/learner_stats/vf_loss	

col = 'info/learner/default_policy/learner_stats/vf_loss'

fig, ax = plt.subplots(1, 1)
ax.plot('timesteps_total', col, data=df_rllib1, label='Model finetuned on Carla data')
ax.plot('timesteps_total', col, data=df_rllib2, label='Model only trained on BDD100K data')

# set labels (LaTeX can be used)
#plt.title(r'\textbf{Cumulative Reward for Carla Experiment}', fontsize=11)
ax.set_xlabel(r'\textbf{Total Timesteps}', fontsize=11)
ax.set_ylabel(r'\textbf{Value Function Loss}', fontsize=11)
ax.grid(linestyle='--', linewidth=1)
ax.legend()
fig.savefig(OUTPUT_DIR + 'vf_loss.png', dpi=400, bbox_inches="tight")



# %% 
# info/learner/default_policy/learner_stats/entropy	

col = 'info/learner/default_policy/learner_stats/entropy'

fig, ax = plt.subplots(1, 1)
ax.plot('timesteps_total', col, data=df_rllib1, label='Model finetuned on Carla data')
ax.plot('timesteps_total', col, data=df_rllib2, label='Model only trained on BDD100K data')

# set labels (LaTeX can be used)
#plt.title(r'\textbf{Cumulative Reward for Carla Experiment}', fontsize=11)
ax.set_xlabel(r'\textbf{Total Timesteps}', fontsize=11)
ax.set_ylabel(r'\textbf{Entropy}', fontsize=11)
ax.grid(linestyle='--', linewidth=1)
ax.legend()
fig.savefig(OUTPUT_DIR + 'entropy.png', dpi=400, bbox_inches="tight")


# %% 
# Variance
# for variance, subtract the offset from entropy to have the same x-axis
# e.g. 1.4

col = 'info/learner/default_policy/learner_stats/entropy'


fig, ax = plt.subplots(1, 1)
ax.plot(df_rllib1['timesteps_total'], df_rllib1[col] - 1.4, label='Model finetuned on Carla data')
ax.plot(df_rllib2['timesteps_total'], df_rllib2[col] - 1.4, label='Model only trained on BDD100K data')

# set labels (LaTeX can be used)
#plt.title(r'\textbf{Cumulative Reward for Carla Experiment}', fontsize=11)
ax.set_xlabel(r'\textbf{Total Timesteps}', fontsize=11)
ax.set_ylabel(r'\textbf{Variance}', fontsize=11)
ax.grid(linestyle='--', linewidth=1)
ax.legend()
fig.savefig(OUTPUT_DIR + 'variance.png', dpi=400, bbox_inches="tight")

# %%
