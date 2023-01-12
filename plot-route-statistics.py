
#%%

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os


if not os.path.exists('plots'):
   os.makedirs('plots')

CSV_PATH = 'data/route_statistics_2023-01-11-14-44.csv'

# Configure matplotlib for LaTeX

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex", # make sure you have pdflatex installed or change the tex distribution
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

# Read file

df = pd.read_csv(CSV_PATH, sep=',')
# df

# Add episode as new column

df['episode'] = df.index + 1

#%% 
# Plot cumulative reward

fig, ax = plt.subplots(1, 1)
ax.plot('episode', 'Cumulative reward', data=df)

# set labels (LaTeX can be used)
#plt.title(r'\textbf{Cumulative Reward for Carla Experiment}', fontsize=11)

ax.set_xlabel(r'\textbf{Episode Number}', fontsize=11)
ax.set_ylabel(r'\textbf{Cumulative Reward}', fontsize=11)
ax.grid(linestyle='--', linewidth=1)
#ax.legend()
fig.savefig('plots/cumulative_reward.png', dpi=400, bbox_inches="tight")

# %% 
# Plot lane change

fig, ax = plt.subplots(1, 1)
ax.plot('episode', 'Wrong lane change', data=df)

# set labels (LaTeX can be used)
#plt.title(r'\textbf{Number of Incorrect Lane Changes for Carla Experiment}', fontsize=11)
ax.set_xlabel(r'\textbf{Episode Number}', fontsize=11)
ax.set_ylabel(r'\textbf{Number of Incorrect Lane Changes}', fontsize=11)
ax.grid(linestyle='--', linewidth=1)
#ax.legend()

fig.savefig('plots/wrong_lane.png', dpi=400, bbox_inches="tight")

# %% 
# Plot collision

fig, ax = plt.subplots(1, 1)
ax.plot('episode', 'Collision', data=df)

# set labels (LaTeX can be used)
#plt.title(r'\textbf{Collision Test for Carla Experiment}', fontsize=11)
ax.set_xlabel(r'\textbf{Episode Number}', fontsize=11)
ax.set_ylabel(r'\textbf{Collison}', fontsize=11)
ax.grid(linestyle='--', linewidth=1)
#ax.legend()

fig.savefig('plots/collision.png', dpi=400, bbox_inches="tight")


# %%
