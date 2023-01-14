
#%%

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os


OUTPUT_DIR = 'plots/route-stats/'

if not os.path.exists(OUTPUT_DIR):
   os.makedirs(OUTPUT_DIR)

CSV_PATH_KL = 'data/route-statistics/route statistics_1action_withKL.csv'
CSV_PATH_NO_KL = 'data/route-statistics/route statistics_1Action_withoutKL.csv'

# Configure matplotlib for LaTeX

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex", # make sure you have pdflatex installed or change the tex distribution
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

# Read file

df_kl = pd.read_csv(CSV_PATH_KL, sep=',')
df_no_kl = pd.read_csv(CSV_PATH_NO_KL, sep=',')

list_df = [df_kl, df_no_kl]

#%%
# PREPROCESSING

# Iterate through dfs
for df in list_df: 

    # Add episode as new column
    df['episode'] = df.index + 1

    # Calculate exponential moving averages (EMA)

    # Lane change - EMA
    df['lane_change_ema'] = df['Wrong lane change'].ewm(span=100).mean()

    # Total steps - EMA
    df['total_steps_ema'] = df['Max. distance traveled [steps]'].ewm(span=100).mean()

    # Cumulative reward - EMA
    #df['cumulative_reward_ema'] = df['Cumulativ reward'].ewm(span=100).mean()


# %% 
# Plot results

# Plot lane change ------------------------------------------

fig, ax = plt.subplots(1, 1)
ax.plot('episode', 'lane_change_ema', data=df_kl, label='with KL divergence')
ax.plot('episode', 'lane_change_ema', data=df_no_kl, label='without KL divergence')

# set labels (LaTeX can be used)
#plt.title(r'\textbf{Number of Incorrect Lane Changes for Carla Experiment}', fontsize=11)
ax.set_xlabel(r'\textbf{Episode Number}', fontsize=11)
ax.set_ylabel(r'\textbf{Number of Incorrect Lane Changes}', fontsize=11)
ax.grid(linestyle='--', linewidth=1)
ax.legend()

fig.savefig(OUTPUT_DIR +'wrong_lane.png', dpi=400, bbox_inches="tight")


# %% 
# Plot total steps -------------------------------------------

fig, ax = plt.subplots(1, 1)
ax.plot('episode', 'total_steps_ema', data=df_kl, label='with KL divergence')
ax.plot('episode', 'total_steps_ema', data=df_no_kl, label='without KL divergence')

# set labels (LaTeX can be used)
#plt.title(r'\textbf{Collision Test for Carla Experiment}', fontsize=11)
ax.set_xlabel(r'\textbf{Episode Number}', fontsize=11)
ax.set_ylabel(r'\textbf{Number of Agent Steps}', fontsize=11)
ax.grid(linestyle='--', linewidth=1)
ax.legend()

fig.savefig(OUTPUT_DIR +'steps.png', dpi=400, bbox_inches="tight")


#%% 
# Plot cumulative reward ---------------------------------------

fig, ax = plt.subplots(1, 1)
ax.plot('episode', 'Cumulative reward', data=df_kl)
ax.plot('episode', 'Cumulative reward', data=df_no_kl)

# set labels (LaTeX can be used)
#plt.title(r'\textbf{Cumulative Reward for Carla Experiment}', fontsize=11)

ax.set_xlabel(r'\textbf{Episode Number}', fontsize=11)
ax.set_ylabel(r'\textbf{Cumulative Reward}', fontsize=11)
ax.grid(linestyle='--', linewidth=1)
#ax.legend()
fig.savefig(OUTPUT_DIR + 'cumulative_reward.png', dpi=400, bbox_inches="tight")





# %%
