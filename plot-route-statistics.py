
#%%

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import os


# Configure matplotlib for LaTeX

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex", # make sure you have pdflatex installed or change the tex distribution
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

#%%
# For 2 Actions

OUTPUT_DIR = 'plots/route-stats/Shawan/2-actions/' # add '/' at the end

if not os.path.exists(OUTPUT_DIR):
   os.makedirs(OUTPUT_DIR)

# Set csv paths
csv_decDRL_IL_early = 'data/route-statistics/2Actions/decDRL_IL_early.csv'
csv_decDRL_IL = 'data/route-statistics/2Actions/decDRL_IL.csv'
csv_decDRL = 'data/route-statistics/2Actions/Home_2Action_noMimic_NoDF.csv'
csv_decDRL_IL_late = 'data/route-statistics/2Actions/Home_2Actions_mimic_DF_late.csv'

# Read csv files
df_decDRL_IL_early = pd.read_csv(csv_decDRL_IL_early, sep=',')
df_decDRL_IL = pd.read_csv(csv_decDRL_IL, sep=',')
df_decDRL = pd.read_csv(csv_decDRL, sep=',')
df_decDRL_IL_late = pd.read_csv(csv_decDRL_IL_late, sep=',')


# Create a dictionary to store dfs with their correct name
dict_df = {
    #'decDRL': df_decDRL,
    'decDRL+IL': df_decDRL_IL,
    'decDRL+IL+early': df_decDRL_IL_early, 
    'decDRL+IL+late': df_decDRL_IL_late
}


#%% 
# For 1 Action

OUTPUT_DIR = 'plots/route-stats/Shawan/1-action/' # add '/' at the end

if not os.path.exists(OUTPUT_DIR):
   os.makedirs(OUTPUT_DIR)

# 1 Action
csv_decoupled = 'data/route-statistics/1Action/1Action_decoupled.csv'
csv_vanilla ='data/route-statistics/1Action/1Action_vanilla.csv'

# Read csv files
df_decoupled = pd.read_csv(csv_decoupled, sep=',')
df_vanilla = pd.read_csv(csv_vanilla, sep=',')



# Create a dictionary to store dfs with their correct name
dict_df = {
    'DRL': df_vanilla,
    'decDRL': df_decoupled
}


#%%
# PREPROCESSING

# Iterate through dfs
for key, df in dict_df.items(): 

    # Add episode as new column
    df['episode'] = df.index + 1

    # Calculate exponential moving averages (EMA)

    # Lane change - EMA
    df['lane_change_ema'] = df['Wrong lane change'].ewm(span=100).mean()

    # Total steps - EMA
    df['total_steps_ema'] = df['Max. distance traveled [steps]'].ewm(span=100).mean()

    # Cumulative reward - EMA
    df['mean_reward_ema'] = df['Cumulativ reward'].ewm(span=100).mean()

    # Delta acceleration - EMA
    df['delta_acceleration_ema'] = df['Avg. speed compliance'].ewm(span=100).mean()


#%% Crop dfs to same length 

final_size = 5200 # 5200 # 2000 # 5000

for key, df in dict_df.items(): 

    if len(df) > final_size: # 5500:
        dict_df[key] = df.iloc[0:final_size]

    print('len ', key, ': ', len(dict_df[key]))


# %% 
# Plot results

# Plot cumulative reward ---------------------------------------

fig, ax = plt.subplots(1, 1)

for key, df in dict_df.items():
    ax.plot('episode', 'mean_reward_ema', data=df, label=key)

# set labels (LaTeX can be used)
#plt.title(r'\textbf{Cumulative Reward for Carla Experiment}', fontsize=11)

ax.set_xlabel(r'\textbf{Episodes}', fontsize=22)
ax.set_ylabel(r'\textbf{Mean Reward}', fontsize=22)
ax.grid(linestyle='--', linewidth=1)
#leg = ax.legend(fontsize=18) #, framealpha=0) #, frameon=False)
#leg.get_frame().set_linewidth(0.5)

plt.xticks(fontsize=22)
plt.yticks(fontsize=22)

#leg = plt.legend()
#leg.get_frame().set_linewidth(0.0)
#ax.legend(fontsize=30) #, frameon=False)

leg = ax.legend(fontsize=18)
leg.get_frame().set_edgecolor('k')
leg.get_frame().set_linewidth(0.0)

tick_spacing = 1000

ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
#ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

fig.savefig(OUTPUT_DIR + 'mean_reward.png', dpi=400, bbox_inches="tight")



#%%

# Plot lane invasion ------------------------------------------

fig, ax = plt.subplots(1, 1)

for key, df in dict_df.items():
    ax.plot('episode', 'lane_change_ema', data=df, label=key)


# set labels (LaTeX can be used)
#plt.title(r'\textbf{Number of Incorrect Lane Changes for Carla Experiment}', fontsize=11)
ax.set_xlabel(r'\textbf{Episodes}', fontsize=22)
ax.set_ylabel(r'\textbf{Lane Invasions}', fontsize=22)
ax.grid(linestyle='--', linewidth=1)
ax.legend(fontsize=22)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)

leg = ax.legend(fontsize=18)
leg.get_frame().set_edgecolor('k')
leg.get_frame().set_linewidth(0.0)

tick_spacing = 1000

ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
#ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

fig.savefig(OUTPUT_DIR +'wrong_lane.png', dpi=400, bbox_inches="tight")


# %% 
# Plot total steps -------------------------------------------

fig, ax = plt.subplots(1, 1)

for key, df in dict_df.items():
    ax.plot('episode', 'total_steps_ema', data=df, label=key)

# set labels (LaTeX can be used)
#plt.title(r'\textbf{Collision Test for Carla Experiment}', fontsize=11)
ax.set_xlabel(r'\textbf{Episodes}', fontsize=22)
ax.set_ylabel(r'\textbf{Steps}', fontsize=22)
ax.grid(linestyle='--', linewidth=1)
ax.legend(fontsize=22)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)

leg = ax.legend(fontsize=18, loc='lower right')
leg.get_frame().set_edgecolor('k')
leg.get_frame().set_linewidth(0.0)

tick_spacing = 1000

ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
#ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

fig.savefig(OUTPUT_DIR +'steps.png', dpi=400, bbox_inches="tight")



# %%
# Plot delta acceleration ---------------------------------------

fig, ax = plt.subplots(1, 1)

for key, df in dict_df.items():
    ax.plot('episode', 'delta_acceleration_ema', data=df, label=key)

# set labels (LaTeX can be used)
#plt.title(r'\textbf{Cumulative Reward for Carla Experiment}', fontsize=11)

ax.set_xlabel(r'\textbf{Episodes}', fontsize=22)
ax.set_ylabel(r'\textbf{$ \Delta$Acc}', fontsize=22)
ax.grid(linestyle='--', linewidth=1)
ax.legend(fontsize=22)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)

leg = ax.legend(fontsize=18)
leg.get_frame().set_edgecolor('k')
leg.get_frame().set_linewidth(0.0)

tick_spacing = 1000

ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
#ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

fig.savefig(OUTPUT_DIR + 'delta_acceleration.png', dpi=400, bbox_inches="tight")

# %%
