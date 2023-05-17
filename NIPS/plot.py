
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
# Results csvs

OUTPUT_DIR = 'nips-plots/' # add '/' at the end

if not os.path.exists(OUTPUT_DIR):
   os.makedirs(OUTPUT_DIR)

# Set csv paths
csv_ae = 'AE-merged.csv'
csv_dpc = 'DPC-merged.csv'
csv_moco = 'MoCo-merged.csv'
csv_yolo = 'YOLOPv2-merged.csv'

# Read csv files
df_ae = pd.read_csv(csv_ae, sep=',')
df_dpc = pd.read_csv(csv_dpc, sep=',')
df_moco = pd.read_csv(csv_moco, sep=',')
df_yolo = pd.read_csv(csv_yolo, sep=',')

# Create a dictionary to store dfs with their correct name
dict_df = {
    'AE': df_ae,
    'DPC': df_dpc,
    'MoCo': df_moco,
    'YOLOPv2': df_yolo,
}


#%%
# PREPROCESSING
window_size = 100
# Iterate through dfs
for key, df in dict_df.items(): 

    # Add episode as new column
    df['episode'] = df.index + 1

    # Calculate exponential moving averages (EMA)

    # Lane change - EMA
    df['lane_change_ema'] = df['Wrong lane change'].ewm(span=window_size).mean()

    # Total steps - EMA
    df['total_steps_ema'] = df['Max. distance traveled [steps]'].ewm(span=window_size).mean()

    # Cumulative reward - EMA
    df['mean_reward_ema'] = df['Cumulativ reward'].ewm(span=window_size).mean() / 3

    # Delta acceleration - EMA
    df['delta_acceleration_ema'] = df['Avg. speed compliance'].ewm(span=window_size).mean()


#%% Crop dfs to same length 

final_size = 2000 # 5200 for 3 configs # 2000 for IL test # 5000 # 2000 fo 1 action

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

ax.set_xlabel(r'\textbf{Episodes}', fontsize=18)
ax.set_ylabel(r'\textbf{Mean Reward}', fontsize=18)
ax.grid(linestyle='--', linewidth=1)
#leg = ax.legend(fontsize=18) #, framealpha=0) #, frameon=False)
#leg.get_frame().set_linewidth(0.5)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

#leg = plt.legend()
#leg.get_frame().set_linewidth(0.0)
#ax.legend(fontsize=30) #, frameon=False)

leg = ax.legend(fontsize=15)
leg.get_frame().set_edgecolor('k')
leg.get_frame().set_linewidth(0.1)

tick_spacing = 250 # 250 for IL test

ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
#ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

fig.show()
fig.savefig(OUTPUT_DIR + 'mean_reward.png', dpi=400, bbox_inches="tight")



#%%

# Plot lane invasion ------------------------------------------

fig, ax = plt.subplots(1, 1)

for key, df in dict_df.items():
    ax.plot('episode', 'lane_change_ema', data=df, label=key)


# set labels (LaTeX can be used)
#plt.title(r'\textbf{Number of Incorrect Lane Changes for Carla Experiment}', fontsize=11)
ax.set_xlabel(r'\textbf{Episodes}', fontsize=18)
ax.set_ylabel(r'\textbf{Lane Invasions}', fontsize=18)
ax.grid(linestyle='--', linewidth=1)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

leg = ax.legend(fontsize=15)
leg.get_frame().set_edgecolor('k')
leg.get_frame().set_linewidth(0.1)

tick_spacing = 250 # 1000 for 3 configs # 250 for IL test

ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
#ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

fig.savefig(OUTPUT_DIR +'wrong_lane.png', dpi=400, bbox_inches="tight")
fig.show()


# %% 
# Plot total steps -------------------------------------------

fig, ax = plt.subplots(1, 1)

for key, df in dict_df.items():
    ax.plot('episode', 'total_steps_ema', data=df, label=key)

# set labels (LaTeX can be used)
#plt.title(r'\textbf{Collision Test for Carla Experiment}', fontsize=11)
ax.set_xlabel(r'\textbf{Episodes}', fontsize=18)
ax.set_ylabel(r'\textbf{Steps}', fontsize=18)
ax.grid(linestyle='--', linewidth=1)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

leg = ax.legend(fontsize=15)#, loc='lower right')
leg.get_frame().set_edgecolor('k')
leg.get_frame().set_linewidth(0.1)

tick_spacing = 250

ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
#ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

fig.savefig(OUTPUT_DIR +'steps.png', dpi=400, bbox_inches="tight")


# %%
