
#%%
import pandas as pd

ROOT = "ECAI/data/MoCo/1/"

list_csv = {
    "route_statistics_0_2023-05-07-09-58.csv",
    "route_statistics_1_2023-05-07-09-58.csv",
    "route_statistics_2_2023-05-07-09-59.csv",
    "route_statistics_3_2023-05-07-09-59.csv"
}

list_csv = [ROOT + c for c in list_csv]

list_df = [pd.read_csv(df) for df in list_csv]

#%% Add agent id

for idx, df in enumerate(list_df):
    df['agent_id'] = idx


#%%

merged_df = pd.concat(list_df, ignore_index=True)
merged_df = merged_df.sort_values(by=['Episode Number', 'agent_id'], ascending=[True,True])
merged_df = merged_df.reset_index()
merged_df

# %% Check if episode number always increases or stays the same

for idx, row in merged_df[0:-1].iterrows():
    if row['Episode Number'] > merged_df.iloc[idx+1]['Episode Number']:
        print('error in idx: ', idx)


# %% Save merged file as csv

merged_df.to_csv(ROOT+'merged.csv')

# %%
