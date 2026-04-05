import pandas as pd
import pickle
import os
from xgboost import XGBRegressor

from src.data_preprocessing import load_data, merge_data

matches, deliveries = load_data()
data = merge_data(matches, deliveries)

# FIX venue column
if 'venue_x' in data.columns:
    data['venue'] = data['venue_x']
elif 'venue_y' in data.columns:
    data['venue'] = data['venue_y']

# FIX TYPES
data['runs_of_bat'] = pd.to_numeric(data['runs_of_bat'], errors='coerce')
data['extras'] = pd.to_numeric(data['extras'], errors='coerce')
data['over'] = pd.to_numeric(data['over'], errors='coerce')
data['inning'] = pd.to_numeric(data['inning'], errors='coerce')

# RUNS
data['total_runs'] = data['runs_of_bat'] + data['extras']

# FILTER 1ST INNINGS
data = data[data['inning'] == 1]

# CURRENT SCORE
data['current_score'] = data.groupby('match_id')['total_runs'].cumsum()

# BALLS
data['balls'] = data['over'] * 6

# FINAL SCORE
final_score = data.groupby('match_id')['total_runs'].sum().reset_index()
final_score.columns = ['match_id','final_score']

data = data.merge(final_score, on='match_id')

df = data[['batting_team','venue','current_score','balls','final_score']]

print("First innings dataset:", df.shape)

df = pd.get_dummies(df).dropna()

X = df.drop('final_score', axis=1)
y = df['final_score']

model = XGBRegressor(n_estimators=200)

model.fit(X, y)

os.makedirs('model', exist_ok=True)

pickle.dump(model, open('model/first_innings_model.pkl','wb'))
pickle.dump(X.columns, open('model/first_innings_columns.pkl','wb'))

print("✅ First Innings Model Ready")