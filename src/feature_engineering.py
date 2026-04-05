import pandas as pd
import numpy as np

def create_features(data):

    # ---------------- FIX TYPES ----------------
    data['runs_of_bat'] = pd.to_numeric(data['runs_of_bat'], errors='coerce')
    data['extras'] = pd.to_numeric(data['extras'], errors='coerce')
    data['over'] = pd.to_numeric(data['over'], errors='coerce')
    data['inning'] = pd.to_numeric(data['inning'], errors='coerce')

    # ---------------- RUNS ----------------
    data['total_runs'] = data['runs_of_bat'] + data['extras']

    # ---------------- SCORE ----------------
    data['current_score'] = data.groupby(['match_id','inning'])['total_runs'].cumsum()

    # ---------------- BALLS ----------------
    data['balls'] = data['over'] * 6
    data['balls_left'] = 120 - data['balls']

    # ---------------- WICKETS ----------------
    data['is_wicket'] = data['player_dismissed'].apply(
        lambda x: 0 if pd.isna(x) else 1
    ).astype(int)

    data['wickets'] = data.groupby(['match_id','inning'])['is_wicket'].cumsum()
    data['wickets_left'] = 10 - data['wickets']

    # ---------------- RUN RATE ----------------
    data['crr'] = data['current_score'] / (data['balls']/6 + 0.1)

    # ---------------- TARGET ----------------
    data['target'] = data['first_ings_score'] + 1
    data['runs_left'] = data['target'] - data['current_score']
    data['rrr'] = (data['runs_left'] * 6) / (data['balls_left'] + 0.1)

    # ---------------- FILTER 2ND INNINGS ----------------
    data = data[data['inning'] == 2]

    # ---------------- RESULT ----------------
    data.loc[:, 'result'] = (data['batting_team'] == data['match_winner']).astype(int)

    # ---------------- FINAL ----------------
    final_df = data[
        [
            'batting_team','bowling_team','venue',
            'runs_left','balls_left','wickets_left',
            'target','crr','rrr','result'
        ]
    ]

    return final_df.dropna()