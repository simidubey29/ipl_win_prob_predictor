import pandas as pd

def load_data():
    matches = pd.read_csv('data/matches.csv')
    deliveries = pd.read_csv('data/deliveriess.csv')

    deliveries.rename(columns={
        'match_no': 'match_id',
        'innings': 'inning'
    }, inplace=True)

    return matches, deliveries


def merge_data(matches, deliveries):

    data = deliveries.merge(matches, on='match_id', how='inner')

    # 🔥 FIX VENUE COLUMN
    if 'venue_x' in data.columns:
        data['venue'] = data['venue_x']
    elif 'venue_y' in data.columns:
        data['venue'] = data['venue_y']

    return data