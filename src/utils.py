import pandas as pd

def team_vs_team(matches, team1, team2):

    df = matches[
        ((matches['team1'] == team1) & (matches['team2'] == team2)) |
        ((matches['team1'] == team2) & (matches['team2'] == team1))
    ]

    return df.shape[0], \
           df[df['match_winner'] == team1].shape[0], \
           df[df['match_winner'] == team2].shape[0]


def team_stats(matches):

    teams = list(set(matches['team1']).union(set(matches['team2'])))

    stats = []

    for team in teams:
        total = matches[(matches['team1']==team)|(matches['team2']==team)].shape[0]
        wins = matches[matches['match_winner']==team].shape[0]

        stats.append({
            'Team': team,
            'Matches': total,
            'Wins': wins,
            'Win %': round((wins/total)*100,2) if total>0 else 0
        })

    return pd.DataFrame(stats).sort_values(by='Win %', ascending=False)
    # ==============================
    # 🔹 SCORE FEATURES
    # ==============================


def prepare_score_features(runs, overs, wickets):
    current_rr = runs / overs if overs > 0 else 0

    return {
        "runs": runs,
        "wickets": wickets,
        "overs": overs,
        "current_rr": current_rr
    }

    # ==============================
    # 🔹 CHASE FEATURES
    # ==============================


def prepare_chase_features(score, target, overs, wickets):
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets_left = 10 - wickets

    current_rr = score / overs if overs > 0 else 0
    required_rr = (runs_left * 6) / balls_left if balls_left > 0 else 0

    return {
        "runs_left": runs_left,
        "balls_left": balls_left,
        "wickets_left": wickets_left,
        "current_rr": current_rr,
        "required_rr": required_rr
    }