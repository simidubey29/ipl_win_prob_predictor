import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/deliveriess.csv")

print("📊 Columns:", df.columns)

# =========================
# CLEAN DATA
# =========================
df = df.dropna()

# =========================
# FEATURE ENGINEERING
# =========================

# Total runs per ball
df['total_runs'] = df['runs_of_bat'] + df['extras']

# Cumulative score
df['score'] = df.groupby(['match_no', 'innings'])['total_runs'].cumsum()

# Ball number (approx using over)
df['ball_number'] = df.groupby(['match_no', 'innings']).cumcount() + 1

# Balls left
df['balls_left'] = 120 - df['ball_number']

# Wickets
df['is_wicket'] = df['wicket_type'].apply(lambda x: 0 if x == 'None' else 1)
df['wickets'] = df.groupby(['match_no', 'innings'])['is_wicket'].cumsum()

df['wickets_left'] = 10 - df['wickets']

# Run rates
df['crr'] = df['score'] / (df['ball_number'] / 6)
df['rrr'] = (150 - df['score']) * 6 / df['balls_left'].replace(0, 1)

# =========================
# TARGET (TEMPORARY LOGIC)
# =========================
df['result'] = df['runs_of_bat'].apply(lambda x: 1 if x > 0 else 0)

# =========================
# FEATURES
# =========================
features = ['score', 'balls_left', 'wickets_left', 'crr', 'rrr']

df = df.dropna()

X = df[features]
y = df['result']

print("📊 Final Shape:", X.shape)

# =========================
# TRAIN MODEL
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# =========================
# SAVE MODEL
# =========================
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("🎉 SUCCESS: model.pkl created!")