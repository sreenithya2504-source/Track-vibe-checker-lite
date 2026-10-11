
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Load only the columns needed for training
DATA_FILE = "spotify-tracks-dataset-detailed.csv"

FEATURES = [
    "danceability",
    "energy",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "loudness",
    "duration_ms",
]

df = pd.read_csv(
    DATA_FILE,
    usecols=FEATURES + ["popularity"]
)

# Clean invalid or missing values
df = df.dropna()
df = df[df["popularity"].between(0, 100)]

X = df[FEATURES].astype("float32")
y = df["popularity"].astype("float32")

# Keep training manageable
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=40,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Mean Absolute Error:", round(mean_absolute_error(y_test, predictions), 2))
print("R² score:", round(r2_score(y_test, predictions), 3))

# Save the smaller model
joblib.dump(model, "popularity_model_lite.pkl", compress=3)

print("Model saved as popularity_model_lite.pkl")
print("Training rows:", len(X_train))
print("Model features:", FEATURES)
