from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from joblib import dump
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

base_dir = Path(__file__).resolve().parent.parent
csv_path = base_dir / "simple_synthetic_data.csv"
model_path = base_dir / "insane_model.pkl"

expected_columns = [
    "head_x",
    "head_y",
    "health",
    "width",
    "height",
    "closest_food_distance",
    "space",
    "future_space",
    "safe_up",
    "safe_down",
    "safe_left",
    "safe_right",
    "open_area_up",
    "open_area_down",
    "open_area_left",
    "open_area_right",
    "distance_to_nearest_wall",
    "tail_distance",
    "closest_food_is_safe",
    "is_biggest_snake",
    "needs_food",
    "closest_enemy_head_dist",
    "enemy_head_is_adjacent",
    "enemies_within_2",
    "kill_up",
    "kill_down",
    "kill_left",
    "kill_right",
    "dir_up",
    "dir_down",
    "dir_left",
    "dir_right",
    "center_bonus",
    "food_contest_count",
    "num_snakes",
    "path_distance_to_food",
    "move",
]

if not csv_path.exists():
    print("[INFO] Training dataset not found. Creating an empty file.")
    df = pd.DataFrame(columns=pd.Index(expected_columns))
    df.to_csv(csv_path, index=False)
    print("[INFO] Empty dataset created. Collect training data first.")
    exit()

print("[DEBUG] Loading training data...")
try:
    data = pd.read_csv(csv_path)
except Exception as e:
    print(f"[ERROR] Failed to load CSV file: {e}")
    exit()

if "move" not in data.columns:
    print("[INFO] No headers found. Applying expected column names.")
    data = pd.read_csv(csv_path, names=expected_columns)

data.drop_duplicates(inplace=True)

if data.empty:
    print("[WARNING] Dataset is empty. Generate data before training.")
    exit()

print(f"[DEBUG] Rows after duplicate removal: {len(data)}")
print(f"[DEBUG] Columns: {list(data.columns)}")

if "move" not in data.columns:
    print("[ERROR] Missing target column 'move'.")
    exit()

feature_columns = [
    "head_x", "head_y", "health", "width", "height", "closest_food_distance",
    "space", "future_space", "safe_up", "safe_down", "safe_left", "safe_right",
    "open_area_up", "open_area_down", "open_area_left", "open_area_right",
    "distance_to_nearest_wall", "tail_distance", "closest_food_is_safe",
    "is_biggest_snake", "needs_food", "closest_enemy_head_dist",
    "enemy_head_is_adjacent", "enemies_within_2", "kill_up", "kill_down",
    "kill_left", "kill_right", "dir_up", "dir_down", "dir_left", "dir_right",
    "center_bonus", "food_contest_count", "num_snakes", "path_distance_to_food",
]

X = data[feature_columns]
y = data["move"]

print("\nMove distribution:")
print(y.value_counts())

print("\n[DEBUG] Applying random undersampling...")
rus = RandomUnderSampler(random_state=42)
try:
    resampled_data = rus.fit_resample(X, y)
    X, y = resampled_data[0], resampled_data[1]
except ValueError as ve:
    print(f"[ERROR] Undersampling failed: {ve}")
    exit()

print(f"[DEBUG] Samples after resampling: {len(X)}")

print("[DEBUG] Scaling features with StandardScaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("[DEBUG] Creating train/test split...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.1,
    random_state=66,
)

print(f"\n[DEBUG] Train: {len(X_train)} | Test: {len(X_test)}")

print("\n[INFO] Training model...")
clf = LGBMClassifier(
    n_estimators=400,
    max_depth=25,
    class_weight="balanced",
    random_state=42,
)
clf.fit(X_train, y_train)
print("[INFO] Training completed.\n")

print("[INFO] Classification report:")
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

print("[DEBUG] Generating feature importance plot...")
importances = clf.feature_importances_
feature_names = X.columns.tolist()

plt.figure(figsize=(10, 8))
plt.barh(feature_names, importances)
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.tight_layout()

plot_path = base_dir / "feature_importance.png"
plt.savefig(plot_path)
print(f"[INFO] Feature importance saved as '{plot_path.name}'.")

dump(clf, model_path)
print(f"[INFO] Model saved as: '{model_path}'")