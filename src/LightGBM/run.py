from pathlib import Path
import traceback

import pandas as pd
from flask import Flask, request, jsonify
from joblib import load

from LightGBM.ml_features import ml_features

app = Flask(__name__)

base_dir = Path(__file__).resolve().parent
model_path = base_dir / "insane_model.pkl"


def load_model(path):
    """
    Load a trained ML model from disk.
    Return None if loading fails.
    """
    if not path.exists():
        print(f"[WARNING] Model file not found: {path}")
        return None
    try:
        model = load(path)
        print(f"[ML] Model loaded successfully from: {path}")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return None


model = load_model(model_path)


@app.route("/start", methods=["POST"])
def start():
    """
    Called when a new game starts.
    """
    print("[INFO] Game started.")
    return "ok"


@app.route("/end", methods=["POST"])
def end():
    """
    Called when a game ends.
    """
    print("[INFO] Game ended.")
    return "ok"


@app.route("/move", methods=["POST"])
def move():
    """
    Main move decision endpoint.

    Converts the current game state into features, runs the ML model,
    and returns the predicted move. If anything fails, a fallback move is used.
    """
    try:
        try:
            game_state = request.get_json()
            if not game_state:
                print("[ERROR] Invalid request JSON received.")
                return jsonify({"move": "up"})
        except Exception as e:
            print(f"[ERROR] Failed to parse request JSON: {e}")
            return jsonify({"move": "up"})

        features = ml_features(game_state)

        if not features or len(features) != 36:
            print("[ERROR] Invalid feature vector received. Skipping prediction.")
            return jsonify({"move": "up"})

        print(f"[DEBUG] Input features: {features}")

        if not model:
            print("[ERROR] No model available.")
            return jsonify({"move": "up"})

        try:
            columns = [
                "head_x", "head_y", "health", "width", "height",
                "closest_food_distance", "space", "future_space", "safe_up",
                "safe_down", "safe_left", "safe_right", "open_area_up",
                "open_area_down", "open_area_left", "open_area_right",
                "distance_to_nearest_wall", "tail_distance",
                "closest_food_is_safe", "is_biggest_snake", "needs_food",
                "closest_enemy_head_dist", "enemy_head_is_adjacent",
                "enemies_within_2", "kill_up", "kill_down", "kill_left",
                "kill_right", "dir_up", "dir_down", "dir_left", "dir_right",
                "center_bonus", "food_contest_count", "num_snakes",
                "path_distance_to_food",
            ]

            features_df = pd.DataFrame([features], columns=columns)
            prediction = model.predict(features_df)[0]
            print(f"[ML] Predicted move: {prediction}")

            def is_safe_move(move_name, current_game_state):
                moves = {
                    "up": (0, 1),
                    "down": (0, -1),
                    "left": (-1, 0),
                    "right": (1, 0),
                }
                my_head = current_game_state["you"]["body"][0]
                width = current_game_state["board"]["width"]
                height = current_game_state["board"]["height"]
                snakes = current_game_state["board"]["snakes"]
                all_snake_bodies = [coord for snake in snakes for coord in snake["body"]]

                dx, dy = moves[move_name]
                nx, ny = my_head["x"] + dx, my_head["y"] + dy

                if not (0 <= nx < width and 0 <= ny < height):
                    return False
                if {"x": nx, "y": ny} in all_snake_bodies:
                    return False
                return True

            if not is_safe_move(prediction, game_state):
                print(f"[SAFETY] Predicted move '{prediction}' is unsafe. Choosing fallback.")
                for move_name in ["up", "down", "left", "right"]:
                    if is_safe_move(move_name, game_state):
                        print(f"[SAFETY] Safe fallback move: {move_name}")
                        return jsonify({"move": move_name})
                print("[SAFETY] No safe moves found. Returning 'up'.")
                return jsonify({"move": "up"})

            return jsonify({"move": prediction})

        except Exception as ml_error:
            print(f"[WARNING] ML prediction failed: {ml_error}")

    except Exception:
        print("[ERROR] Exception in move handler:")
        traceback.print_exc()

    fallback_move = "up"
    print(f"[Fallback] Using fallback move: {fallback_move}")
    return jsonify({"move": fallback_move})


if __name__ == "__main__":
    """
    Start the Battlesnake server locally on port 8000.
    """
    print("[INFO] Starting Battlesnake server...")
    app.run(host="0.0.0.0", port=8000, debug=True)