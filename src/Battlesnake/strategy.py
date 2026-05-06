import sys
import typing
from pathlib import Path

import pandas as pd
from joblib import load

from Battlesnake.game import Cell, Game
from Battlesnake.heatmap import build_heatmap
from Battlesnake.path_fallback import PathSolver
from Battlesnake.utils import debug
from LightGBM.ml_features import ml_features

# Try multiple paths so the project also works in Replit-like environments.
possible_paths = [
    Path(__file__).resolve().parent.parent / "LightGBM",
    Path.cwd() / "LightGBM",
    Path.cwd() / "Battlesnake" / "LightGBM",
    Path(__file__).resolve().parent / "LightGBM",
]

ml_path = None
for path in possible_paths:
    if path.exists():
        ml_path = path
        break

if ml_path is None:
    debug(f"[ERROR] LightGBM directory not found. Tried paths: {[str(p) for p in possible_paths]}")
    ml_path = Path.cwd() / "LightGBM"

if str(ml_path) not in sys.path:
    sys.path.insert(0, str(ml_path))

model_path = ml_path / "insane_model.pkl"

columns = [
    "head_x", "head_y", "health", "width", "height", "closest_food_distance",
    "space", "future_space", "safe_up", "safe_down", "safe_left", "safe_right",
    "open_area_up", "open_area_down", "open_area_left", "open_area_right",
    "distance_to_nearest_wall", "tail_distance", "closest_food_is_safe", "is_biggest_snake",
    "needs_food", "closest_enemy_head_dist", "enemy_head_is_adjacent", "enemies_within_2",
    "kill_up", "kill_down", "kill_left", "kill_right",
    "dir_up", "dir_down", "dir_left", "dir_right",
    "center_bonus", "food_contest_count", "num_snakes", "path_distance_to_food",
]

try:
    if model_path.exists():
        ml_model = load(str(model_path))
        debug(f"[ML] Loaded insane_model.pkl from: {model_path}")
    else:
        debug(f"[ML] Model file not found at: {model_path}")
        ml_model = None
except Exception as e:
    ml_model = None
    debug(f"[ML] Failed to load insane_model.pkl: {e}")
    debug(f"[ML] Model path was: {model_path}")
    debug(f"[ML] Current working directory: {Path.cwd()}")


def choose_strategy(game_state):
    """
    Select a strategy based on the current game state.
    """
    turn = game_state["turn"]
    health = game_state["you"]["health"]

    if turn < 30:
        debug("[Strategy] Early game")
        return "early"
    if health < 30:
        debug("[Strategy] Low health")
        return "emergency"

    debug("[Strategy] Late game")
    return "late"


def classify_move_quality(proba_dict, actual_move, flood_fill_space_after=None):
    """
    Classify move quality based on model confidence.
    """
    best_move = max(proba_dict, key=proba_dict.get) if proba_dict else None
    confidence_actual = proba_dict.get(actual_move, 0) if proba_dict else 0

    if actual_move == best_move and confidence_actual > 0.8:
        return "Excellent"
    if confidence_actual >= 0.6:
        return "Good"
    if confidence_actual >= 0.4:
        return "Uncertain"
    if flood_fill_space_after is not None and flood_fill_space_after < 3:
        return "Poor"
    return "Weak"


def move(game_state: typing.Dict) -> typing.Dict:
    """
    Main decision function for the next move.

    Order of decision-making:
    1. Heatmap-based evaluation
    2. A* pathfinding fallback
    3. ML-based fallback
    4. Final safety validation
    """
    debug("[move] Selecting next move...")

    my_head_dict = game_state["you"]["body"][0]
    my_head = Cell(my_head_dict["x"], my_head_dict["y"])
    health = game_state["you"]["health"]
    width = game_state["board"]["width"]
    height = game_state["board"]["height"]

    moves = {
        "up": (0, 1),
        "down": (0, -1),
        "left": (-1, 0),
        "right": (1, 0),
    }

    features = ml_features(game_state)
    if features is None:
        debug("[move] Feature extraction failed, using fallback move 'up'")
        return {"move": "up"}

    debug(f"[move] Extracted features: {features}")

    best_move = None
    proba_dict = {}

    # 1. Heatmap-based primary strategy
    debug("[Heatmap] Evaluating heatmap-based move...")
    heatmap = build_heatmap(game_state, my_head, health)
    move_scores = {}

    for direction, (dx, dy) in moves.items():
        nx, ny = my_head.x + dx, my_head.y + dy
        if 0 <= nx < width and 0 <= ny < height:
            move_scores[direction] = heatmap[nx][ny]

    if move_scores:
        best_move = max(move_scores, key=lambda k: move_scores[k])
        debug(f"[Heatmap] Selected move: {best_move}")
    else:
        debug("[Heatmap] No safe heatmap-based move found")

    # 2. A* fallback
    if best_move is None:
        debug("[A*] Activating A* fallback...")
        try:
            game_obj = Game(game_state)
            path_solver = PathSolver(game_obj)
            food_goals = {Cell(f["x"], f["y"]) for f in game_state["board"]["food"]}
            valid_goals = [goal for goal in food_goals if goal not in path_solver.forbidden_cells]

            if not valid_goals:
                valid_goals = list(path_solver.neighbors(my_head))

            astar_result = path_solver.astar(my_head, valid_goals[0]) if valid_goals else None
            path = list(astar_result) if astar_result else None

            if path:
                next_cell = path[1] if len(path) > 1 else path[0]
                for direction, (dx, dy) in moves.items():
                    if next_cell.x == my_head.x + dx and next_cell.y == my_head.y + dy:
                        best_move = direction
                        debug(f"[A*] Selected move: {best_move}")
                        break
            else:
                debug("[A*] No valid path found")
        except Exception as ex:
            debug(f"[A*] Error: {ex}")

    # 3. ML fallback
    if best_move is None:
        debug("[ML] Activating ML fallback...")
        try:
            if ml_model:
                features_df = pd.DataFrame([features], columns=pd.Index(columns))
                proba = ml_model.predict_proba(features_df)
                confidence = max(proba[0])
                ml_prediction = ml_model.predict(features_df)[0]
                proba_dict = dict(zip(ml_model.classes_, proba[0]))

                debug(f"[ML] Prediction: {ml_prediction}, probabilities: {proba_dict}")

                if ml_prediction in moves and confidence > 0.6:
                    best_move = ml_prediction
                    debug(f"[ML] Selected move: {best_move}")
                else:
                    debug("[ML] Confidence too low")
            else:
                debug("[ML] No model loaded")
        except Exception as e:
            debug(f"[ML] Prediction failed: {e}")

    if best_move is None:
        debug("[Fallback] No strategy succeeded, using 'up'")
        best_move = "up"

    def is_safe_move(move_name, current_game_state):
        move_vectors = {
            "up": (0, 1),
            "down": (0, -1),
            "left": (-1, 0),
            "right": (1, 0),
        }

        my_head_local = current_game_state["you"]["body"][0]
        board_width = current_game_state["board"]["width"]
        board_height = current_game_state["board"]["height"]
        snakes = current_game_state["board"]["snakes"]
        all_snake_bodies = [coord for snake in snakes for coord in snake["body"]]

        dx, dy = move_vectors[move_name]
        nx, ny = my_head_local["x"] + dx, my_head_local["y"] + dy

        if not (0 <= nx < board_width and 0 <= ny < board_height):
            return False
        if {"x": nx, "y": ny} in all_snake_bodies:
            return False
        return True

    if not is_safe_move(best_move, game_state):
        debug(f"[SAFETY] Move '{best_move}' is unsafe, looking for fallback")
        for move_name in ["up", "down", "left", "right"]:
            if is_safe_move(move_name, game_state):
                debug(f"[SAFETY] Safe fallback move: {move_name}")
                return {"move": move_name}

        debug("[SAFETY] No safe move found, returning 'up'")
        return {"move": "up"}

    move_quality = classify_move_quality(proba_dict, best_move)
    debug(f"[Move Quality] {move_quality}")

    return {"move": best_move}