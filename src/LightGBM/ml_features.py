from Battlesnake.path_fallback import next_step
from Battlesnake.game import Cell
from Battlesnake.heatmap import flood_fill_space, simulate_future_space, is_food_contested


def closest_food_is_safe(my_head, food, game_state):
    """
    Determine whether the closest food is uncontested by enemy snakes.
    """
    try:
        if not food:
            return 0

        food_sorted = sorted(
            food,
            key=lambda f: abs(f["x"] - my_head.x) + abs(f["y"] - my_head.y),
        )

        closest_food = food_sorted[0]
        closest_food_cell = Cell(closest_food["x"], closest_food["y"])

        if is_food_contested(closest_food_cell, game_state):
            return 0

        return 1
    except Exception as e:
        print(f"[ERROR] closest_food_is_safe failed: {e}")
        return 0


def starvation_risk(health, closest_food_dist, food_safe):
    """
    Estimate whether starvation risk is high.
    """
    if health < 30:
        if closest_food_dist > 5 or food_safe == 0:
            return 1
    return 0


def ml_features(game_state):
    """
    Extract ML features from the current game state.
    """
    try:
        if not isinstance(game_state, dict):
            print("[ERROR] game_state is not a dictionary")
            return None

        my_head_dict = game_state["you"]["body"][0]
        my_head = Cell(my_head_dict["x"], my_head_dict["y"])

        health = game_state["you"].get("health", 100)
        width = game_state["board"].get("width", 11)
        height = game_state["board"].get("height", 11)
        food = game_state["board"].get("food", [])
        snakes = game_state["board"].get("snakes", [])
        my_id = game_state["you"]["id"]
        my_length = len(game_state["you"].get("body", []))

        all_snake_bodies = [
            coord for snake in snakes if snake["id"] != my_id
            for coord in snake["body"]
        ]

        if not food:
            closest_food_dist = width + height
        else:
            closest_food_dist = min(
                [abs(f["x"] - my_head.x) + abs(f["y"] - my_head.y) for f in food],
                default=width + height,
            )

        try:
            from Battlesnake.game import Game

            game_obj = Game(game_state)
            next_cell = next_step(game_obj)
            path_distance_to_food = (
                abs(my_head.x - next_cell.x) + abs(my_head.y - next_cell.y)
                if next_cell else 0
            )
        except Exception as e:
            print(f"[ERROR] next_step failed: {e}")
            path_distance_to_food = closest_food_dist

        try:
            space = flood_fill_space(game_state, my_head) or 0
        except Exception as e:
            print(f"[ERROR] flood_fill_space failed: {e}")
            space = 0

        try:
            future_space = simulate_future_space(game_state, my_head) or 0
        except Exception as e:
            print(f"[ERROR] simulate_future_space failed: {e}")
            future_space = 0

        cx, cy = width // 2, height // 2
        dist_to_center = abs(my_head.x - cx) + abs(my_head.y - cy)
        center_bonus = max(0, 10 - dist_to_center)

        food_safe = closest_food_is_safe(my_head, food, game_state)
        starvation_risk(health, closest_food_dist, food_safe)

        def is_safe(move):
            move_dx, move_dy = move
            x, y = my_head.x + move_dx, my_head.y + move_dy

            if not (0 <= x < width and 0 <= y < height):
                return 0

            if {"x": x, "y": y} in all_snake_bodies:
                return 0

            my_body = game_state["you"]["body"]
            if len(my_body) > 1:
                for body_part in my_body[:-1]:
                    if body_part["x"] == x and body_part["y"] == y:
                        return 0

            return 1

        safe_up = is_safe((0, 1))
        safe_down = is_safe((0, -1))
        safe_left = is_safe((-1, 0))
        safe_right = is_safe((1, 0))

        def open_area(move):
            dx, dy = move
            x, y = my_head.x + dx, my_head.y + dy
            if not (0 <= x < width and 0 <= y < height):
                return 0
            try:
                return flood_fill_space(game_state, Cell(x, y)) or 0
            except Exception as e:
                print(f"[ERROR] open_area flood_fill failed at ({x}, {y}): {e}")
                return 0

        open_area_up = open_area((0, 1))
        open_area_down = open_area((0, -1))
        open_area_left = open_area((-1, 0))
        open_area_right = open_area((1, 0))

        distance_to_nearest_wall = min(
            my_head.x,
            width - 1 - my_head.x,
            my_head.y,
            height - 1 - my_head.y,
        )

        tail = game_state["you"]["body"][-1]
        tail_distance = abs(tail["x"] - my_head.x) + abs(tail["y"] - my_head.y)

        def am_i_biggest():
            return int(
                all(len(s["body"]) < my_length or s["id"] == my_id for s in snakes)
            )

        is_biggest = am_i_biggest()
        needs_food = int(health < 30 and not is_biggest)

        enemy_heads = [snake["body"][0] for snake in snakes if snake["id"] != my_id]
        closest_enemy_head_dist = min(
            [abs(h["x"] - my_head.x) + abs(h["y"] - my_head.y) for h in enemy_heads],
            default=width + height,
        )
        enemy_head_is_adjacent = int(
            any(abs(h["x"] - my_head.x) + abs(h["y"] - my_head.y) == 1 for h in enemy_heads)
        )
        enemies_within_2 = sum(
            abs(h["x"] - my_head.x) + abs(h["y"] - my_head.y) <= 2
            for h in enemy_heads
        )

        try:
            food_contest_count = sum(
                is_food_contested(Cell(f["x"], f["y"]), game_state)
                for f in food
            )
        except Exception as e:
            print(f"[ERROR] food_contest_count failed: {e}")
            food_contest_count = 0

        num_snakes = len(snakes)

        kill_up = kill_down = kill_left = kill_right = 0
        directions = {
            "up": (0, 1),
            "down": (0, -1),
            "left": (-1, 0),
            "right": (1, 0),
        }

        for dir_name, (dx, dy) in directions.items():
            nx, ny = my_head.x + dx, my_head.y + dy
            for s in snakes:
                if s["id"] == my_id:
                    continue
                if s["body"][0]["x"] == nx and s["body"][0]["y"] == ny and len(s["body"]) < my_length:
                    if dir_name == "up":
                        kill_up = 1
                    elif dir_name == "down":
                        kill_down = 1
                    elif dir_name == "left":
                        kill_left = 1
                    elif dir_name == "right":
                        kill_right = 1

        if len(game_state["you"]["body"]) >= 2:
            neck = game_state["you"]["body"][1]
            dx = my_head.x - neck["x"]
            dy = my_head.y - neck["y"]
            if dx == 0 and dy == 1:
                current_dir = "up"
            elif dx == 0 and dy == -1:
                current_dir = "down"
            elif dx == 1 and dy == 0:
                current_dir = "right"
            elif dx == -1 and dy == 0:
                current_dir = "left"
            else:
                current_dir = "up"
        else:
            current_dir = "up"

        dir_up = int(current_dir == "up")
        dir_down = int(current_dir == "down")
        dir_left = int(current_dir == "left")
        dir_right = int(current_dir == "right")

        features = [
            my_head.x, my_head.y, health, width, height, closest_food_dist,
            space, future_space, safe_up, safe_down, safe_left, safe_right,
            open_area_up, open_area_down, open_area_left, open_area_right,
            distance_to_nearest_wall, tail_distance, food_safe, is_biggest,
            needs_food, closest_enemy_head_dist, enemy_head_is_adjacent,
            enemies_within_2, kill_up, kill_down, kill_left, kill_right,
            dir_up, dir_down, dir_left, dir_right, center_bonus,
            food_contest_count, num_snakes, path_distance_to_food,
        ]

        if any(f is None for f in features) or len(features) != 36:
            print("[WARNING] Feature vector contains invalid values.")
            return None

        return features

    except Exception as fatal_error:
        print(f"[FATAL] Error during feature computation: {fatal_error}")
        return None