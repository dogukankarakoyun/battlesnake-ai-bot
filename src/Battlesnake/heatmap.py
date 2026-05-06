from collections import deque

from Battlesnake.game import Cell
from Battlesnake.utils import debug


def flood_fill_space(game_state, start):
    """
    Perform flood-fill from a start position and count reachable free cells.
    """
    debug(f"[flood_fill_space] Start: {start}")

    width = game_state["board"]["width"]
    height = game_state["board"]["height"]
    snakes = [seg for s in game_state["board"]["snakes"] for seg in s["body"]]
    blocked = {(s["x"], s["y"]) for s in snakes}

    for x in range(width):
        blocked.add((x, 0))
        blocked.add((x, height - 1))
    for y in range(height):
        blocked.add((0, y))
        blocked.add((width - 1, y))

    visited = set()
    count = 0
    queue = deque([(start.x, start.y)])

    while queue:
        x, y = queue.popleft()

        if (x, y) in visited or (x, y) in blocked:
            continue

        visited.add((x, y))
        count += 1

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited:
                queue.append((nx, ny))

    return count


def simulate_future_space(game_state, head_pos, turns=2, visited=None, memo=None):
    """
    Recursively simulate future moves and estimate available space.
    """
    if visited is None:
        visited = set()
    if memo is None:
        memo = {}

    key = (head_pos.x, head_pos.y, turns)
    if key in memo:
        return memo[key]

    if turns == 0:
        result = flood_fill_space(game_state, head_pos)
        memo[key] = result
        return result

    width = game_state["board"]["width"]
    height = game_state["board"]["height"]

    visited.add((head_pos.x, head_pos.y))

    best = -1
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = head_pos.x + dx, head_pos.y + dy
        if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited:
            score = simulate_future_space(
                game_state,
                Cell(nx, ny),
                turns - 1,
                visited.copy(),
                memo,
            )
            best = max(best, score)

    memo[key] = best if best != -1 else 0
    return memo[key]


def is_food_contested(food_pos, game_state):
    """
    Check whether a food cell is directly adjacent to an enemy head.
    """
    for snake in game_state["board"]["snakes"]:
        head = snake["body"][0]
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if (head["x"] + dx, head["y"] + dy) == (food_pos.x, food_pos.y):
                return True
    return False


def apply_food_layer(heatmap, game_state, health):
    """
    Increase food attractiveness in the heatmap.
    """
    debug(f"[apply_food_layer] Health: {health}")

    multiplier = 50 if health < 60 else 20
    for food in game_state["board"]["food"]:
        x, y = food["x"], food["y"]
        if is_food_contested(Cell(x, y), game_state):
            heatmap[x][y] += 5
        else:
            heatmap[x][y] += multiplier


def apply_snake_penalty_layer(heatmap, game_state):
    """
    Penalize cells occupied by snakes and cells adjacent to enemy heads.
    """
    width = game_state["board"]["width"]
    height = game_state["board"]["height"]

    for snake in game_state["board"]["snakes"]:
        for segment in snake["body"]:
            heatmap[segment["x"]][segment["y"]] -= 100

        head = snake["body"][0]
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = head["x"] + dx, head["y"] + dy
            if 0 <= nx < width and 0 <= ny < height:
                heatmap[nx][ny] -= 50


def apply_wall_penalty(heatmap, game_state):
    """
    Penalize wall cells and cells close to walls.
    """
    width = game_state["board"]["width"]
    height = game_state["board"]["height"]

    for x in range(width):
        heatmap[x][0] -= 20
        heatmap[x][height - 1] -= 20
    for y in range(height):
        heatmap[0][y] -= 20
        heatmap[width - 1][y] -= 20

    for x in range(width):
        heatmap[x][1] -= 5
        heatmap[x][height - 2] -= 5
    for y in range(height):
        heatmap[1][y] -= 5
        heatmap[width - 2][y] -= 5


def apply_flood_fill_layer(heatmap, game_state, my_head):
    """
    Use flood-fill to estimate available space in each direction.
    """
    width = game_state["board"]["width"]
    height = game_state["board"]["height"]

    for dx, dy, _ in [(-1, 0, "left"), (1, 0, "right"), (0, -1, "down"), (0, 1, "up")]:
        nx, ny = my_head.x + dx, my_head.y + dy
        if 0 <= nx < width and 0 <= ny < height:
            count = flood_fill_space(game_state, Cell(nx, ny))
            heatmap[nx][ny] += min(count, 50)


def apply_tail_priority(heatmap, game_state):
    """
    Add a bonus to the snake's own tail position.
    """
    tail = game_state["you"]["body"][-1]
    if 0 <= tail["x"] < len(heatmap) and 0 <= tail["y"] < len(heatmap[0]):
        heatmap[tail["x"]][tail["y"]] += 20


def apply_center_bonus(heatmap, game_state):
    """
    Slightly reward cells closer to the center of the board.
    """
    width = game_state["board"]["width"]
    height = game_state["board"]["height"]
    cx, cy = width // 2, height // 2

    for x in range(width):
        for y in range(height):
            dist = abs(x - cx) + abs(y - cy)
            heatmap[x][y] += max(0, 10 - dist)


def build_heatmap(game_state, my_head, health):
    """
    Build the complete heatmap from all scoring layers.
    """
    debug("[build_heatmap] Building heatmap...")

    width = game_state["board"]["width"]
    height = game_state["board"]["height"]
    heatmap = [[0 for _ in range(height)] for _ in range(width)]

    apply_food_layer(heatmap, game_state, health)
    apply_snake_penalty_layer(heatmap, game_state)
    apply_wall_penalty(heatmap, game_state)
    apply_flood_fill_layer(heatmap, game_state, my_head)
    apply_tail_priority(heatmap, game_state)
    apply_center_bonus(heatmap, game_state)

    return heatmap