from Battlesnake.utils import debug
from Battlesnake.strategy import move
from Battlesnake.server import run_server

import typing
import os
import sys

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "LightGBM"))
if base_dir not in sys.path:
    sys.path.append(base_dir)


def info() -> typing.Dict:
    """
    Return Battlesnake metadata.
    """
    debug("[info] Sending Battlesnake metadata.")
    return {
        "apiversion": "1",
        "author": "battlesnake-ai-bot",
        "color": "#7B1E1E",
        "head": "mlh-gene",
        "tail": "mystic",
    }


def start(game_state: typing.Dict):
    """
    Called when a new game starts.
    """
    debug("[start] Game started.")


def end(game_state: typing.Dict):
    """
    Called when the game ends.
    """
    debug("[end] Game ended.")


if __name__ == "__main__":
    run_server({
        "info": info,
        "start": start,
        "move": move,
        "end": end
    })