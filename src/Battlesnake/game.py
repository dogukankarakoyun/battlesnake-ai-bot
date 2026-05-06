import typing


class Cell:
    """
    Represent a single cell on the Battlesnake board.
    """
    __slots__ = ("x", "y", "prime_encode")

    def __init__(self, x: int, y: int):
        """
        Create a new cell with x and y coordinates.
        """
        self.x = x
        self.y = y
        self.prime_encode = 2 ** self.x * 3 ** self.y

    def __str__(self):
        return f"({self.x}, {self.y})"

    def __repr__(self):
        return f"Cell(x={self.x}, y={self.y})"

    def __eq__(self, other):
        if not isinstance(other, Cell):
            return False
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash(self.prime_encode)

    def distance(self, other: typing.Self) -> int:
        """
        Return the Manhattan distance to another cell.
        """
        return abs(self.x - other.x) + abs(self.y - other.y)

    @staticmethod
    def from_json(json: typing.Dict):
        """
        Create a cell from a JSON object like {'x': 3, 'y': 5}.
        """
        return Cell(int(json["x"]), int(json["y"]))


class Snake:
    """
    Represent a snake in the game.
    """
    __slots__ = ("game_id", "body", "length")

    def __init__(self, game_id: str, body: list[Cell]):
        """
        Create a snake with an id and body cells.
        """
        self.game_id = game_id
        self.body = body
        self.length = len(self.body)

    def __str__(self):
        return f"{self.game_id}: {self.body})"

    def __repr__(self):
        return f"Snake(game_id={self.game_id}, body={self.body})"

    def __eq__(self, other):
        if not isinstance(other, Snake):
            return False
        return self.game_id == other.game_id and self.body == other.body

    def __hash__(self):
        return hash((self.game_id, self.body))

    def update_from_json(self, json: typing.Dict):
        """
        Update the snake based on a new game state.
        """
        self.body.insert(0, Cell.from_json(json["head"]))

        if int(json["length"]) == self.length:
            self.body.pop(-1)
        else:
            self.length = len(self.body)

    @staticmethod
    def from_json(json: typing.Dict):
        """
        Create a snake from JSON data.
        """
        game_id: str = json["id"]
        body: list[Cell] = [Cell.from_json(cell_obj) for cell_obj in json["body"]]
        return Snake(game_id, body)


class Game:
    """
    Hold the full game state including board size, snakes, food, and hazards.
    """
    __slots__ = ("turn", "width", "height", "snakes", "you", "ownfood", "hazards")

    def __init__(self, game_state: typing.Dict):
        """
        Build a game object from the current Battlesnake game state.
        """
        self.turn: int = int(game_state["turn"])
        self.width: int = int(game_state["board"]["width"])
        self.height: int = int(game_state["board"]["height"])
        self.snakes: list[Snake] = [
            Snake.from_json(snake_obj) for snake_obj in game_state["board"]["snakes"]
        ]
        self.you: Snake = next(
            x for x in self.snakes if x.game_id == game_state["you"]["id"]
        )
        self.ownfood: list[Cell] = [
            Cell.from_json(food_obj) for food_obj in game_state["board"]["food"]
        ]
        self.hazards: list[Cell] = [
            Cell.from_json(hazard_obj) for hazard_obj in game_state["board"]["hazards"]
        ]

    def __str__(self):
        return f"Turn {self.turn}: {self.snakes}"

    def update(self, game_state: typing.Dict) -> None:
        """
        Update the game object from a new game state.
        """
        self.turn = int(game_state["turn"])
        if self.turn == 0:
            return

        self.ownfood = [
            Cell.from_json(food_obj) for food_obj in game_state["board"]["food"]
        ]

        sn_ids: list[str] = [sn["id"] for sn in game_state["board"]["snakes"]]
        to_delete: list[Snake] = []

        for snake in self.snakes:
            if snake.game_id in sn_ids:
                snake.update_from_json(
                    game_state["board"]["snakes"][sn_ids.index(snake.game_id)]
                )
            else:
                to_delete.append(snake)

        for snake in to_delete:
            self.snakes.remove(snake)