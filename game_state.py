from utils import is_within_bounds, is_position_free, get_l_positions, validate_move

class GameState:
    def __init__(self):
        """
        Initialize the game state.
        """
        # 4x4 board represented as a 2D list (0 = empty, 1 = Player 1's L, 2 = Player 2's L, "N" = neutral piece)
        self.board = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]

        # Initial positions of L pieces and neutral pieces
        self.neutral_positions = [(0, 0), (3, 3)]
        self.l_positions = {
            1: {"x": 1, "y": 3, "config": 2},  # Player 1's L-piece
            2: {"x": 2, "y": 0, "config": 1},  # Player 2's L-piece
        }

        # Place the pieces on the board
        self._initialize_board()

        # Player 1 starts the game
        self.current_player = 1

        # Track board state history to detect repeated states
        self.state_history = {}
        self.total_turns = 0  # Track total turns

        # Limits for repetitions and total turns
        self.max_repetitions = 3
        self.max_turns = 100

    def _initialize_board(self):
        """
        Place the initial L pieces and neutral pieces on the board.
        """
        self.board = [[0 for _ in range(4)] for _ in range(4)]

        # Place neutral pieces
        for x, y in self.neutral_positions:
            self.board[y][x] = "N"

        # Place L-pieces for both players dynamically from self.l_positions
        for player, l_data in self.l_positions.items():
            x, y, config = l_data["x"], l_data["y"], l_data["config"]
            for px, py in get_l_positions(x, y, config):
                self.board[py][px] = player

    def _hash_board(self):
        """
        Generate a hashable representation of the board for tracking repeated states.
        """
        return tuple(tuple(row) for row in self.board)

    def is_terminal(self, check_ties=False):
        """
        Check if the current game state is terminal (no legal moves for the current player or a tie).
        Args:
            check_ties (bool): Whether to check for repeated states and ties.
        """
        if check_ties:
            # Check for repeated states (tie condition)
            board_hash = self._hash_board()
            if board_hash in self.state_history:
                self.state_history[board_hash] += 1
                if self.state_history[board_hash] >= self.max_repetitions:
                    print("The same board configuration has occurred three times. Declaring a tie.")
                    return True
            else:
                self.state_history[board_hash] = 1

            # Check for maximum turns
            if self.total_turns >= self.max_turns:
                print(f"Game has reached the maximum turn limit of {self.max_turns}. Declaring a tie.")
                return True

        # Check if the current player has no legal moves
        return len(self.get_legal_moves(self.current_player)) == 0

    def get_legal_moves(self, player):
        """
        Get all legal moves for the given player, combining L-piece and neutral piece moves.
        Args:
            player (int): The player (1 or 2) whose legal moves are to be generated.
        Returns:
            list[dict]: A list of legal moves, each containing the L-piece move and optionally a neutral piece move.
        """
        legal_moves = set()

        # Iterate over all possible L-piece configurations
        for x in range(4):
            for y in range(4):
                for config in range(8):  # Use 0-7 for L-piece configurations
                    l_positions = get_l_positions(x, y, config)

                    # Check if the L-piece move is valid
                    if not all(
                        is_within_bounds(px, py) and is_position_free(self.board, px, py, player)
                        for px, py in l_positions
                    ):
                        continue

                    # Create the L-piece move
                    base_move = {"L_piece": {"x": x, "y": y, "config": config}, "neutral_move": None}

                    # Validate the L-piece move without neutral moves
                    if not validate_move(self.board, base_move, player):
                        continue

                    legal_moves.add((
                        frozenset(base_move["L_piece"].items()),
                        frozenset(base_move["neutral_move"].items()) if base_move["neutral_move"] else None,
                    ))

                    # Add possible neutral piece moves
                    for nx, ny in self.neutral_positions:
                        for tx in range(4):
                            for ty in range(4):
                                if not (is_within_bounds(tx, ty) and is_position_free(self.board, tx, ty, player)):
                                    continue
                                if (tx, ty) in l_positions:
                                    continue

                                neutral_move = {"from": (nx, ny), "to": (tx, ty)}
                                move_with_neutral = base_move.copy()
                                move_with_neutral["neutral_move"] = neutral_move

                                if validate_move(self.board, move_with_neutral, player):
                                    legal_moves.add((
                                        frozenset(move_with_neutral["L_piece"].items()),
                                        frozenset(move_with_neutral["neutral_move"].items()),
                                    ))

        return [
            {"L_piece": dict(l_piece), "neutral_move": dict(neutral_move) if neutral_move else None}
            for l_piece, neutral_move in legal_moves
        ]

    def apply_move(self, move):
        """
        Apply a move to the game state.
        Args:
            move (dict): The move to be applied, containing L-piece and optional neutral piece moves.
        """
        # Increment turn counter
        self.total_turns += 1

        # Extract L-piece move data
        l_data = move["L_piece"]
        x, y, config = l_data["x"], l_data["y"], l_data["config"]
        l_positions = get_l_positions(x, y, config)

        # Clear the current L-piece positions for the player
        for row in range(4):
            for col in range(4):
                if self.board[row][col] == self.current_player:
                    self.board[row][col] = 0

        # Place the L-piece at the new positions
        for px, py in l_positions:
            self.board[py][px] = self.current_player

        # Apply the neutral piece move
        if move.get("neutral_move"):
            fx, fy = move["neutral_move"]["from"]
            tx, ty = move["neutral_move"]["to"]

            self.board[fy][fx] = 0
            self.board[ty][tx] = "N"

        # Update the current player
        self.current_player = 3 - self.current_player
        return self

    def copy(self):
        """
        Create a deep copy of the game state.
        Returns:
            GameState: A new instance of the game state with the same data.
        """
        new_state = GameState()
        new_state.board = [row[:] for row in self.board]
        new_state.l_positions = self.l_positions.copy()
        new_state.neutral_positions = self.neutral_positions.copy()
        new_state.current_player = self.current_player
        new_state.state_history = self.state_history.copy()  # Copy state history
        new_state.total_turns = self.total_turns  # Copy turn counter
        return new_state