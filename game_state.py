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
            1: {"x": 1, "y": 1, "config": 0},  # Player 1's L-piece
            2: {"x": 2, "y": 2, "config": 3},  # Player 2's L-piece
        }

        # Place the pieces on the board
        self._initialize_board()

        # Player 1 starts the game
        self.current_player = 1

    def _initialize_board(self):
        """
        Place the initial L pieces and neutral pieces on the board.
        """
        # Initialize board
        self.board = [[0 for _ in range(4)] for _ in range(4)]

        # Place neutral pieces
        for x, y in self.neutral_positions:
            self.board[y][x] = "N"

        # Place L-pieces for both players dynamically from self.l_positions
        for player, l_data in self.l_positions.items():
            x, y, config = l_data["x"], l_data["y"], l_data["config"]
            for px, py in get_l_positions(x, y, config):
                self.board[py][px] = player

    def copy(self):
        """
        Create a deep copy of the game state.
        """
        new_state = GameState()
        new_state.board = [row[:] for row in self.board]  # Deep copy the board
        new_state.l_positions = self.l_positions.copy()  # Copy player positions
        new_state.neutral_positions = self.neutral_positions.copy()  # Copy neutral pieces
        new_state.current_player = self.current_player
        return new_state

    def get_legal_moves(self, player):
        """
        Get all legal moves for the given player, combining L-piece and neutral piece moves.
        Args:
            player (int): The player (1 or 2) whose legal moves are to be generated.
        Returns:
            list[dict]: A list of legal moves, each containing the L-piece move and optionally a neutral piece move.
        """
        legal_moves = []

        # Get the current L-piece position and neutral piece positions
        l_data = self.l_positions[player]

        # Iterate over all possible L-piece configurations
        for x in range(4):
            for y in range(4):
                for config in range(8):  # Use 0-7 for L-piece configurations
                    l_positions = get_l_positions(x, y, config)
                    
                    # Check if the L-piece move is valid
                    if all(
                        is_within_bounds(px, py)
                        and is_position_free(self.board, px, py, player)
                        for px, py in l_positions
                    ):
                        move = {"L_piece": {"x": x, "y": y, "config": config}, "neutral_move": None}

                        # Add all possible neutral piece moves
                        for idx, (nx, ny) in enumerate(self.neutral_positions):
                            for tx in range(4):
                                for ty in range(4):
                                    if is_within_bounds(tx, ty) and is_position_free(self.board, tx, ty, player):
                                        neutral_move = {"from": (nx, ny), "to": (tx, ty)}
                                        move_with_neutral = move.copy()
                                        move_with_neutral["neutral_move"] = neutral_move
                                        if validate_move(self.board, move_with_neutral, player):
                                            legal_moves.append(move_with_neutral)

                        # Add the L-piece move without a neutral piece move
                        if validate_move(self.board, move, player):
                            legal_moves.append(move)

        return legal_moves

    def apply_move(self, move):
        """
        Apply a move to the game state.
        """
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

            # Clear old neutral piece position and place in new position
            self.board[fy][fx] = 0
            self.board[ty][tx] = "N"

        # Update the current player
        self.current_player = 3 - self.current_player
        return self

    def _clear_l_piece(self, player):
        """
        Clear the current L piece of the given player from the board.
        """
        for y in range(4):
            for x in range(4):
                if self.board[y][x] == player:
                    self.board[y][x] = 0

    def is_terminal(self):
        """
        Check if the current game state is terminal (no legal moves for the current player).
        """
        return len(self.get_legal_moves(self.current_player)) == 0