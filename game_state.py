# gamestate.py

from utils import is_within_bounds, is_position_free, get_l_positions

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
        new_state.board = [row[:] for row in self.board]
        new_state.l_positions = {k: v.copy() for k, v in self.l_positions.items()}
        new_state.neutral_positions = self.neutral_positions[:]
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
                                        legal_moves.append(move_with_neutral)

                        # Add the L-piece move without a neutral piece move
                        legal_moves.append(move)

        return legal_moves

    def apply_move(self, move):
        """
        Apply a move to the game state.
        Args:
            move (dict): A move dictionary containing L_piece and/or neutral_move.
        """
        # Apply L piece move
        l_data = move.get("L_piece")
        print("L data", l_data)
        if l_data:
            player = self.current_player
            self._clear_l_piece(player)
            l_positions = get_l_positions(l_data["x"], l_data["y"], l_data["config"])
            for x, y in l_positions:
                self.board[y][x] = player
            self.l_positions[player] = l_data

        # Apply neutral piece move
        neutral_move = move.get("neutral_move")
        if neutral_move:
            fx, fy = neutral_move["from"]
            tx, ty = neutral_move["to"]
            self.board[fy][fx] = 0  # Clear old neutral piece position
            self.board[ty][tx] = "N"  # Place in the new position
            self.neutral_positions.remove((fx, fy))
            self.neutral_positions.append((tx, ty))

        # Switch to the next player
        self.current_player = 3 - self.current_player

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