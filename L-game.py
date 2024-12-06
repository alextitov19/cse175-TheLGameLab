import math # Minimax.py imports math

import tkinter as tk # Visualizer.py imports tkinter

#--------------------------------Game State--------------------------------

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
    

#--------------------------------Heuristic--------------------------------

def evaluate_board(state, player):
    """
    Heuristic evaluation function for the L-game.
    Scores the board state based on factors like mobility and control.
    
    Args:
        state (GameState): The current game state.
        player (int): The player for whom to evaluate the board (1 or 2).
    
    Returns:
        int: A score representing the desirability of the state for the player.
    """
    opponent = 3 - player

    # Number of legal moves for the player
    player_moves = len(state.get_legal_moves(player))

    # Number of legal moves for the opponent
    opponent_moves = len(state.get_legal_moves(opponent))

    # Control of the neutral pieces (e.g., closer neutral pieces give advantage)
    print("L positions", state.l_positions[player])
    neutral_control = distance_to_l(state.neutral_positions, state.l_positions[player])
    

    # Combine factors into a weighted score
    # Favor states with more moves for the player and fewer moves for the opponent
    score = (
        10 * player_moves  # Mobility
        - 10 * opponent_moves  # Limit opponent's mobility
        + 5 * neutral_control  # Control of neutral pieces
    )

    return score

from utils import get_l_positions  # Ensure this is imported

def distance_to_l(neutral_positions, l_data):
    """
    Calculate the sum of distances from neutral pieces to the L-piece.
    Converts the L-piece's corner and config into positions if necessary.

    Args:
        neutral_positions (list[tuple]): Positions of neutral pieces [(x1, y1), (x2, y2)].
        l_data (dict or list[tuple]): Either a dictionary with {'x', 'y', 'config'} or a list of tuples.

    Returns:
        int: Total distance from neutral pieces to the L-piece.
    """
    # If l_data is a dictionary, calculate positions using get_l_positions
    if isinstance(l_data, dict):
        l_positions = get_l_positions(l_data['x'], l_data['y'], l_data['config'])
    elif isinstance(l_data, list):
        l_positions = l_data  # Already in the correct format
    else:
        raise ValueError(f"Invalid l_data format: {l_data}")

    total_distance = 0
    for nx, ny in neutral_positions:
        min_distance = float("inf")
        for lx, ly in l_positions:
            distance = abs(nx - lx) + abs(ny - ly)  # Manhattan distance
            min_distance = min(min_distance, distance)
        total_distance += min_distance
    return total_distance

#--------------------------------Minimax--------------------------------

def minimax(state, depth, maximizing_player, evaluate_fn, get_legal_moves_fn, apply_move_fn, is_terminal_fn):
    """
    Implements the Minimax algorithm.

    Args:
        state: The current game state.
        depth: Maximum depth to search (use math.inf for no limit).
        maximizing_player: True if the current player is the maximizing player.
        evaluate_fn: Function to evaluate the state (heuristic).
        get_legal_moves_fn: Function to get legal moves for the current player.
        apply_move_fn: Function to apply a move and return the resulting state.
        is_terminal_fn: Function to check if the state is terminal.

    Returns:
        tuple: (best_move, best_score)
    """
    if depth == 0 or is_terminal_fn(state):
        return None, evaluate_fn(state)

    if maximizing_player:
        best_score = -math.inf
        best_move = None
        for move in get_legal_moves_fn(state, maximizing_player):
            new_state = apply_move_fn(state, move)
            _, score = minimax(new_state, depth - 1, False, evaluate_fn, get_legal_moves_fn, apply_move_fn, is_terminal_fn)
            if score > best_score:
                best_score = score
                best_move = move
        return best_move, best_score
    else:
        best_score = math.inf
        best_move = None
        for move in get_legal_moves_fn(state, maximizing_player):
            new_state = apply_move_fn(state, move)
            _, score = minimax(new_state, depth - 1, True, evaluate_fn, get_legal_moves_fn, apply_move_fn, is_terminal_fn)
            if score < best_score:
                best_score = score
                best_move = move
        return best_move, best_score
    
def alpha_beta_pruning(state, depth, alpha, beta, maximizing_player, evaluate_fn, get_legal_moves_fn, apply_move_fn, is_terminal_fn):
    """
    Perform minimax with alpha-beta pruning.
    
    Args:
        state: The current game state.
        depth: The depth of the search.
        alpha: Alpha value for pruning.
        beta: Beta value for pruning.
        maximizing_player: True if the current player is maximizing, False otherwise.
        evaluate_fn: The evaluation function to score the board.
        get_legal_moves_fn: Function to get legal moves for the current state.
        apply_move_fn: Function to apply a move and generate a new state.
        is_terminal_fn: Function to check if the game is in a terminal state.
        
    Returns:
        best_move: The best move for the current player.
        score: The score of the best move.
    """
    current_player = 1 if maximizing_player else 2  # Determine the current player
    if depth == 0 or is_terminal_fn(state):
        print(f"Evaluating terminal state at depth {depth}: {state}")
        return None, evaluate_fn(state, current_player)  # Pass the current player to evaluate_fn

    print(f"Alpha-beta pruning at depth {depth}, maximizing: {maximizing_player}")
    print(f"State:\n{state}")
    print(f"Alpha: {alpha}, Beta: {beta}")

    best_move = None
    if maximizing_player:
        max_eval = float("-inf")
        for move in get_legal_moves_fn(state):
            print(f"Maximizing, considering move: {move}")
            new_state = apply_move_fn(state, move)
            if new_state is None:
                print(f"Error: new_state is None after applying move: {move}")
            _, eval = alpha_beta_pruning(new_state, depth - 1, alpha, beta, False, evaluate_fn, get_legal_moves_fn, apply_move_fn, is_terminal_fn)
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return best_move, max_eval
    else:
        min_eval = float("inf")
        for move in get_legal_moves_fn(state):
            print(f"Minimizing, considering move: {move}")
            new_state = apply_move_fn(state, move)
            _, eval = alpha_beta_pruning(new_state, depth - 1, alpha, beta, True, evaluate_fn, get_legal_moves_fn, apply_move_fn, is_terminal_fn)
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return best_move, min_eval
    
def evaluate_fn(state):
    """
    A heuristic evaluation function for the game state.
    This is a placeholder and should be replaced with a game-specific evaluation.

    Args:
        state: The game state to evaluate.

    Returns:
        int: A score indicating the desirability of the state for the maximizing player.
    """
    # Example: Favor states with more legal moves for the current player
    current_player_moves = len(state.get_legal_moves(state.current_player))
    opponent_moves = len(state.get_legal_moves(3 - state.current_player))
    return current_player_moves - opponent_moves


def is_terminal_fn(state):
    """
    Checks if the game state is terminal.

    Args:
        state: The game state to check.

    Returns:
        bool: True if the state is terminal, False otherwise.
    """
    return not state.get_legal_moves(state.current_player)


def apply_move_fn(state, move):
    """
    Apply a move to the game state and return the new state.

    Args:
        state: The current game state.
        move: The move to apply.

    Returns:
        state: The new game state.
    """
    new_state = state.copy()  # Assume state has a copy method
    new_state.apply_move(move)
    return new_state

#--------------------------------Utils--------------------------------

def parse_move_input(input_string):
    """
    Parse a human-readable move input string into a structured format.
    Example input: "1 2 0 4 3 1 1"
    Returns:
        dict: A dictionary containing the parsed move:
              {
                  "L_piece": {"x": 1, "y": 2, "config": 0},
                  "neutral_move": {"from": (4, 3), "to": (1, 1)} or None
              }
    """
    parts = input_string.split()
    if len(parts) < 3 or len(parts) not in {3, 7}:
        raise ValueError("Invalid move format. Expected 3 or 7 parts.")

    move = {
        "L_piece": {
            "x": int(parts[0]),
            "y": int(parts[1]),
            "config": int(parts[2])  # Now expects configuration ID
        },
        "neutral_move": None
    }

    if len(parts) == 7:
        move["neutral_move"] = {
            "from": (int(parts[3]), int(parts[4])),
            "to": (int(parts[5]), int(parts[6]))
        }

    return move


def is_within_bounds(x, y):
    """
    Check if a position (x, y) is within the bounds of the 4x4 grid.
    Returns:
        bool: True if within bounds, False otherwise.
    """
    return 0 <= x < 4 and 0 <= y < 4


def is_position_free(board, x, y, player):
    """
    Check if a position (x, y) is free (not occupied) on the board.
    Returns:
        bool: True if free, False otherwise.
    """
    return board[y][x] == 0 or board[y][x] == player


def validate_move(board, move, player):
    """
    Validate a move to ensure it adheres to the game's rules.
    Args:
        board (list[list[int]]): The current game board.
        move (dict): The parsed move dictionary.
    Returns:
        bool: True if the move is valid, False otherwise.
    """
    # Validate L-piece move
    l_piece = move["L_piece"]
    x, y, config = l_piece["x"], l_piece["y"], l_piece["config"]
    l_positions = get_l_positions(x, y, config)
    print("L Positions:", l_positions)

    #  Get the current L-piece positions on the board
    current_l_positions = []
    for row in range(4):
        for col in range(4):
            if board[row][col] == player:
                current_l_positions.append((col, row))  # Note: (col, row) matches x, y coordinates

    # Ensure the L-piece is moved to a different position
    if set(l_positions) == set(current_l_positions):
        print("L-piece was not moved")
        return False

    if not all(is_within_bounds(px, py) for px, py in l_positions):
        print("Not within bounds")
    if not all(is_position_free(board, px, py, player) for px, py in l_positions):
        print("Not free")

    # Ensure all positions are within bounds and free
    if not all(is_within_bounds(px, py) and is_position_free(board, px, py, player) for px, py in l_positions):
        return False

    # Validate neutral piece move, if applicable
    neutral_move = move["neutral_move"]
    if neutral_move:
        fx, fy = neutral_move["from"]
        tx, ty = neutral_move["to"]
        if not (is_within_bounds(tx, ty) and is_position_free(board, tx, ty, 0)):
            return False
         # Ensure the neutral piece is not moved to a position occupied by the new L-piece
        if (tx, ty) in l_positions:
            print("Neutral piece overlaps with L-piece")
            return False
        if board[fy][fx] != "N":
            return False

    return True


def get_l_positions(x, y, config):
    """
    Get the grid positions occupied by an L-piece based on its configuration.
    
    Args:
        x (int): X-coordinate of the L's pivot (corner).
        y (int): Y-coordinate of the L's pivot (corner).
        config (int): Configuration ID (0-7).
                      - 0-3: Rotated clockwise.
                      - 4-7: Mirrored orientations.
    
    Returns:
        list[tuple]: List of grid positions occupied by the L-piece.
    """
    if config == 0:  
        return [(x, y), (x, y + 1), (x, y + 2), (x + 1, y + 2)]
    elif config == 1:  
        return [(x, y), (x, y + 1), (x, y + 2), (x - 1, y + 2)]
    elif config == 2:  
        return [(x, y), (x, y - 1), (x, y - 2), (x + 1, y - 2)]
    elif config == 3: 
        return [(x, y), (x, y - 1), (x, y - 2), (x - 1, y - 2)]
    elif config == 4:  
        return [(x, y), (x + 1, y), (x + 2, y), (x + 2, y - 1)]
    elif config == 5:  
        return [(x, y), (x + 1, y), (x + 2, y), (x + 2, y + 1)]
    elif config == 6: 
        return [(x, y), (x - 1, y), (x - 2, y), (x - 2, y - 1)]
    elif config == 7:  
        return [(x, y), (x - 1, y), (x - 2, y), (x - 2, y + 1)]
    return []


def apply_l_piece_move(board, x, y, config, player):
    """
    Apply the L-piece move on the board.
    """
    # Clear current L-piece positions
    for row in range(4):
        for col in range(4):
            if board[row][col] == player:
                board[row][col] = 0

    # Add new L-piece positions
    l_positions = get_l_positions(x, y, config)
    for px, py in l_positions:
        board[py][px] = player


def apply_neutral_piece_move(board, neutral_move):
    """
    Apply the neutral piece move on the board.
    """
    if neutral_move:
        fx, fy = neutral_move["from"]
        tx, ty = neutral_move["to"]
        board[fy][fx] = 0  # Remove neutral piece from old position
        board[ty][tx] = "N"  # Place neutral piece in new position


def apply_move(board, move, player):
    """
    Apply a complete move (L-piece + optional neutral move) to the board.
    """
    l_piece = move["L_piece"]
    apply_l_piece_move(board, l_piece["x"], l_piece["y"], l_piece["config"], player)

    if move["neutral_move"]:
        apply_neutral_piece_move(board, move["neutral_move"])


def print_board(board):
    """
    Print the current state of the board in a readable format.
    Args:
        board (list[list[int]]): The 4x4 game board.
    """
    print("\nCurrent Board:")
    for row in board:
        # Replace 0 with ".", 1 and 2 for players, and "N" for neutral pieces
        print(" ".join(str(cell) if cell != 0 else "." for cell in row))
    print()

def draw_all_l_configurations():
    """
    Draw all 8 configurations of the L-piece in 2 rows of 4, each on a 3x3 board, with 5 spaces between each.
    """
    print("\nAll L-piece Configurations (0-7):\n")

    configs = [[0, 0, 0], [1, 0, 1], [0, 2, 2], [1, 2, 3], [0, 1, 4], [0, 0, 5], [2, 1, 6], [2, 0, 7]]

    # Initialize the boards for all 8 configurations
    boards = []
    for x, y, c in configs:
        # Create a blank 3x3 board
        blank_board = [[" " for _ in range(3)] for _ in range(3)]

        # Get the L-piece positions
        l_positions = get_l_positions(x, y, c)

        # Mark the L-piece on the blank board
        for px, py in l_positions:
            if 0 <= px < 3 and 0 <= py < 3:
                blank_board[py][px] = "*"
        
        # Add the board to the list
        boards.append((c, blank_board))

    # Print configurations in 2 rows of 4
    for row_start in range(0, 8, 4):
        # Print headers (configuration IDs)
        header_row = "     ".join(f"Config {c:<2}" for c, _ in boards[row_start:row_start + 4])
        print(header_row)

        # Print the boards row by row
        for board_row in range(3):  # Each board has 3 rows
            row = "         ".join(" ".join(board[board_row]) for _, board in boards[row_start:row_start + 4])
            print("  " + row)
        
        print()  # Add a blank line between rows

#--------------------------------Visualizer--------------------------------

class GameVisualizer:
    def __init__(self, board):
        self.board = board
        self.window = tk.Tk()
        self.window.title("L-Game")
        self.cell_size = 100  # Size of each cell
        self.canvas = tk.Canvas(
            self.window,
            width=self.cell_size * 4,
            height=self.cell_size * 4,
            bg="white"
        )
        self.canvas.pack()

    def draw_board(self):
        """Draw the current board on the canvas."""
        self.canvas.delete("all")  # Clear the canvas

        for y, row in enumerate(self.board):
            for x, cell in enumerate(row):
                x0 = x * self.cell_size
                y0 = y * self.cell_size
                x1 = x0 + self.cell_size
                y1 = y0 + self.cell_size

                # Draw cell background
                self.canvas.create_rectangle(x0, y0, x1, y1, fill="lightgray", outline="black")

                # Draw pieces
                if cell == 1:
                    self.canvas.create_text(
                        (x0 + x1) // 2,
                        (y0 + y1) // 2,
                        text="P1",
                        font=("Helvetica", 20),
                        fill="blue"
                    )
                elif cell == 2:
                    self.canvas.create_text(
                        (x0 + x1) // 2,
                        (y0 + y1) // 2,
                        text="P2",
                        font=("Helvetica", 20),
                        fill="red"
                    )
                elif cell == "N":
                    self.canvas.create_text(
                        (x0 + x1) // 2,
                        (y0 + y1) // 2,
                        text="N",
                        font=("Helvetica", 20),
                        fill="green"
                    )

    def update_board(self, board):
        """Update the board and redraw."""
        self.board = board
        self.draw_board()

    def run(self):
        """Run the Tkinter main loop."""
        self.draw_board()
        self.window.mainloop()

#--------------------------------Main--------------------------------


def human_move(game_state):
    """
    Prompt the human player to make a move.
    """
    print_board(game_state.board)
    while True:
        try:
            draw_all_l_configurations()
            move_input = input("Enter your move (e.g., '1 2 E 4 3 1 1'): ")
            move = parse_move_input(move_input)
            if validate_move(game_state.board, move, game_state.current_player):
                return move
            else:
                print("Invalid move. Please try again.")
        except ValueError as e:
            print(f"Error: {e}. Try again.")


def computer_move(game_state, depth=3):
    """
    Determine the computer's move using minimax with alpha-beta pruning.
    """
    print("Computer is thinking...")
    print("Terminal state:", game_state.is_terminal())
    move, _ = alpha_beta_pruning(
        state=game_state,
        depth=depth,
        alpha=-math.inf,
        beta=math.inf,
        maximizing_player=(game_state.current_player == 1),
        evaluate_fn=evaluate_board,
        get_legal_moves_fn=lambda state: state.get_legal_moves(state.current_player),
        apply_move_fn=lambda state, move: debug_apply_move(state, move),
        is_terminal_fn=lambda state: state.is_terminal(),
    )
    print(f"Computer's Move: {move}")
    return move

def main():
    """
    Main function to run the L-game.
    """
    print("\nWelcome to the L-game!\n")
    print("You can choose one of the following modes:")
    print("1. Human vs Human")
    print("2. Human vs Computer")
    print("3. Computer vs Computer\n")

    mode = input("Enter the mode number (1, 2, or 3): ")
    if mode not in {"1", "2", "3"}:
        print("Invalid mode. Exiting.")
        return
    
    x = 1
    if mode == "2":
        s = input("\nDo you want to go first? (y/n): ")
        if s == "n":
            x = 2
        elif s != "y":
            print("Invalid input. Exiting.")
            return
            

    game_state = GameState()

    visualizer = GameVisualizer(game_state.board)

    # Draw the initial board before the game loop starts
    visualizer.draw_board()

    while not game_state.is_terminal():
        print(f"\nPlayer {game_state.current_player}'s turn.")

        if (mode == "1") or (mode == "2" and game_state.current_player == x):
            # Human move
            move = human_move(game_state)
            game_state.apply_move(move)

            # Update GUI immediately after the human move
            visualizer.update_board(game_state.board)
            visualizer.window.update()  # Force GUI to process all events
        else:
            # Computer move
            move = computer_move(game_state)
            game_state.apply_move(move)

            # Update GUI for the computer's move
            visualizer.update_board(game_state.board)

    # Final game over logic
    print("\nFinal Board State:")
    print_board(game_state.board)
    print(f"Game Over! Player {3 - game_state.current_player} wins!")
    visualizer.run()  # Keeps the GUI window open at the end of the game
    
def debug_apply_move(state, move):
    """
    Debug wrapper for applying moves to a game state.
    """
    print(f"Applying move: {move}")
    new_state = state.copy()
    if new_state is None:
        print("Error: state.copy() returned None.")
        return None
    try:
        new_state.apply_move(move)
        return new_state
    except Exception as e:
        print(f"Error during apply_move: {e}")
        return None
    
if __name__ == "__main__":
    main()