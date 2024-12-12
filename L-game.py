import math
#--------------------------------------- Game State ---------------------------------------
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
        legal_moves = []

        # Iterate over all possible L-piece configurations
        for x in range(4):
            for y in range(4):
                for config in range(8):  # Use 0-7 for L-piece configurations
                    l_positions = get_l_positions(x, y, config)

                    # Validate L-piece positions early
                    if not all(
                        is_within_bounds(px, py) and is_position_free(self.board, px, py, player)
                        for px, py in l_positions
                    ):
                        continue

                    # Create the base L-piece move
                    base_move = {"L_piece": {"x": x, "y": y, "config": config}, "neutral_move": None}

                    # Validate the base L-piece move
                    if not validate_move(self.board, base_move, player):
                        continue

                    # Add the L-piece move without neutral piece adjustment
                    legal_moves.append(base_move)

                    # Iterate over neutral piece moves
                    for nx, ny in self.neutral_positions:
                        for tx in range(4):
                            for ty in range(4):
                                # Skip invalid neutral piece moves
                                if not is_within_bounds(tx, ty) or not is_position_free(self.board, tx, ty, player):
                                    continue
                                if (tx, ty) in l_positions:
                                    continue  # Neutral piece cannot overlap with the new L-piece positions

                                # Add the neutral piece move
                                neutral_move = {"from": (nx, ny), "to": (tx, ty)}
                                move_with_neutral = base_move.copy()
                                move_with_neutral["neutral_move"] = neutral_move

                                if validate_move(self.board, move_with_neutral, player):
                                    legal_moves.append(move_with_neutral)

        return legal_moves

    def apply_move(self, move):
        """
        Apply a move to the game state.
        Args:
            move (dict): The move to be applied, containing L-piece and optional neutral piece moves.
        """
        # Increment turn counter
        self.total_turns += 1

        # Extract L-piece move data
        # print("Move: ", move)
        l_data = move["L_piece"]
        # print("L_data: ", l_data)
        x, y, config = l_data["x"], l_data["y"], l_data["config"]
        # print("X, Y, Config: ", x, y, config)
        l_positions = get_l_positions(x, y, config)

        # Clear the current L-piece positions for the current player
        for row in range(4):
            for col in range(4):
                if self.board[row][col] == self.current_player:
                    self.board[row][col] = 0  # Clear the player's L-piece from the board

        # Apply the neutral piece move, if applicable
        if move.get("neutral_move"):
            fx, fy = move["neutral_move"]["from"]
            if self.board[fy][fx] == "N":  # Ensure the source position is a neutral piece
                self.board[fy][fx] = 0  # Clear the neutral piece's source position

        # Place the L-piece at the new positions
        for px, py in l_positions:
            # print("Position: ", px, py)
            self.board[py][px] = self.current_player  # Place the player's L-piece

        # Place the neutral piece in its new position, if applicable
        if move.get("neutral_move"):
            tx, ty = move["neutral_move"]["to"]
            self.board[ty][tx] = "N"  # Place the neutral piece in the new position

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
    
#--------------------------------------- Heuristic ---------------------------------------



def evaluate_board(state, player):
    """
    Ultra-aggressive heuristic evaluation function for the L-game.
    Focuses almost exclusively on minimizing the opponent's possible moves.

    Args:
        state (GameState): The current game state.
        player (int): The player for whom to evaluate the board (1 or 2).

    Returns:
        int: A score representing the desirability of the state for the player.
    """
    opponent = 3 - player

    # Cache legal moves to avoid redundant computations
    player_moves = len(state.get_legal_moves(player))
    opponent_moves = len(state.get_legal_moves(opponent))

    # Check for terminal states early
    if player_moves == 0:
        return -math.inf  # Current player has no moves, losing state
    if opponent_moves == 0:
        return math.inf  # Opponent has no moves, winning state

    # Neutral piece control (Manhattan distance from player's L-piece)
    neutral_control = distance_to_l(state.neutral_positions, state.l_positions[player])

    # Opponent blockade: Drastically penalize states where the opponent has very few moves
    # Using cubic scaling for even stronger penalty
    opponent_blockade = (5 - opponent_moves) ** 3 if opponent_moves < 5 else 0

    # Positioning: Favor central positions slightly and heavily penalize edge positions
    central_positions = {(1, 1), (1, 2), (2, 1), (2, 2)}
    l_positions = get_l_positions(
        state.l_positions[player]['x'],
        state.l_positions[player]['y'],
        state.l_positions[player]['config']
    )
    central_score = sum(1 for x, y in l_positions if (x, y) in central_positions)
    edge_penalty = sum(1 for x, y in l_positions if x in {0, 3} or y in {0, 3})

    # Combine factors into a weighted score
    # Heavily prioritize reducing the opponent's mobility, even at the cost of minor penalties for player positioning
    score = (
        10 * player_moves              # Keep a reasonable weight for player's mobility
        - 30 * opponent_moves          # Very strongly limit opponent's mobility
        - 5 * neutral_control          # Neutral piece control remains a secondary concern
        + 50 * opponent_blockade       # Make blocking the opponent the top priority
        + 2 * central_score            # Slightly reward central positioning
        - 5 * edge_penalty             # Heavily penalize edge positions
    )

    return score


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

    # Calculate distances from each neutral piece to the nearest L-piece segment
    for nx, ny in neutral_positions:
        min_distance = min(abs(nx - lx) + abs(ny - ly) for lx, ly in l_positions)  # Manhattan distance
        total_distance += min_distance

    return total_distance

#--------------------------------------- Minimax ---------------------------------------


class SymmetryManager:
    @staticmethod
    def generate_all_rotations(board):
        """Generate all rotations of the board."""
        rotations = []
        current = board
        for _ in range(4):
            current = [list(row) for row in zip(*current[::-1])]
            rotations.append(tuple(tuple(row) for row in current))
        return rotations

    @staticmethod
    def generate_all_symmetries(board):
        """Generate all symmetrical configurations of the board."""
        rotations = SymmetryManager.generate_all_rotations(board)
        reflections = [tuple(tuple(reversed(row)) for row in rotation) for rotation in rotations]
        return set(rotations + reflections)
    
def alpha_beta_pruning(state, depth, alpha, beta, maximizing_player, evaluate_fn, get_legal_moves_fn, apply_move_fn, is_terminal_fn, visited_states=None):
    """
    Perform minimax with alpha-beta pruning and symmetry pruning.

    Args:
        state: Current game state.
        depth: Depth to search.
        alpha: Alpha value for pruning.
        beta: Beta value for pruning.
        maximizing_player: Whether this is the maximizing player's turn.
        evaluate_fn: Function to evaluate board states.
        get_legal_moves_fn: Function to generate legal moves.
        apply_move_fn: Function to apply a move and return the new state.
        is_terminal_fn: Function to check if the game is over.
        visited_states: A set of visited states to prune symmetrical states.

    Returns:
        best_move: The best move determined by the search.
        best_score: The score associated with the best move.
    """
    if visited_states is None:
        visited_states = set()

    # Use SymmetryManager to generate symmetries and prune symmetrical states
    symmetries = SymmetryManager.generate_all_symmetries(state.board)
    if any(symmetry in visited_states for symmetry in symmetries):
        return None, evaluate_fn(state, state.current_player)  # Skip symmetrical states

    # Add symmetries of the current state to visited states
    visited_states.update(symmetries)

    # Terminal node or max depth reached
    if depth == 0 or is_terminal_fn(state):
        return None, evaluate_fn(state, state.current_player)

    best_move = None

    if maximizing_player:
        max_eval = -math.inf
        for move in sorted(get_legal_moves_fn(state), key=lambda m: evaluate_fn(apply_move_fn(state.copy(), m), state.current_player), reverse=True):
            new_state = apply_move_fn(state, move)
            _, eval = alpha_beta_pruning(
                state=new_state,
                depth=depth - 1,
                alpha=alpha,
                beta=beta,
                maximizing_player=False,
                evaluate_fn=evaluate_fn,
                get_legal_moves_fn=get_legal_moves_fn,
                apply_move_fn=apply_move_fn,
                is_terminal_fn=is_terminal_fn,
                visited_states=visited_states,
            )
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # Prune branches
        return best_move, max_eval
    else:
        min_eval = math.inf
        for move in sorted(get_legal_moves_fn(state), key=lambda m: evaluate_fn(apply_move_fn(state.copy(), m), state.current_player)):
            new_state = apply_move_fn(state, move)
            _, eval = alpha_beta_pruning(
                state=new_state,
                depth=depth - 1,
                alpha=alpha,
                beta=beta,
                maximizing_player=True,
                evaluate_fn=evaluate_fn,
                get_legal_moves_fn=get_legal_moves_fn,
                apply_move_fn=apply_move_fn,
                is_terminal_fn=is_terminal_fn,
                visited_states=visited_states,
            )
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break  # Prune branches
        return best_move, min_eval
    
    #--------------------------------------- Utils ---------------------------------------
def parse_move_input(input_string):
    """
    Parse a human-readable move input string into a structured format.
    Converts input from 1-based to 0-based indexing.
    
    Example input: "1 2 0 4 3 1 1"
    Returns:
        dict: A dictionary containing the parsed move:
              {
                  "L_piece": {"x": 0, "y": 1, "config": 0},
                  "neutral_move": {"from": (3, 2), "to": (0, 0)} or None
              }
    """
    parts = input_string.split()
    if len(parts) < 3 or len(parts) not in {3, 7}:
        raise ValueError("Invalid move format. Expected 3 or 7 parts.")
    
    if parts[2].upper() not in {"E", "W", "N", "S"}:
        raise ValueError("Invalid L-piece orientation. Expected 'E', 'W', 'N', or 'S'.")
    

    #TODO: If config = letter, convert to number by checking which one of the two numbers is valid move
    num = lton(int(parts[0]) - 1, int(parts[1]) - 1, parts[2].upper())
    print("Num:", num)

    move = {
        "L_piece": {
            "x": int(parts[0]) - 1,
            "y": int(parts[1]) - 1,
            "config": num  # Configuration ID remains unchanged
        },
        "neutral_move": None
    }

    if len(parts) == 7:
        move["neutral_move"] = {
            "from": (int(parts[3]) - 1, int(parts[4]) - 1),
            "to": (int(parts[5]) - 1, int(parts[6]) - 1)
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
    if (move["L_piece"]["config"] == None):
        return False
    """
    Validate a move to ensure it adheres to the game's rules.
    Args:
        board (list[list[int]]): The current game board.
        move (dict): The parsed move dictionary.
        player (int): The player making the move (1 or 2).
    Returns:
        bool: True if the move is valid, False otherwise.
    """
    # Validate L-piece move
    l_piece = move["L_piece"]
    x, y, config = l_piece["x"], l_piece["y"], l_piece["config"]
    l_positions = get_l_positions(x, y, config)

    # Get the current L-piece positions on the board
    current_l_positions = []
    for row in range(4):
        for col in range(4):
            if board[row][col] == player:
                current_l_positions.append((col, row))  # Note: (col, row) matches x, y coordinates

    # Ensure the L-piece is moved to a different position
    if set(l_positions) == set(current_l_positions):
        # The L-piece was not moved
        return False

    # Ensure all positions for the new L-piece are within bounds and free
    if not all(is_within_bounds(px, py) for px, py in l_positions):
        return False

    # Temporarily "remove" the neutral piece being moved
    neutral_move = move["neutral_move"]
    temp_board = [row[:] for row in board]
    if neutral_move:
        fx, fy = neutral_move["from"]
        tx, ty = neutral_move["to"]
        if not is_within_bounds(tx, ty):
            return False
        if temp_board[fy][fx] == "N":  # Neutral piece at the source position
            temp_board[fy][fx] = 0  # Temporarily clear the source position
        else:
            return False
        if (tx, ty) in l_positions:
            # Ensure the neutral piece is not moved to a position occupied by the new L-piece
            return False

    # Check if all L-piece positions are free on the updated board
    if not all(is_position_free(temp_board, px, py, player) for px, py in l_positions):
        return False

    # Validate the neutral piece destination
    if neutral_move:
        if temp_board[neutral_move["to"][1]][neutral_move["to"][0]] != 0:
            return False

    return True


def get_l_positions(x, y, config):
    """
    Get the grid positions occupied by an L-piece based on its configuration.
    The (x, y) coordinates represent the corner of the L (where the long and short pieces meet).
    
    Args:
        x (int): X-coordinate of the L's corner.
        y (int): Y-coordinate of the L's corner.
        config (int): Configuration ID (0-7).
                      - 0-3: Rotated clockwise.
                      - 4-7: Mirrored orientations.
    
    Returns:
        list[tuple]: List of grid positions occupied by the L-piece.
    """
    if config == 0:  # Vertical long, short to the right (corner at top-left)
        return [(x, y), (x, y + 1), (x, y + 2), (x + 1, y)]
    elif config == 1:  # Vertical long, short to the left
        return [(x, y), (x, y + 1), (x, y + 2), (x - 1, y)]
    elif config == 2:  # Vertical long, short to the right (corner at bottom-left)
        return [(x, y), (x, y - 1), (x, y - 2), (x + 1, y)]
    elif config == 3:  # Vertical long, short to the left
        return [(x, y), (x, y - 1), (x, y - 2), (x - 1, y)]
    elif config == 4:  # Horizontal long, short down
        return [(x, y), (x + 1, y), (x + 2, y), (x, y + 1)]
    elif config == 5:  # Horizontal long, short up
        return [(x, y), (x + 1, y), (x + 2, y), (x, y - 1)]
    elif config == 6:  # Horizontal long, short down (corner at right)
        return [(x, y), (x - 1, y), (x - 2, y), (x, y + 1)]
    elif config == 7:  # Horizontal long, short up
        return [(x, y), (x - 1, y), (x - 2, y), (x, y - 1)]
    return []

def lton(x, y, letter):
    print("Letter:", letter)
    """
    Get the grid positions occupied by an L-piece based on its configuration.
    The (x, y) coordinates represent the corner of the L (where the long and short pieces meet).
    
    Args:
        x (int): X-coordinate of the L's corner.
        y (int): Y-coordinate of the L's corner.
        config (int): Configuration ID (0-7).
                      - 0-3: Rotated clockwise.
                      - 4-7: Mirrored orientations.
    
    Returns:
        list[tuple]: List of grid positions occupied by the L-piece.
    """
    configs = {}
    if letter == "E":
        configs = {
            0: [(x, y), (x, y + 1), (x, y + 2), (x + 1, y)],
            2: [(x, y), (x, y - 1), (x, y - 2), (x + 1, y)]
        }
    elif letter == "W":
        configs = {
            1: [(x, y), (x, y + 1), (x, y + 2), (x - 1, y)],
            3: [(x, y), (x, y - 1), (x, y - 2), (x - 1, y)]
        }
    elif letter == "N":
        configs = {
            5: [(x, y), (x + 1, y), (x + 2, y), (x, y - 1)],
            7: [(x, y), (x - 1, y), (x - 2, y), (x, y - 1)]
        }
    elif letter == "S":
        configs = {
            4: [(x, y), (x + 1, y), (x + 2, y), (x, y + 1)],
            6: [(x, y), (x - 1, y), (x - 2, y), (x, y + 1)]
        }
    
    for index, config in configs.items():
        if all(is_within_bounds(px, py) for px, py in config):
            return index

    return None

def ntol(num):
    if num == 0 or num == 2:
        return "E"
    if num == 1 or num == 3:
        return "W"
    if num == 5 or num == 7:
        return "N"
    if num == 4 or num == 6:
        return "S"
    return "Invalid"



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

#--------------------------------------- Visualizer ---------------------------------------
import tkinter as tk

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

#--------------------------------------- Main ---------------------------------------


def human_move(game_state):
    """
    Prompt the human player to make a move.
    """
    print_board(game_state.board)
    while True:
        try:
            move_input = input("Enter your move (e.g., '1 2 E 4 3 1 1'): ")
            move = parse_move_input(move_input)
            if validate_move(game_state.board, move, game_state.current_player):
                return move
            else:
                print("Invalid move. Please try again.")
        except ValueError as e:
            print(f"Error: {e} Try again.")


def computer_move(game_state, depth=5):
    """
    Determine the computer's move using minimax with alpha-beta pruning.
    """
    print("Computer is thinking...")
    move, val = alpha_beta_pruning(
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
    # print("Computer's value:", val)
    adjusted_move = {
        "L_piece": {
            "x": move["L_piece"]["x"] + 1,
            "y": move["L_piece"]["y"] + 1,
            "config": ntol(move["L_piece"]["config"]),  # Configuration remains unchanged
        },
        "neutral_move": None if move["neutral_move"] is None else {
            "from": (move["neutral_move"]["from"][0] + 1, move["neutral_move"]["from"][1] + 1),
            "to": (move["neutral_move"]["to"][0] + 1, move["neutral_move"]["to"][1] + 1),
        },
    }

    s = f"{adjusted_move["L_piece"]["x"]} {adjusted_move["L_piece"]["y"]} {adjusted_move["L_piece"]["config"]}"

    if adjusted_move["neutral_move"]:
        s += f" {adjusted_move["neutral_move"]["from"][0]} {adjusted_move["neutral_move"]["from"][1]}"
        s += f" {adjusted_move["neutral_move"]["to"][0]} {adjusted_move["neutral_move"]["to"][1]}"

    print(f"Computer's move: {s}")

    # input("Press any key to continue...")  # Wait for key input
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

    while not game_state.is_terminal(check_ties=True):
        print(f"\nPlayer {game_state.current_player}'s turn.")

        if (mode == "1") or (mode == "2" and game_state.current_player == x):
            # Human move
            move = human_move(game_state)
            print("\nCurrent Board State:")
            print_board(game_state.board)
            game_state.apply_move(move)
            print("\nUpdated Board State:")
            print_board(game_state.board)
        

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

    # Check tie conditions
    board_hash = game_state._hash_board()
    if game_state.state_history.get(board_hash, 0) >= game_state.max_repetitions:
        print("Game Over! It's a tie due to repeated states.")
    elif game_state.total_turns >= game_state.max_turns:
        print("Game Over! It's a tie due to exceeding the maximum number of turns.")
    else:
        print(f"Game Over! Player {3 - game_state.current_player} wins!")

    visualizer.run()  # Keeps the GUI window open at the end of the game
    
def debug_apply_move(state, move):
    """
    Debug wrapper for applying moves to a game state.
    """
    # print(f"Applying move: {move}")
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