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

    move = {
        "L_piece": {
            "x": int(parts[0]) - 1,
            "y": int(parts[1]) - 1,
            "config": int(parts[2])  # Configuration ID remains unchanged
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