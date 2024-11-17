# utils.py

def parse_move_input(input_string):
    """
    Parse a human-readable move input string into a structured format.
    Example input: "1 2 E 4 3 1 1"
    Returns:
        dict: A dictionary containing the parsed move:
              {
                  "L_piece": {"x": 1, "y": 2, "orientation": "E"},
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
            "orientation": parts[2]
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


def is_position_free(board, x, y):
    """
    Check if a position (x, y) is free (not occupied) on the board.
    Returns:
        bool: True if free, False otherwise.
    """
    return board[y][x] == 0


def validate_move(board, move):
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
    x, y, orientation = l_piece["x"], l_piece["y"], l_piece["orientation"]
    l_positions = get_l_positions(x, y, orientation)

    if not all(is_within_bounds(px, py) and is_position_free(board, px, py) for px, py in l_positions):
        return False

    # Validate neutral piece move, if applicable
    neutral_move = move["neutral_move"]
    if neutral_move:
        fx, fy = neutral_move["from"]
        tx, ty = neutral_move["to"]
        if not (is_within_bounds(tx, ty) and is_position_free(board, tx, ty)):
            return False
        if board[fy][fx] != "N":
            return False

    return True

def get_l_positions(x, y, orientation):
    """
    Get the grid positions occupied by a 3x2 L-piece given its corner (x, y) and orientation.
    Args:
        x (int): X-coordinate of the L's corner.
        y (int): Y-coordinate of the L's corner.
        orientation (str): Orientation of the L ('N', 'S', 'E', 'W').
    Returns:
        list[tuple]: List of positions occupied by the L-piece.
    """
    if orientation == "E":  # L oriented East
        return [(x, y), (x, y + 1), (x + 1, y), (x + 1, y + 1)]
    elif orientation == "W":  # L oriented West
        return [(x, y), (x, y - 1), (x - 1, y), (x - 1, y - 1)]
    elif orientation == "N":  # L oriented North
        return [(x, y), (x - 1, y), (x, y - 1), (x - 1, y - 1)]
    elif orientation == "S":  # L oriented South
        return [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)]
    return []

def apply_l_piece_move(board, x, y, orientation, player):
    """
    Apply the L-piece move on the board.
    """
    for row in board:
        for col in range(len(row)):
            if row[col] == player:
                row[col] = 0  # Clear current L positions

    l_positions = get_l_positions(x, y, orientation)
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
    apply_l_piece_move(board, l_piece["x"], l_piece["y"], l_piece["orientation"], player)

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