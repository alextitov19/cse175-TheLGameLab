def parse_move_input(input_string):
    parts = input_string.split()
    if len(parts) < 3 or len(parts) not in {3, 7} or parts[2].upper() not in {"E", "W", "N", "S"}:
        raise ValueError("Invalid move format. Expected 3 or 7 parts.")

    move = {
        "L_piece": {
            "x": int(parts[0]) - 1,
            "y": int(parts[1]) - 1,
            "config": parts[2]  # Configuration ID remains unchanged
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
    # l_positions = l_positions_array[0]
    # for px, py in l_positions:
    #     if not is_within_bounds(px, py):
    #         l_positions = l_positions_array[1]
    #         break
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
    if config.upper() == "E":
        positions = [
            [(x, y), (x, y + 1), (x, y + 2), (x + 1, y)],
            [(x, y), (x, y - 1), (x, y - 2), (x + 1, y)]
        ]
    elif config.upper() == "W":
        positions = [
            [(x, y), (x, y + 1), (x, y + 2), (x - 1, y)],
            [(x, y), (x, y - 1), (x, y - 2), (x - 1, y)]
        ]
    elif config.upper() == "N":
        positions = [
            [(x, y), (x + 1, y), (x + 2, y), (x, y - 1)],
            [(x, y), (x - 1, y), (x - 2, y), (x, y - 1)]
        ]
    elif config.upper() == "S":
        positions = [
            [(x, y), (x + 1, y), (x + 2, y), (x, y + 1)],
            [(x, y), (x - 1, y), (x - 2, y), (x, y + 1)]
        ]
    else:
        return []

    # Check if any of the positions are within bounds and return the first valid one
    for pos in positions:
        if all(is_within_bounds(px, py) for px, py in pos):
            return pos
    
    # If none are valid, return an empty list
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

