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