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