# heuristic.py

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
    neutral_control = sum(
        distance_to_l(state.neutral_positions, state.l_positions[player])
    )

    # Combine factors into a weighted score
    # Favor states with more moves for the player and fewer moves for the opponent
    score = (
        10 * player_moves  # Mobility
        - 10 * opponent_moves  # Limit opponent's mobility
        + 5 * neutral_control  # Control of neutral pieces
    )

    return score


def distance_to_l(neutral_positions, l_positions):
    """
    Calculate the sum of distances from neutral pieces to the L-piece.
    This can be used as part of the heuristic to control neutral piece placement.

    Args:
        neutral_positions (list[tuple]): Positions of neutral pieces [(x1, y1), (x2, y2)].
        l_positions (list[tuple]): Positions of the L-piece [(x1, y1), ...]

    Returns:
        int: Total distance from neutral pieces to the L-piece.
    """
    total_distance = 0
    for nx, ny in neutral_positions:
        min_distance = float("inf")
        for lx, ly in l_positions:
            distance = abs(nx - lx) + abs(ny - ly)  # Manhattan distance
            min_distance = min(min_distance, distance)
        total_distance += min_distance
    return total_distance