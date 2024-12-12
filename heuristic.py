import math
from utils import get_l_positions


def evaluate_board(state, player):
    """
    Enhanced heuristic evaluation function for the L-game.
    Scores the board state based on factors like mobility, neutral piece control, and strategic positioning.

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

    # Opponent blockade: Penalize states where the opponent's mobility is restricted
    opponent_blockade = max(0, 3 - opponent_moves)  # Strong penalty if opponent has <3 moves

    # Positioning: Reward central positions and penalize edges
    central_positions = {(1, 1), (1, 2), (2, 1), (2, 2)}
    l_positions = get_l_positions(
        state.l_positions[player]['x'],
        state.l_positions[player]['y'],
        state.l_positions[player]['config']
    )
    central_score = sum(1 for x, y in l_positions if (x, y) in central_positions)
    edge_penalty = sum(1 for x, y in l_positions if x in {0, 3} or y in {0, 3})

    # Combine factors into a weighted score
    score = (
        15 * player_moves             # Strong weight for mobility
        - 12 * opponent_moves          # Limit opponent's mobility
        - 5 * neutral_control          # Encourage neutral piece control
        + 20 * opponent_blockade       # Strongly reward blocking the opponent
        + 4 * central_score            # Reward central positioning
        - 3 * edge_penalty             # Penalize edge positions
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