# minimax.py

import math


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
    Implements the Minimax algorithm with Alpha-Beta Pruning.

    Args:
        state: The current game state.
        depth: Maximum depth to search (use math.inf for no limit).
        alpha: Alpha value for pruning.
        beta: Beta value for pruning.
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
            _, score = alpha_beta_pruning(new_state, depth - 1, alpha, beta, False, evaluate_fn, get_legal_moves_fn, apply_move_fn, is_terminal_fn)
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break  # Prune the remaining branches
        return best_move, best_score
    else:
        best_score = math.inf
        best_move = None
        for move in get_legal_moves_fn(state, maximizing_player):
            new_state = apply_move_fn(state, move)
            _, score = alpha_beta_pruning(new_state, depth - 1, alpha, beta, True, evaluate_fn, get_legal_moves_fn, apply_move_fn, is_terminal_fn)
            if score < best_score:
                best_score = score
                best_move = move
            beta = min(beta, best_score)
            if beta <= alpha:
                break  # Prune the remaining branches
        return best_move, best_score


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