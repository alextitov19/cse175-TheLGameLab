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