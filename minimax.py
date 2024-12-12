import math


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