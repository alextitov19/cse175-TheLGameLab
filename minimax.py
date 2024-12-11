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


def minimax(state, depth, maximizing_player, evaluate_fn, get_legal_moves_fn, apply_move_fn, is_terminal_fn, visited_states=None):
    """
    Implements the Minimax algorithm with symmetry pruning.
    """
    if visited_states is None:
        visited_states = set()

    # Convert state to a hashable format and check for symmetry
    symmetries = SymmetryManager.generate_all_symmetries(state.board)
    if any(symmetry in visited_states for symmetry in symmetries):
        return None, 0  # Skip symmetrical states
    visited_states.update(symmetries)

    if depth == 0 or is_terminal_fn(state):
        return None, evaluate_fn(state, state.current_player)  # Pass the player explicitly

    if maximizing_player:
        best_score = -math.inf
        best_move = None
        for move in get_legal_moves_fn(state, maximizing_player):
            new_state = apply_move_fn(state, move)
            _, score = minimax(new_state, depth - 1, False, evaluate_fn, get_legal_moves_fn, apply_move_fn, is_terminal_fn, visited_states)
            if score > best_score:
                best_score = score
                best_move = move
        return best_move, best_score
    else:
        best_score = math.inf
        best_move = None
        for move in get_legal_moves_fn(state, maximizing_player):
            new_state = apply_move_fn(state, move)
            _, score = minimax(new_state, depth - 1, True, evaluate_fn, get_legal_moves_fn, apply_move_fn, is_terminal_fn, visited_states)
            if score < best_score:
                best_score = score
                best_move = move
        return best_move, best_score


def alpha_beta_pruning(state, depth, alpha, beta, maximizing_player, evaluate_fn, get_legal_moves_fn, apply_move_fn, is_terminal_fn, visited_states=None):
    """
    Perform minimax with alpha-beta pruning and symmetry checks.
    """
    if visited_states is None:
        visited_states = set()

    # Convert state to a hashable format and check for symmetry
    symmetries = SymmetryManager.generate_all_symmetries(state.board)
    if any(symmetry in visited_states for symmetry in symmetries):
        return None, 0  # Skip symmetrical states
    visited_states.update(symmetries)

    if depth == 0 or is_terminal_fn(state):
        return None, evaluate_fn(state, state.current_player)  # Pass the player explicitly

    best_move = None
    if maximizing_player:
        max_eval = -math.inf
        for move in get_legal_moves_fn(state):
            new_state = apply_move_fn(state, move)
            _, eval = alpha_beta_pruning(new_state, depth - 1, alpha, beta, False, evaluate_fn, get_legal_moves_fn, apply_move_fn, is_terminal_fn, visited_states)
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # Prune
        return best_move, max_eval
    else:
        min_eval = math.inf
        for move in get_legal_moves_fn(state):
            new_state = apply_move_fn(state, move)
            _, eval = alpha_beta_pruning(new_state, depth - 1, alpha, beta, True, evaluate_fn, get_legal_moves_fn, apply_move_fn, is_terminal_fn, visited_states)
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break  # Prune
        return best_move, min_eval


def evaluate_fn(state, player):
    """
    A heuristic evaluation function for the game state.
    """
    current_player_moves = len(state.get_legal_moves(player))
    opponent_moves = len(state.get_legal_moves(3 - player))
    return current_player_moves - opponent_moves


def is_terminal_fn(state):
    """
    Checks if the game state is terminal.
    """
    return not state.get_legal_moves(state.current_player)


def apply_move_fn(state, move):
    """
    Apply a move to the game state and return the new state.
    """
    new_state = state.copy()
    new_state.apply_move(move)
    return new_state