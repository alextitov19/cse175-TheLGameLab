# main.py

from game_state import GameState
from utils import parse_move_input, validate_move, print_board, draw_all_l_configurations
from minimax import alpha_beta_pruning
from heuristic import evaluate_board
import math


def human_move(game_state):
    """
    Prompt the human player to make a move.
    """
    print_board(game_state.board)
    while True:
        try:
            draw_all_l_configurations()
            move_input = input("Enter your move (e.g., '1 2 E 4 3 1 1'): ")
            move = parse_move_input(move_input)
            if validate_move(game_state.board, move, game_state.current_player):
                return move
            else:
                print("Invalid move. Please try again.")
        except ValueError as e:
            print(f"Error: {e}. Try again.")


def computer_move(game_state, depth=1):
    """
    Determine the computer's move using minimax with alpha-beta pruning.
    """
    print("Computer is thinking...")
    print("Terminal state:", game_state.is_terminal())
    move, _ = alpha_beta_pruning(
        state=game_state,
        depth=depth,
        alpha=-math.inf,
        beta=math.inf,
        maximizing_player=(game_state.current_player == 1),
        evaluate_fn=evaluate_board,
        get_legal_moves_fn=lambda state: state.get_legal_moves(state.current_player),
        apply_move_fn=lambda state, move: debug_apply_move(state, move),
        is_terminal_fn=lambda state: state.is_terminal(),
    )
    print(f"Computer's Move: {move}")
    return move


def main():
    """
    Main function to run the L-game.
    """
    print("Welcome to the L-game!")
    print("You can choose one of the following modes:")
    print("1. Human vs Human")
    print("2. Human vs Computer")
    print("3. Computer vs Computer")


    mode = input("Enter the mode number (1, 2, or 3): ")
    if mode not in {"1", "2", "3"}:
        print("Invalid mode. Exiting.")
        return

    game_state = GameState()

    while not game_state.is_terminal():
        print(f"\nPlayer {game_state.current_player}'s turn.")
        if (mode == "1") or (mode == "2" and game_state.current_player == 1):
            # Human move
            move = human_move(game_state)
        else:
            # Computer move
            move = computer_move(game_state)

        # Apply the move
        game_state.apply_move(move)

    print(f"Game Over! Player {3 - game_state.current_player} wins!")

def debug_apply_move(state, move):
    """
    Debug wrapper for applying moves to a game state.
    """
    print(f"Applying move: {move}")
    new_state = state.copy()
    if new_state is None:
        print("Error: state.copy() returned None.")
        return None
    try:
        new_state.apply_move(move)
        return new_state
    except Exception as e:
        print(f"Error during apply_move: {e}")
        return None
    
if __name__ == "__main__":
    main()