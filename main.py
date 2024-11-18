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
            move_input = input("Enter your move (e.g., '1 2 E 4 3 1 1'): ")
            move = parse_move_input(move_input)
            if validate_move(game_state.board, move):
                return move
            else:
                print("Invalid move. Please try again.")
        except ValueError as e:
            print(f"Error: {e}. Try again.")


def computer_move(game_state, depth=3):
    """
    Determine the computer's move using minimax with alpha-beta pruning.
    """
    print("Computer is thinking...")
    move, _ = alpha_beta_pruning(
        state=game_state,
        depth=depth,
        alpha=-math.inf,
        beta=math.inf,
        maximizing_player=(game_state.current_player == 1),
        evaluate_fn=evaluate_board,
        get_legal_moves_fn=lambda state, _: state.get_legal_moves(state.current_player),
        apply_move_fn=lambda state, move: state.copy().apply_move(move),
        is_terminal_fn=lambda state: state.is_terminal(),
    )
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

    draw_all_l_configurations()


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


if __name__ == "__main__":
    main()