# **L-Game**

An interactive implementation of Edward de Bono's **L-Game**. This project supports human vs. human, human vs. computer, and computer vs. computer gameplay modes. The game includes a GUI for visualizing the board state and allows the computer to play intelligently using **minimax with alpha-beta pruning**.

---

## **Game Rules**

- The board is a 4x4 grid.
- Two players each control an **L-shaped piece**.
- Two neutral pieces occupy the corners of the board at the start.
- Players take turns performing the following actions:
  1. Move their **L-piece** to a new valid position.
  2. Optionally move one neutral piece to a free spot.
- The game ends when a player can no longer move their **L-piece**.

---

## **Features**

- **GUI Visualization:** Displays the board state dynamically using Tkinter.
- **Gameplay Modes:**
  - Human vs Human
  - Human vs Computer
  - Computer vs Computer
- **AI with Alpha-Beta Pruning:** The computer uses an efficient search algorithm to determine its moves.
- **Real-Time Updates:** The GUI updates dynamically after each move.
- **8 Configurations Visualization:** Shows all possible configurations of the L-piece for reference.

---

## **Requirements**

- **Python 3.8+**
- **Tkinter** (comes pre-installed with Python on most systems)

---

## **How to Run the Game**

Clone the repository, navigate into it, and run the game:

```bash
git clone <https://github.com/alextitov19/cse175-TheLGameLab>
cd <rcse175-TheLGameLab>
python3 main.py
```
