import tkinter as tk

class GameVisualizer:
    def __init__(self, board):
        self.board = board
        self.window = tk.Tk()
        self.window.title("L-Game")
        self.cell_size = 100  # Size of each cell
        self.canvas = tk.Canvas(
            self.window,
            width=self.cell_size * 4,
            height=self.cell_size * 4,
            bg="white"
        )
        self.canvas.pack()

    def draw_board(self):
        """Draw the current board on the canvas."""
        self.canvas.delete("all")  # Clear the canvas

        for y, row in enumerate(self.board):
            for x, cell in enumerate(row):
                x0 = x * self.cell_size
                y0 = y * self.cell_size
                x1 = x0 + self.cell_size
                y1 = y0 + self.cell_size

                # Draw cell background
                self.canvas.create_rectangle(x0, y0, x1, y1, fill="lightgray", outline="black")

                # Draw pieces
                if cell == 1:
                    self.canvas.create_text(
                        (x0 + x1) // 2,
                        (y0 + y1) // 2,
                        text="P1",
                        font=("Helvetica", 20),
                        fill="blue"
                    )
                elif cell == 2:
                    self.canvas.create_text(
                        (x0 + x1) // 2,
                        (y0 + y1) // 2,
                        text="P2",
                        font=("Helvetica", 20),
                        fill="red"
                    )
                elif cell == "N":
                    self.canvas.create_text(
                        (x0 + x1) // 2,
                        (y0 + y1) // 2,
                        text="N",
                        font=("Helvetica", 20),
                        fill="green"
                    )

    def update_board(self, board):
        """Update the board and redraw."""
        self.board = board
        self.draw_board()

    def run(self):
        """Run the Tkinter main loop."""
        self.draw_board()
        self.window.mainloop()