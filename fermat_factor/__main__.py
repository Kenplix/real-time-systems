import math
import tkinter as tk
from typing import List, Tuple

expansion: List[int] = []


def factorise(n: int) -> Tuple[int, int]:
    s = math.ceil(math.sqrt(n))
    y = s**2 - n
    while not math.sqrt(y).is_integer():
        s += 1
        y = s**2 - n
    return s + int(math.sqrt(y)), s - int(math.sqrt(y))


def full_factor(n: int) -> None:
    a, b = factorise(n)
    if b != 1:
        full_factor(a)
        full_factor(b)
    else:
        expansion.append(a)


class Application(tk.Frame):

    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.master.title("Fermat's factor")

        self.entered_number = 0

        self.factorise_text = tk.StringVar()
        self.factorise_text.set("")

        self.label_text = tk.StringVar()
        self.label_text.set("Factorization of :")
        self.create_widgets()

    def create_widgets(self):
        self.factorise_label = tk.Label(self.master, textvariable=self.factorise_text)

        self.label = tk.Label(self.master, textvariable=self.label_text)

        vcmd = self.master.register(self.validate)  # we have to wrap the command
        self.entry = tk.Entry(self.master, validate="key", validatecommand=(vcmd, "%P"))

        self.calculate_button = tk.Button(self.master, text="Factorise", command=lambda: self.update())

        # Layout
        self.label.grid(row=0, column=0, sticky=tk.W)
        self.factorise_label.grid(row=0, column=1, columnspan=2, sticky=tk.E)
        self.entry.grid(row=1, column=0, columnspan=3, sticky=tk.W + tk.E)
        self.calculate_button.grid(row=2, column=0, columnspan=3, sticky=tk.W + tk.E)

    def validate(self, new_text):
        if not new_text:  # the field is being cleared
            return True

        try:
            self.entered_number = int(new_text)
            return True
        except ValueError:
            return False

    def update(self):
        if self.entered_number % 2 != 0:
            full_factor(self.entered_number)
            result = " * ".join(map(lambda x: str(int(x)), expansion))
            expansion.clear()
        else:
            result = "Not allowed for even numbers"

        self.label_text.set(f"Factorization of {self.entered_number}:")
        self.factorise_text.set(result)
        self.entry.delete(0, tk.END)


if __name__ == "__main__":
    root = tk.Tk()
    my_gui = Application(master=root)
    root.mainloop()
