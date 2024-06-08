from tkinter import *
import tkinter as tk
import torch

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.x=0
        self.y=0
        self.button_held = False

        self.geometry("500x500")
        self.canvas = tk.Canvas(self, bg='gray10')
        self.text = tk.Label(self, text="Guess: N/A", font=('Helvetica bold', 24), anchor="n").pack(pady=5)

        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonPress-1>", self.press)
        self.canvas.bind("<ButtonRelease-1>", self.release)
        self.canvas.bind("<ButtonPress-3>", self.clear)

        self.canvas.pack(anchor='nw', fill='both', expand=1)

    def press(self, event):
        self.button_held = True

    def release(self, event):
        self.button_held = False

    def clear(self, event):
        self.canvas.delete("all")

    def draw(self, event):
        if self.button_held:
            size=5
            self.x = event.x
            self.y = event.y
            self.canvas.create_oval(self.x-size, self.y-size, self.x + size, self.y + size, fill="white", outline="white")

    #def classify(self)

def start_drawing_app():
    app = App()
    mainloop()

start_drawing_app()