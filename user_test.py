from tkinter import *
import torch

global last_x, last_y
app = Tk()
app.geometry("500x500")
canvas = Canvas(app, bg='gray10')
text = Label(app, text="Guess: N/A", font=('Helvetica bold',24), anchor="n").pack(pady=5)

def update_xy(event):
    global last_x, last_y
    last_x, last_y = event.x, event.y

def draw(event):
    global last_x, last_y
    canvas.create_line((last_x, last_y, event.x, event.y), fill='white', width=5)
    last_x, last_y = event.x, event.y

def clear(event):
    canvas.delete("all")

def start_drawing():

    canvas.pack(anchor='nw', fill='both', expand=1)
    canvas.bind("<ButtonPress-1>", update_xy)
    canvas.bind("<ButtonPress-3>", clear)
    canvas.bind("<B1-Motion>", draw)

    app.mainloop()

start_drawing()