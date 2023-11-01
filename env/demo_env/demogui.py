# -*- coding: utf-8 -*-
# @Author: Meleko
# @Date:   2023-10-20 05:13:50
# @Last Modified by:   Melkor
# @Last Modified time: 2023-10-25 02:21:05

import numpy as np
from karanir import *

import tkinter as tk
from PIL import ImageTk, Image


if __name__ == "__main__":

	main_window = tk.Tk()
	main_window.title("Main Window")

	button = tk.Button(main_window,text="stop",width=12,command=main_window.destroy)
	button.pack()

	scale = 4
	center_canvas = tk.Canvas(main_window, width = 128 * scale, height = 128 * scale)
	center_canvas.pack()

	center_canvas.create_line(0,0,128,128)

	background_image = ImageTk.PhotoImage(
		Image.open("/Users/melkor/Desktop/bg.png").resize([128 * scale * 2,128 * scale * 2]))

	center_canvas.create_image(0, 0, image = background_image)

	label = tk.Label(main_window, text = "e1")
	label.pack()
	e1 = tk.Entry(main_window)
	#e1.grid(row = 1, column = 1)
	e1.pack()

	main_window.mainloop()