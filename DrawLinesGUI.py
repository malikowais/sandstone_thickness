import tkinter as tk

import cv2
from PIL import Image, ImageTk

from draw_widths import draw_widths
from ImageUploaderGUI2 import ImageUploaderGUI


class DrawLinesGUI(tk.Frame):
    def __init__(self, master: tk.Frame, image_uploader: ImageUploaderGUI):
        super().__init__(master)
        self.master = master
        self.pack(side="right", fill="both", expand=True)

        self.image_uploader = image_uploader

        # Create entry forms and labels
        self.min_label = tk.Label(self, text="Min rock thickness:")
        self.min_label.grid(row=0, column=0, padx=10, pady=5, sticky="e")
        self.min_entry = tk.Entry(self)
        self.min_entry.insert(0, "0")
        self.min_entry.grid(row=0, column=1, padx=10, pady=5)

        self.max_label = tk.Label(self, text="Max rock thickness:")
        self.max_label.grid(row=1, column=0, padx=10, pady=5, sticky="e")
        self.max_entry = tk.Entry(self)
        self.max_entry.insert(0, "100")
        self.max_entry.grid(row=1, column=1, padx=10, pady=5)

        self.diff_label = tk.Label(self, text="Minimum difference between thicknesses:")
        self.diff_label.grid(row=2, column=0, padx=10, pady=5, sticky="e")
        self.diff_entry = tk.Entry(self)
        self.diff_entry.insert(0, "5")
        self.diff_entry.grid(row=2, column=1, padx=10, pady=5)

        self.vgd_label = tk.Label(self, text="Distance between each layer:")
        self.vgd_label.grid(row=3, column=0, padx=10, pady=5, sticky="e")
        self.vgd_entry = tk.Entry(self)
        self.vgd_entry.insert(0, "2")
        self.vgd_entry.grid(row=3, column=1, padx=10, pady=5)

        # Create draw lines button
        self.draw_button = tk.Button(
            self, text="Draw Lines", command=self.draw_lines, width=15
        )
        self.draw_button.grid(row=4, column=0, columnspan=2, padx=10, pady=10)

        # Create sample image
        self.sample_image = tk.Label(self)
        self.sample_image.grid(row=5, column=0, columnspan=2, padx=10, pady=5)

        # Create list to store generated images
        self.images = []
        # Index of currently displayed image
        self.curr_img_index = 0

        # Create back and next buttons
        self.back_button = tk.Button(
            self, text="Back", command=self.show_previous_image, width=10
        )
        self.next_button = tk.Button(
            self, text="Next", command=self.show_next_image, width=10
        )
        self.back_button.grid(row=6, column=0, pady=5)
        self.next_button.grid(row=6, column=1, pady=5)

    def draw_lines(self):
        self.images.clear()
        # perform task using previously uploaded images and calculated ratio
        # create new image and display it using self.sample_image.config(image=new_image)
        image = self.image_uploader.image_cv2
        mask = self.image_uploader.mask_cv2
        ratio = self.image_uploader.ratio

        annotated_images = draw_widths(
            image,
            mask,
            ratio=ratio,
            w_min=float(self.min_entry.get()),
            w_max=float(self.max_entry.get()),
            line_sim_thresh=float(self.diff_entry.get()),
            group_y_range=float(self.vgd_entry.get()),
        )[0]

        for img in annotated_images:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("uint8")
            img = Image.fromarray(img)
            photo = ImageTk.PhotoImage(img)
            self.images.append(photo)

        if len(self.images) > 0:
            # display first image
            self.sample_image.config(image=self.images[0])
            # show back and next buttons
            self.back_button.grid(row=11, column=0, pady=10)
            self.next_button.grid(row=11, column=1, pady=10)

    def show_previous_image(self):
        self.curr_img_index = (self.curr_img_index - 1) % len(self.images)
        self.sample_image.config(image=self.images[self.curr_img_index])

    def show_next_image(self):
        self.curr_img_index = (self.curr_img_index + 1) % len(self.images)
        self.sample_image.config(image=self.images[self.curr_img_index])
