import functools
import math
import tkinter as tk
from tkinter import Entry, filedialog

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageTk


class ImageUploaderGUI:
    def __init__(self, master, processor, b2, b4):
        self.master = master
        self.image_processor = processor
        self.model_b2 = b2
        self.model_b4 = b4

        # Create buttons
        self.button1 = tk.Button(master, text="Upload Image", command=self.upload_image)
        self.button1.grid(row=0, column=0, padx=10, pady=10)

        # "Segment ver. 1" and "Segment ver. 2" buttons
        self.button_segment1 = tk.Button(
            master,
            text="Segment ver. 1",
            command=functools.partial(self.segment_image, "b2"),
        )
        # self.button_segment1 = tk.Button(
        #     master, text="Segment ver. 1", command=self.upload_mask
        # )
        self.button_segment1.grid(row=0, column=1, padx=0, pady=10)

        self.button_segment2 = tk.Button(
            master,
            text="Segment ver. 2",
            command=functools.partial(self.segment_image, "b4"),
        )
        # self.button_segment2 = tk.Button(
        #     master, text="Segment ver. 2", command=self.upload_mask
        # )
        self.button_segment2.grid(row=0, column=2, padx=(0, 20), pady=10)

        # Create canvas for image and mask
        self.canvas = tk.Canvas(master, width=256, height=256)
        self.canvas.grid(row=1, column=0, padx=10, pady=10)

        self.image2_label = tk.Label(master)
        self.image2_label.grid(row=1, column=1, padx=10, pady=10)

        # Create button, entry form, and horizontal scale
        self.draw_button = tk.Button(
            master, text="Draw", command=self.enter_drawing_mode, width=15
        )
        self.draw_button.grid(row=2, column=0, columnspan=2, padx=10, pady=10)
        # Create Auto Draw button
        self.auto_draw_button = tk.Button(
            master, text="Auto Draw", command=self.auto_draw, width=15
        )
        self.auto_draw_button.grid(row=2, column=1, padx=10, pady=10)

        self.scale_actual_width_label = tk.Label(
            master, text="Enter Scale Actual Thickness:"
        )
        self.scale_actual_width_label.grid(row=3, column=0, padx=10, pady=5, sticky="e")

        self.scale_actual_width_entry = Entry(master)
        self.scale_actual_width_entry.grid(row=3, column=1, padx=10, pady=5, sticky="w")
        self.scale_actual_width_entry.insert(0, "19")

        # Initialize instance variables
        self.image_cv2 = None
        self.image1_tk = None
        self.image2_tk = None
        self.start_pos = None
        self.drawing = False
        self.line_length_label = tk.Label(master)
        self.line_length_label.grid(row=4, column=0, columnspan=2)

        # Calculated ratio of pixels to length
        self.ratio = None

        # Bind the mouse button click event to the on_mouse_click function
        self.canvas.bind("<Button-1>", self.on_mouse_click)

    def upload_image(self):
        # get filename
        filename = filedialog.askopenfilename(title="Select Image")
        # open and resize image
        img = Image.open(filename)
        img = img.resize((256, 256), Image.LANCZOS)
        # convert image to opencv format
        self.image_cv2 = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        # update image label
        self.image1_tk = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, image=self.image1_tk, anchor="nw")

    def upload_mask(self):
        # get filename
        filename = filedialog.askopenfilename(title="Select Mask")
        # open and resize image
        img = Image.open(filename)
        img = img.resize((256, 256), Image.LANCZOS)
        # convert image to opencv format
        self.mask_cv2 = np.asarray(img)
        # display image with colors
        min_value, max_value = self.mask_cv2.min(), self.mask_cv2.max()
        norm_mask = (self.mask_cv2 - min_value) / (max_value - min_value)
        norm_mask = (norm_mask * 255).astype("uint8")
        # update image label
        self.image2 = Image.fromarray(norm_mask)
        self.image2_tk = ImageTk.PhotoImage(self.image2)
        self.image2_label.config(image=self.image2_tk)

    def segment_image(self, model_type: str):
        if self.image_cv2 is None:
            return
        image_rgb = cv2.cvtColor(self.image_cv2, cv2.COLOR_BGR2RGB)
        inputs = self.image_processor(images=image_rgb, return_tensors="pt")
        with torch.no_grad():
            if model_type == "b2":
                outputs = self.model_b2(**inputs)
            else:
                outputs = self.model_b4(**inputs)
        logits = outputs.logits
        # Rescale logits to original image size
        upsampled_logits = F.interpolate(
            logits,
            size=(256, 256),  # (height, width)
            mode="bilinear",
            align_corners=False,
        )
        # Apply argmax on the class dimension
        pred_seg = upsampled_logits.argmax(dim=1)[0]
        # convert image to opencv format
        self.mask_cv2 = pred_seg.numpy().astype("uint8")
        # display image with colors
        min_value, max_value = self.mask_cv2.min(), self.mask_cv2.max()
        norm_mask = (self.mask_cv2 - min_value) / (max_value - min_value)
        norm_mask = (norm_mask * 255).astype("uint8")
        # update image label
        self.image2 = Image.fromarray(norm_mask)
        self.image2_tk = ImageTk.PhotoImage(self.image2)
        self.image2_label.config(image=self.image2_tk)

    def get_notebook_width(self):
        """Get the width of the notebook in pixels"""
        if not isinstance(self.mask_cv2, np.ndarray):
            return
        background_mask = cv2.threshold(self.mask_cv2, 0, 255, cv2.THRESH_BINARY_INV)[1]
        background_mask = cv2.morphologyEx(
            background_mask, cv2.MORPH_OPEN, np.ones((3, 9), np.uint8)
        )
        background_mask = cv2.morphologyEx(
            background_mask, cv2.MORPH_OPEN, np.ones((9, 3), np.uint8)
        )
        # get left and right edges of the notebook
        contours, _ = cv2.findContours(
            background_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            pt1 = (x, y + h // 2)
            pt2 = (x + w, y + h // 2)
            return pt1, pt2
        else:
            return None, None

    def auto_draw(self):
        pt1, pt2 = self.get_notebook_width()
        if pt1 is not None and pt2 is not None:
            self.draw_temporary_shapes(pt1, pt2)
            # Calculate the distance between the two points
            distance_pixels = math.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)
            # Get the actual width from the scale_actual_width_entry
            try:
                actual_width_cm = float(self.scale_actual_width_entry.get())
            except ValueError:
                # If the entry is not a valid float, you can set a default value or alert the user
                actual_width_cm = 1.0  # Default value
            # Calculate the ratio
            self.ratio = distance_pixels / actual_width_cm
            # Update the line_length_label
            self.line_length_label.config(text=f"Ratio: {self.ratio:.2f} pixels / cm")

    def draw_temporary_shapes(self, pt1, pt2):
        self.circle1_id = self.canvas.create_oval(
            pt1[0] - 2, pt1[1] - 2, pt1[0] + 2, pt1[1] + 2, outline="red", width=2
        )
        self.circle2_id = self.canvas.create_oval(
            pt2[0] - 2, pt2[1] - 2, pt2[0] + 2, pt2[1] + 2, outline="red", width=2
        )
        self.line_id = self.canvas.create_line(
            pt1[0] + 2, pt1[1], pt2[0] - 2, pt2[1], fill="green"
        )

    def remove_temporary_shapes(self):
        self.canvas.delete(self.circle1_id)
        self.canvas.delete(self.circle2_id)
        self.canvas.delete(self.line_id)

    def on_mouse_click(self, event):
        # If in drawing mode and the start position is None,
        # then this is the first click so save the start position
        if self.drawing and self.start_pos is None:
            self.start_pos = (event.x, event.y)
            # Draw a red dot at the start position with the "dot" tag
            self.canvas.create_oval(
                event.x - 3,
                event.y - 3,
                event.x + 3,
                event.y + 3,
                fill="red",
                tags="dot",
            )
        # If in drawing mode and the start position is already set,
        # then this is the second click, so draw the line and calculate its length
        elif self.drawing and self.start_pos is not None:
            # Add the line with the "line" tag
            self.canvas.create_line(
                self.start_pos[0],
                self.start_pos[1],
                event.x,
                event.y,
                fill="red",
                tags="line",
            )
            # Calculate and display line length
            line_length = math.sqrt(
                (self.start_pos[0] - event.x) ** 2 + (self.start_pos[1] - event.y) ** 2
            )
            scale_actual_width = float(self.scale_actual_width_entry.get())
            self.ratio = (
                line_length / scale_actual_width
                if scale_actual_width != 0
                else "Scale Actual Width can't be zero."
            )
            # self.line_length_label.config(text=f"Line length: {line_length:.2f}, Ratio: {ratio} pixels / cm")
            self.line_length_label.config(text=f"Ratio: {self.ratio:.2f} pixels / cm")
            # Reset the start position and exit drawing mode
            self.start_pos = None
            self.drawing = False

    # Function to enter drawing mode and clear existing lines
    def enter_drawing_mode(self):
        self.remove_temporary_shapes()
        # Clear all existing lines and dots
        self.canvas.delete("line")
        self.canvas.delete("dot")
        self.line_length_label.config(text="")
        self.drawing = True


if __name__ == "__main__":
    root = tk.Tk()
    uploader = ImageUploaderGUI(root)
    root.mainloop()
