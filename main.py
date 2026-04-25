import os
import threading
import tkinter as tk
from tkinter import ttk

import cv2
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from draw_widths import draw_widths
from DrawLinesGUI import DrawLinesGUI
from get_scale_ratio import get_notebook_width
from ImageUploaderGUI2 import ImageUploaderGUI
from util import resource_path

image_processor = SegformerImageProcessor(size={"width": 256, "height": 256})
id2label = {0: "background", 1: "sandstone", 2: "mudstone"}
label2id = {v: k for k, v in id2label.items()}
model_b2 = SegformerForSemanticSegmentation.from_pretrained(
    "assets/b2", id2label=id2label, label2id=label2id
).eval()
model_b4 = SegformerForSemanticSegmentation.from_pretrained(
    "assets/b4", id2label=id2label, label2id=label2id
).eval()


class MainWindow:
    def __init__(self, master):
        self.master = master
        master.title("Rock Width Identification")
        path = resource_path("assets")
        path = os.path.join(path, "rock.ico")
        master.iconbitmap(path)

        # Create styled frames
        style = ttk.Style()
        style.configure(
            "LeftFrame.TFrame", background="#F0F0F0", borderwidth=2, relief="raised"
        )
        style.configure(
            "RightFrame.TFrame", background="#F0F0F0", borderwidth=2, relief="raised"
        )

        # Create a new frame at the top of the master window
        top_frame = tk.Frame(master)
        top_frame.grid(row=0, column=0, columnspan=2, sticky="ew")

        # Entry box for input directory
        self.input_dir_entry = tk.Entry(top_frame, width=100)
        self.input_dir_entry.grid(row=0, column=0, padx=10, pady=5)

        # Button to open dialog box for input directory
        input_dir_button = tk.Button(
            top_frame,
            text="Choose Input Directory",
            command=self.open_input_directory_dialog,
        )
        input_dir_button.grid(row=0, column=1, padx=10, pady=5)

        # Entry box for output directory
        self.output_dir_entry = tk.Entry(top_frame, width=100)
        self.output_dir_entry.grid(row=1, column=0, padx=10, pady=5)

        # Button to open dialog box for input directory
        output_dir_button = tk.Button(
            top_frame,
            text="Choose Output Directory",
            command=self.open_output_directory_dialog,
        )
        output_dir_button.grid(row=1, column=1, padx=10, pady=5)

        # Button to start batch processing
        start_processing_button = tk.Button(
            top_frame, text="Start Processing", command=self.start_batch_process_thread
        )
        start_processing_button.grid(row=2, column=0, padx=10, pady=5)

        # Create a Progress Bar
        self.progress = ttk.Progressbar(
            top_frame, orient="horizontal", length=200, mode="determinate"
        )
        self.progress.grid(row=3, columnspan=3, pady=10)

        # Create a label for progress text
        self.progress_text = tk.Label(top_frame, text="0 / 0 (0%)")
        self.progress_text.grid(row=3, column=3, padx=10)

        left_frame = ttk.Frame(master, style="LeftFrame.TFrame", width=400, height=400)
        left_frame.grid(row=1, column=0, sticky="nsew")

        right_frame = ttk.Frame(
            master, style="RightFrame.TFrame", width=400, height=400
        )
        right_frame.grid(row=1, column=1, sticky="nsew")

        # place ImageUploaderGUI widget in left frame
        self.image_uploader = ImageUploaderGUI(
            left_frame, image_processor, model_b2, model_b4
        )
        self.lines_gui = DrawLinesGUI(right_frame, self.image_uploader)

    def open_input_directory_dialog(self):
        folder_selected = tk.filedialog.askdirectory()
        self.input_dir_entry.delete(0, tk.END)
        self.input_dir_entry.insert(0, folder_selected)

    def open_output_directory_dialog(self):
        folder_selected = tk.filedialog.askdirectory()
        self.output_dir_entry.delete(0, tk.END)
        self.output_dir_entry.insert(0, folder_selected)

    def start_batch_process_thread(self):
        threading.Thread(target=self.batch_process).start()

    def batch_process(self):
        input_dir = self.input_dir_entry.get()
        is_not_duplicate = lambda x: not x.endswith("(1).jpg")
        sort_by_number = lambda x: int(x.split("\\")[-1].split(".")[0])
        all_files = os.listdir(input_dir)
        all_files = list(filter(is_not_duplicate, all_files))
        all_files = sorted(all_files, key=sort_by_number)
        output_dir = self.output_dir_entry.get()

        if input_dir == "" or output_dir == "":
            return

        total_files = len(all_files)
        self.progress["maximum"] = total_files

        for i, file in enumerate(all_files):
            image = cv2.imread(os.path.join(input_dir, file))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            inputs = image_processor(images=image_rgb, return_tensors="pt")
            with torch.no_grad():
                outputs = model_b2(**inputs)
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
            mask = pred_seg.numpy().astype("uint8")
            pt1, pt2 = get_notebook_width(mask)
            if pt1 is None or pt2 is None:
                continue
            difference = pt2[0] - pt1[0]
            # fixed 19cm width of notebook
            ratio = difference / 19
            image_256 = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
            annotated_image = draw_widths(
                image_256,
                mask,
                ratio=ratio,
                w_min=1,
                w_max=80,
                line_sim_thresh=2,
                group_y_range=4,
            )[1]
            cv2.imwrite(os.path.join(output_dir, file), annotated_image)

            # Update the progress bar and text
            self.master.after(0, self.update_progress, i + 1, total_files)

    def update_progress(self, current, total):
        self.progress["value"] = current
        percent_complete = (current / total) * 100
        self.progress_text.config(text=f"{current} / {total} ({percent_complete:.2f}%)")
        self.master.update_idletasks()


if __name__ == "__main__":
    root = tk.Tk()
    app = MainWindow(root)
    root.mainloop()
