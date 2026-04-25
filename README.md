
# 🪨 Sandstone Layer Thickness Measurement Pipeline

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![License](https://img.shields.io/badge/License-Academic-lightgrey)



## 📌 Overview

This project provides an **automated pipeline** for estimating **sandstone layer thickness** from digital outcrop images.

It integrates:

* 🤖 **Deep Learning (SegFormer - Vision Transformers)**
* 👁️ **Computer Vision techniques**
* 🖥️ **Interactive GUI for end-users**

👉 The goal is to move from **image segmentation → quantitative geological measurement**, enabling faster and more consistent analysis for geoscientists.



## ✨ Features

* ✔️ Semantic segmentation of sandstone and mudstone
* ✔️ Automated contour extraction and layer detection
* ✔️ Thickness estimation with real-world scaling
* ✔️ User-friendly desktop GUI (Tkinter)
* ✔️ Adjustable measurement parameters
* ✔️ Visualization of results on images



## 🧰 Requirements

### 💻 Hardware

* Minimum **8 GB RAM**
* GPU (optional, recommended for faster inference)

### 🧪 Software

* Python 3.8+
* PyTorch
* HuggingFace Transformers
* OpenCV
* NumPy
* Tkinter



## ⚙️ Installation

```bash
git clone https://github.com/malikowais/sandstone_thickness.git
cd sandstone_thickness
pip install -r requirements.txt
```



## 🚀 Usage

### ▶️ Run the Application

```bash
python main.py
```



### 🪜 Workflow

1. **Upload Image**

   * Use a digital outcrop image (sample images available as 2.jpg and 3.jpg)
   * (Recommended: rotate 90° counterclockwise) - sample image are already rotated

2. **Run Segmentation**

   * Choose model: `MiT-b2` or `MiT-b4`
   * View segmentation mask

3. **Set Scale**

   * Draw a line on an object with known length
   * Defines pixel → cm conversion

4. **Adjust Parameters (Optional)**

   * `Distance between layers` → controls sampling density
   * `Minimum difference between thickness` → removes redundant values

5. **Compute Thickness**

   * Click **Compute Thickness**
   * Outputs:

     * Thickness values (cm)
     * Visual measurement lines on image

**More Details** - Find more details in the RWI Tutorial with all GUIs explained and some results are shown.

## 📊 Output

* Annotated image with thickness measurements
* Numerical thickness values for detected layers



## ⚠️ Notes

* Accuracy depends on segmentation quality
* Ensure clear contrast between rock layers
* Correct scale calibration is critical
* Works best with well-exposed outcrop images



## 🧠 Method Summary

The pipeline includes:

* Transformer-based segmentation (**SegFormer**)
* Morphological processing
* Contour extraction and interpolation
* Layer grouping and filtering
* Pixel-to-physical measurement conversion



## 📁 Project Structure

```
sandstone_thickness/
│── models/            # Pre-trained models
│── data/              # Sample images
│── src/               # Core processing code
│── gui/               # Tkinter interface
│── main.py            # Entry point
│── requirements.txt
```


## 📬 Contact

**Owais A. Malik**
📧 [owais.malik@atu.ie](mailto:owais.malik@atu.ie)



## 📜 License

This project is intended for **research and academic use**.



## ⭐ Citation

If you use this work, please cite:

```
Malik, O.A., Puasa, I., Lai, D.T.C.
From Segmentation to Measurement: A Transformer-Based Pipeline for Sandstone Layer Thickness Estimation in Digital Outcrops.
```



## 🙌 Acknowledgements

This work was developed as part of ongoing research in **AI for geoscience applications** at Universiti Brunei Darussalam.


