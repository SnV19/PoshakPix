
# FashionStudio AI

## Garment Segmentation, Background Generation and Visual Consistency Analysis

**Project ID:** HSR-AIML-055

**Domain:** Digital Image Processing and Computer Vision

**Industry Reference:** Neuropixel.ai

## 1. Overview

FashionStudio AI is a fashion product image processing application developed using Python, OpenCV and Streamlit.

The application extracts garments from fashion product images, refines segmentation masks through morphological operations, replaces backgrounds and evaluates garment consistency.

## 2. Features

- Fashion image upload.
- Image resizing and Gaussian denoising.
- Classical GrabCut segmentation.
- Pretrained U2Net segmentation through rembg.
- Morphological mask refinement.
- Background replacement.
- Alpha compositing.
- Shadow generation.
- SSIM evaluation.
- MAE evaluation.
- Colour histogram similarity.
- CIE Delta E colour difference.
- Segmentation runtime comparison.
- Optional CLIP semantic similarity.

## 3. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| OpenCV | Image processing |
| NumPy | Numerical operations |
| Pillow | Image handling |
| rembg | Pretrained segmentation |
| Streamlit | Web application |
| scikit-image | SSIM |
| Pandas | Experimental data |
| Matplotlib | Visualization |

## 4. Installation

Open PowerShell inside the project directory.

Create a virtual environment:

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Run the Application

```powershell
streamlit run app.py
```

The application will open in a browser.

## 6. Dataset Preparation

Place fashion product images inside:

```text
data/sample_images/
```

Supported formats:

- JPG
- JPEG
- PNG

For segmentation accuracy experiments, place corresponding binary ground-truth masks inside:

```text
data/ground_truth/
```

Use the same filename as the original image, with a PNG extension.

Example:

```text
data/
├── sample_images/
│   ├── shirt.jpg
│   └── dress.jpg
│
└── ground_truth/
    ├── shirt.png
    └── dress.png
```

Ground-truth masks should use white for garment pixels and black for background pixels.

## 7. Run Experiments

```powershell
python experiments/comparison.py
```

Results are generated inside:

```text
outputs/
```

Generated files include:

- experiment_results.csv
- runtime_comparison.png
- foreground_comparison.png
- accuracy_comparison.png (when ground-truth masks exist)

## 8. Evaluation Metrics

### SSIM

Measures structural similarity between aligned garment regions.

### MAE

Measures average absolute pixel difference.

### Histogram Similarity

Measures similarity between colour distributions.

### Delta E 76

Measures colour difference in CIE Lab space.

### IoU

Measures overlap between predicted and ground-truth segmentation masks.

### Dice Coefficient

Measures segmentation overlap using the Dice formula.

### Runtime

Measures computational execution time.

## 9. Project Structure

```text
FashionStudio-AI/
├── app.py
├── requirements.txt
├── README.md
├── modules/
├── utils/
├── experiments/
├── data/
├── outputs/
├── report/
├── docs/
└── screenshots/
```

## 10. Limitations

- Segmentation quality depends on input image complexity.
- GrabCut may fail when foreground and background colours are similar.
- Pretrained models may occasionally produce inaccurate boundaries.
- Background replacement does not generate new garment designs.
- CLIP similarity is semantic and does not guarantee exact garment identity.
- IoU and Dice require ground-truth masks.

## 11. Future Scope

- More advanced segmentation models.
- AI-based background generation.
- Garment defect detection.
- Virtual try-on integration.
- Large-scale fashion product evaluation.

## 12. Conclusion

FashionStudio AI demonstrates how classical Digital Image Processing techniques and pretrained computer vision models can be integrated into a practical fashion image processing application.
