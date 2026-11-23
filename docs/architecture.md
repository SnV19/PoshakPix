
# FashionStudio AI – System Architecture

## Overall Pipeline

```mermaid
flowchart TD
    A[Input Fashion Image] --> B[Image Preprocessing]
    B --> C[Resizing and Gaussian Filtering]

    C --> D{Segmentation}

    D --> E[Classical GrabCut]
    D --> F[Pretrained rembg U2Net]

    E --> G[Binary Mask]
    F --> G

    G --> H[Thresholding]
    H --> I[Morphological Refinement]

    I --> J[Foreground Extraction]

    J --> K[Background Generation]
    K --> L[Alpha Compositing]

    L --> M[Final Fashion Image]

    M --> N[Consistency Evaluation]

    N --> O[SSIM]
    N --> P[Colour Similarity]
    N --> Q[MAE]
    N --> R[CLIP Similarity - Optional]

    O --> S[Evaluation Dashboard]
    P --> S
    Q --> S
    R --> S
```

## Classical DIP Components

1. Image resizing.
2. Gaussian filtering.
3. Binary thresholding.
4. Erosion.
5. Dilation.
6. Opening.
7. Closing.
8. Contour detection.
9. Alpha compositing.
10. Image inpainting.

## AI Components

1. Pretrained U2Net segmentation through rembg.
2. Optional CLIP feature extraction.

## Evaluation

1. SSIM.
2. MAE.
3. Histogram similarity.
4. CIE Delta E 76.
5. IoU and Dice when ground-truth masks are available.
6. Runtime comparison.
