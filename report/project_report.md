
## PoshakPix - Fashion meets pixels
## Garment Segmentation, Background Generation and Visual Consistency Analysis

**Project ID:** HSR-AIML-055

**Course:** Digital Image Processing

**Industry Reference:** Neuropixel.ai

---

# 1. Abstract

FashionStudio AI is a computer vision and Digital Image Processing application designed to improve fashion product photographs for e-commerce applications.

The system extracts garments from their original backgrounds using classical GrabCut segmentation and pretrained U2Net segmentation. The resulting binary masks are refined using morphological operations, including erosion, dilation, opening and closing.

The extracted garment is placed on a selected studio-style background through alpha compositing. The application evaluates garment preservation using Structural Similarity Index (SSIM), Mean Absolute Error (MAE), colour histogram similarity and CIE Lab colour difference.

The project demonstrates the integration of classical image processing techniques and pretrained computer vision models in a practical fashion imaging workflow.

# 2. Introduction

Product photographs play an important role in online fashion retail. Background consistency, garment visibility and image quality influence the presentation of clothing products.

Traditional manual image editing can require considerable time and effort.

FashionStudio AI addresses this problem through an automated image processing pipeline.

The system performs image preprocessing, foreground segmentation, mask refinement, background replacement and visual consistency evaluation.

# 3. Problem Statement

Fashion product images frequently contain unwanted backgrounds, inconsistent lighting and unnecessary visual elements.

Extracting garments without damaging their boundaries and original colours is an important image processing challenge.

The project aims to develop an application that automates garment extraction and background replacement while measuring visual preservation.

# 4. Objectives

1. Develop an image preprocessing pipeline.
2. Implement classical GrabCut segmentation.
3. Integrate pretrained U2Net segmentation.
4. Apply morphological operations to refine masks.
5. Generate multiple studio-style backgrounds.
6. Preserve garment appearance during background replacement.
7. Evaluate processing quality using quantitative metrics.
8. Compare classical and pretrained segmentation methods.

# 5. Existing Approaches

Traditional image editing tools require manual selection and background removal.

Classical segmentation approaches such as GrabCut use colour distributions and foreground/background models.

Deep learning segmentation methods use pretrained neural networks to identify foreground objects.

This project compares a classical method with a pretrained segmentation approach.

# 6. Proposed Methodology

The proposed system consists of the following stages:

1. Image input.
2. Image preprocessing.
3. Garment segmentation.
4. Binary mask generation.
5. Morphological mask refinement.
6. Background generation.
7. Alpha compositing.
8. Consistency evaluation.
9. Experimental comparison.

# 7. Image Preprocessing

The uploaded image is converted into RGB representation and resized while maintaining aspect ratio.

Gaussian filtering reduces high-frequency noise.

The Gaussian filter is represented as:

G(x,y) = (1 / 2πσ²) exp(-(x²+y²)/(2σ²))

The resulting image is normalized to the intensity range [0,255].

# 8. Garment Segmentation

## 8.1 GrabCut

GrabCut is a classical foreground segmentation technique based on graph cuts and Gaussian Mixture Models.

The image is initialized using a rectangular foreground region.

The algorithm estimates foreground and background distributions and iteratively refines segmentation.

## 8.2 Pretrained U2Net

U2Net is a salient object detection architecture.

The pretrained U2Net model is accessed through the rembg library.

The model generates an alpha mask that identifies foreground garment regions.

## 8.3 Comparison

Both approaches are compared based on:

- Foreground extraction.
- Mask quality.
- Processing time.
- Behaviour on complex boundaries.

# 9. Morphological Operations

Morphological operations improve binary segmentation masks.

## 9.1 Erosion

Erosion reduces foreground regions and can remove small foreground noise.

## 9.2 Dilation

Dilation expands foreground regions and can reconnect small gaps.

## 9.3 Opening

Opening performs erosion followed by dilation and removes small foreground objects.

## 9.4 Closing

Closing performs dilation followed by erosion and fills small gaps or holes.

## 9.5 Contour Detection

Contours identify the external boundaries of segmented garment regions.

# 10. Background Generation

The system supports solid colour and gradient backgrounds.

The extracted garment is combined with the selected background through alpha compositing.

The compositing equation is:

I_final = αI_foreground + (1-α)I_background

where α is the normalized segmentation mask.

A Gaussian-blurred shadow can optionally be generated from the garment mask.

# 11. Garment Consistency Analysis

Consistency analysis determines whether garment appearance is preserved after background replacement.

## 11.1 Structural Similarity Index

SSIM measures structural similarity between aligned image regions.

## 11.2 Mean Absolute Error

MAE measures average pixel-level differences.

MAE = (1/N) Σ |I_original - I_processed|

## 11.3 Histogram Similarity

Colour histograms are compared using the Bhattacharyya distance.

## 11.4 CIE Delta E 76

Colour differences are measured in CIE Lab space.

ΔE76 = sqrt((L1-L2)² + (a1-a2)² + (b1-b2)²)

Lower values indicate smaller colour differences.

## 11.5 CLIP Similarity

Optional CLIP image embeddings are compared using cosine similarity.

CLIP measures semantic similarity rather than exact pixel-level identity.

# 12. Experimental Setup

Experiments are conducted using fashion product photographs.

The implementation uses Python, OpenCV, NumPy, rembg, scikit-image and Streamlit.

The application is designed for CPU execution.

Dataset details:

- Dataset name: [Insert actual dataset]
- Number of images: [Insert actual count]
- Image formats: JPG / JPEG / PNG
- Image resolution: [Insert actual resolution range]

# 13. Experimental Results

Insert the actual results generated after running the experiments.

| Method | Runtime | IoU | Dice |
|---|---|---|---|
| GrabCut | [Actual] | [Actual/NA] | [Actual/NA] |
| rembg | [Actual] | [Actual/NA] | [Actual/NA] |
| rembg + Morphology | [Actual] | [Actual/NA] | [Actual/NA] |

For IoU and Dice, ground-truth masks are required.

Consistency results:

| Metric | Actual Result |
|---|---|
| SSIM | [Insert result] |
| MAE | [Insert result] |
| Histogram Similarity | [Insert result] |
| Delta E 76 | [Insert result] |

# 14. Failure Analysis

Potential failure cases include:

1. White garments on white backgrounds.
2. Complex garment boundaries.
3. Transparent fabrics.
4. Similar foreground and background colours.
5. Incorrect segmentation around sleeves and collars.

The performance of segmentation methods depends on image conditions.

Morphological operations can reduce small noise and fill small gaps but may also remove fine garment details if kernel sizes are too large.

# 15. Limitations

- Segmentation may fail on difficult images.
- GrabCut requires suitable foreground initialization.
- Morphological operations may distort fine boundaries.
- Background replacement does not generate new garment designs.
- CLIP similarity does not prove exact garment identity.
- IoU and Dice cannot be measured without ground-truth masks.

# 16. Future Scope

- Integration of diffusion-based background generation.
- More advanced garment segmentation models.
- Virtual try-on systems.
- Garment attribute recognition.
- Automatic defect detection.
- Larger-scale fashion image evaluation.

# 17. Conclusion

FashionStudio AI demonstrates the application of classical Digital Image Processing and pretrained computer vision models to fashion product image enhancement.

The implementation combines segmentation, morphological operations, background replacement and visual consistency analysis.

The project provides a reproducible framework for studying how classical image processing and AI-based methods can work together in an e-commerce imaging application.

# 18. References

1. Rother, C., Kolmogorov, V. and Blake, A. GrabCut: Interactive Foreground Extraction using Iterated Graph Cuts. ACM Transactions on Graphics, 2004.
2. Qin, X. et al. U2-Net: Going Deeper with Nested U-Structure for Salient Object Detection. Pattern Recognition, 2020.
3. Wang, Z. et al. Image Quality Assessment: From Error Visibility to Structural Similarity. IEEE Transactions on Image Processing, 2004.
4. OpenCV Documentation: https://docs.opencv.org/
5. scikit-image Documentation: https://scikit-image.org/
6. rembg Documentation: https://github.com/danielgatis/rembg
7. Radford, A. et al. Learning Transferable Visual Models From Natural Language Supervision. ICML, 2021.
