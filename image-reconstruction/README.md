# Image Reconstruction with Tikhonov and Wiener Deconvolution

## 🔬 Project Summary
This project explores image reconstruction as an inverse problem using classical signal processing methods.  
It implements and compares Tikhonov regularization and Wiener deconvolution, analyzing the trade-off
between numerical accuracy (PSNR) and perceptual quality (SSIM).

## Project Overview
This project implements a simple image reconstruction pipeline in Python for an inverse problem setting.

A clean grayscale image is degraded using:
- Gaussian blur
- Additive Gaussian noise

Then, two classical reconstruction methods are applied:
- Tikhonov regularization
- Wiener deconvolution

The reconstruction quality is evaluated using:
- PSNR
- SSIM

---

## Forward Model
The degradation model is:

y = Hx + n

where:
- x: original image  
- H: blur operator  
- n: additive noise  
- y: observed image  

---

## Methods

### 1. Tikhonov Regularization
The reconstruction is performed in the Fourier domain with derivative-based regularization to stabilize the inverse problem.

### 2. Wiener Deconvolution
Wiener filtering is used as a frequency-domain method that accounts for noise during reconstruction.

---

## Evaluation Metrics
- **PSNR**: measures pixel-wise similarity  
- **SSIM**: measures structural and perceptual similarity  

---

## Observations
- Increasing the regularization parameter improves stability
- Large regularization leads to oversmoothing
- PSNR does not always reflect perceptual quality
- SSIM provides better insight into visual quality

---

## Project Structure

project/
│
├── main.py
├── utils.py
├── results/
│ ├── observed.png
│ ├── tikhonov.png
│ ├── wiener.png
│ └── best_tikhonov.png


---

## Requirements

Install dependencies:

```bash
pip install numpy scipy matplotlib scikit-image


***Running the Project***

python main.py


Output

The project generates:

Observed (blurred + noisy) image
Tikhonov reconstruction
Wiener reconstruction
Best Tikhonov reconstruction
Lambda vs PSNR plot
Lambda vs SSIM plot