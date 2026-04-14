# Image Reconstruction in Inverse Problems  
### Tikhonov Regularization vs. Wiener Deconvolution

## 🔬 Overview
This project studies image reconstruction as a classical inverse problem, where the goal is to recover an original image from degraded observations.

A clean grayscale image is degraded using Gaussian blur and additive Gaussian noise, and then reconstructed using two frequency-domain methods:
- Tikhonov regularization  
- Wiener deconvolution  

The focus of this project is not only implementation, but also **analysis of reconstruction behavior**, particularly the trade-off between numerical fidelity and perceptual image quality.

---

## 🎯 Motivation
Image reconstruction is a fundamental problem in signal processing, computer vision, and computational imaging.  
Inverse problems in imaging are typically **ill-posed**, requiring regularization and robust reconstruction methods.

This project explores how classical reconstruction methods behave under noise and blur, and highlights their limitations as baselines for more advanced approaches.

---

## ⚙️ Forward Model

The degradation process is modeled as:

```

y = Hx + n

```

Where:
- `x`: original image  
- `H`: blur operator (Gaussian PSF)  
- `n`: additive Gaussian noise  
- `y`: observed image  

---

## 🧠 Methods

### 1. Tikhonov Regularization
- Implemented in the Fourier domain  
- Uses derivative-based smoothing (∥∇x∥²)  
- Stabilizes inversion but may cause oversmoothing  

### 2. Wiener Deconvolution
- Frequency-domain filtering method  
- Incorporates noise-aware stabilization  
- More robust under noisy conditions  

---

## 📊 Evaluation Metrics

- **PSNR (Peak Signal-to-Noise Ratio)**  
  Measures pixel-wise reconstruction accuracy  

- **SSIM (Structural Similarity Index)**  
  Measures perceptual and structural similarity  

---

## 🔍 Key Results & Observations

- Both methods improve degraded images but fail to fully restore high-frequency details  
- Increasing regularization improves stability but causes **oversmoothing**  
- **Wiener deconvolution slightly outperforms Tikhonov** in both PSNR and SSIM  
- **Higher PSNR does not necessarily imply better perceptual quality**  
- SSIM provides better insight into visual reconstruction quality  

👉 This highlights a fundamental limitation of classical reconstruction metrics.

---

## 📈 Experiments

- Gaussian blur with σ = 2  
- Additive Gaussian noise  
- Parameter sweep for Tikhonov regularization:

```

λ ∈ {0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10}

```

---

## 🧪 Project Structure

```

project/
│
├── main.py
├── utils.py
├── results/
│   ├── observed.png
│   ├── tikhonov.png
│   ├── wiener.png
│   └── best_tikhonov.png

````

---

## ▶️ Running the Project

Install dependencies:

```bash
pip install numpy scipy matplotlib scikit-image
````

Run the project:

```bash
python main.py
```

---

## 📤 Outputs

The project generates:

* Observed (blurred + noisy) image
* Tikhonov reconstruction
* Wiener reconstruction
* Best Tikhonov reconstruction
* λ vs. PSNR plot
* λ vs. SSIM plot

---

## 🚀 Future Work

* Deep learning-based reconstruction (CNN-based deblurring)
* Blind deconvolution
* Total variation (TV) regularization
* Non-Gaussian blur models

---

## 📎 Reference

This project is based on a self-developed study:

**"Image Reconstruction in Inverse Problems: A Comparative Study of Tikhonov Regularization and Wiener Deconvolution" (Preprint)**

---

## 👤 Author

Mohammad Reza Shirkhan
M.Sc. Electrical Engineering (Communication Systems)


