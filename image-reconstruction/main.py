import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage import data, color, img_as_float
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

from utils import (
    gaussian_kernel,
    tikhonov_deblur,
    wiener_deblur,
    show_image,
    show_comparison,
    save_image,
)


if __name__ == "__main__":
    # Load image
    image = data.astronaut()
    image = color.rgb2gray(image)
    image = img_as_float(image)

    show_image(image, "Original Image")

    # Create blurred and noisy observation
    blurred = gaussian_filter(image, sigma=2)

    np.random.seed(42)
    noise = 0.005 * np.random.randn(*image.shape)
    observed = blurred + noise
    observed = np.clip(observed, 0, 1)

    show_image(observed, "Blurred + Noisy Image")
    save_image(observed, "observed.png")

    # Build blur kernel
    kernel = gaussian_kernel(size=9, sigma=2)

    # -----------------------------
    # Tikhonov reconstruction
    # -----------------------------
    reconstructed_tikhonov = tikhonov_deblur(observed, kernel, lambd=0.1)
    reconstructed_tikhonov = np.clip(reconstructed_tikhonov, 0, 1)

    show_image(reconstructed_tikhonov, "Tikhonov Reconstruction")
    show_comparison(
        image,
        "Original",
        observed,
        "Observed",
        reconstructed_tikhonov,
        "Tikhonov",
    )

    tikhonov_psnr = psnr(image, reconstructed_tikhonov, data_range=1.0)
    tikhonov_ssim = ssim(image, reconstructed_tikhonov, data_range=1.0)

    print("Tikhonov PSNR:", tikhonov_psnr)
    print("Tikhonov SSIM:", tikhonov_ssim)

    save_image(reconstructed_tikhonov, "tikhonov.png")

    # -----------------------------
    # Wiener reconstruction
    # -----------------------------
    reconstructed_wiener = wiener_deblur(observed, kernel, k=0.1)
    reconstructed_wiener = np.clip(reconstructed_wiener, 0, 1)

    show_image(reconstructed_wiener, "Wiener Reconstruction")
    show_comparison(
        image,
        "Original",
        observed,
        "Observed",
        reconstructed_wiener,
        "Wiener",
    )

    wiener_psnr = psnr(image, reconstructed_wiener, data_range=1.0)
    wiener_ssim = ssim(image, reconstructed_wiener, data_range=1.0)

    print("Wiener PSNR:", wiener_psnr)
    print("Wiener SSIM:", wiener_ssim)

    save_image(reconstructed_wiener, "wiener.png")

    # -----------------------------
    # Lambda sweep for Tikhonov
    # -----------------------------
    lambdas = [0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
    psnr_values = []
    ssim_values = []

    for lambd in lambdas:
        rec = tikhonov_deblur(observed, kernel, lambd=lambd)
        rec = np.clip(rec, 0, 1)

        psnr_score = psnr(image, rec, data_range=1.0)
        ssim_score = ssim(image, rec, data_range=1.0)

        psnr_values.append(psnr_score)
        ssim_values.append(ssim_score)

        print(f"lambda={lambd}, PSNR={psnr_score:.2f}, SSIM={ssim_score:.4f}")

    print("len(lambdas):", len(lambdas))
    print("len(psnr_values):", len(psnr_values))
    print("len(ssim_values):", len(ssim_values))

    import os

    os.makedirs("results", exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(lambdas, psnr_values, marker="o")
    plt.xscale("log")
    plt.xlabel("Lambda")
    plt.ylabel("PSNR")
    plt.title("Tikhonov: Lambda vs PSNR")
    plt.grid(True)
    plt.savefig("results/lambda_psnr.png", dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(lambdas, ssim_values, marker="o")
    plt.xscale("log")
    plt.xlabel("Lambda")
    plt.ylabel("SSIM")
    plt.title("Tikhonov: Lambda vs SSIM")
    plt.grid(True)
    plt.savefig("results/lambda_ssim.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Best lambda based on PSNR
    best_index = np.argmax(psnr_values)
    best_lambda = lambdas[best_index]
    best_psnr = psnr_values[best_index]
    best_ssim = ssim_values[best_index]

    print(f"Best lambda: {best_lambda}")
    print(f"Best PSNR: {best_psnr:.2f}")
    print(f"Best SSIM: {best_ssim:.4f}")

    # Best Tikhonov reconstruction
    best_tikhonov = tikhonov_deblur(observed, kernel, lambd=best_lambda)
    best_tikhonov = np.clip(best_tikhonov, 0, 1)

    show_image(best_tikhonov, f"Best Tikhonov (lambda={best_lambda})")
    show_comparison(
        image,
        "Original",
        observed,
        "Observed",
        best_tikhonov,
        f"Best Tikhonov ({best_lambda})",
    )

    save_image(best_tikhonov, "best_tikhonov.png")