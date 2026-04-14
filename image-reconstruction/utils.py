import os
import numpy as np
import matplotlib.pyplot as plt


def gaussian_kernel(size=9, sigma=2):
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)


def tikhonov_deblur(y, kernel, lambd):
    y_fft = np.fft.fft2(y)
    kernel_fft = np.fft.fft2(kernel, s=y.shape)
    kernel_conj = np.conj(kernel_fft)

    rows, cols = y.shape

    dx = np.array([[1, -1]])
    dy = np.array([[1], [-1]])

    dx_fft = np.fft.fft2(dx, s=(rows, cols))
    dy_fft = np.fft.fft2(dy, s=(rows, cols))

    regularization = np.abs(dx_fft) ** 2 + np.abs(dy_fft) ** 2

    x_fft = (kernel_conj * y_fft) / (np.abs(kernel_fft) ** 2 + lambd * regularization)
    x = np.fft.ifft2(x_fft)

    return np.real(x)


def wiener_deblur(y, kernel, k):
    y_fft = np.fft.fft2(y)
    kernel_fft = np.fft.fft2(kernel, s=y.shape)
    kernel_conj = np.conj(kernel_fft)

    x_fft = (kernel_conj / (np.abs(kernel_fft) ** 2 + k)) * y_fft
    x = np.fft.ifft2(x_fft)

    return np.real(x)


def show_image(image, title):
    plt.imshow(image, cmap="gray", vmin=0, vmax=1)
    plt.title(title)
    plt.axis("off")
    plt.show()


def show_comparison(image1, title1, image2, title2, image3, title3):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(image1, cmap="gray", vmin=0, vmax=1)
    axs[0].set_title(title1)

    axs[1].imshow(image2, cmap="gray", vmin=0, vmax=1)
    axs[1].set_title(title2)

    axs[2].imshow(image3, cmap="gray", vmin=0, vmax=1)
    axs[2].set_title(title3)

    for ax in axs:
        ax.axis("off")

    plt.show()


def save_image(image, filename):
    folder = "results"
    os.makedirs(folder, exist_ok=True)

    path = os.path.join(folder, filename)
    plt.imsave(path, image, cmap="gray", vmin=0, vmax=1)