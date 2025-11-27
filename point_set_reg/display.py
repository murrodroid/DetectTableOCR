import cv2
import matplotlib.pyplot as plt
import numpy as np


def show_image(img, title=None):
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img

    plt.figure(figsize=(6, 6))
    if title:
        plt.title(title)
    if len(img_rgb.shape) == 2:
        plt.imshow(img_rgb, cmap='gray')
    else:
        plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()

def show_image_with_points(img: np.ndarray,
                           points: np.ndarray,
                           title: str | None = None,
                           point_size: int = 30,
                           point_color: str = "red"):
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img

    plt.figure(figsize=(6, 6))
    if title:
        plt.title(title)

    if len(img_rgb.shape) == 2:
        plt.imshow(img_rgb, cmap="gray")
    else:
        plt.imshow(img_rgb)

    xs = points[:, 0]
    ys = points[:, 1]
    plt.scatter(xs, ys, s=point_size, c=point_color)

    plt.axis("off")
    plt.show()