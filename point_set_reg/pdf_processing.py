from pathlib import Path
import pymupdf
import numpy as np
import cv2


def open_pdf(path: str | Path) -> pymupdf.Document:
    return pymupdf.open(str(path))


def page_to_pixmap(page: pymupdf.Page, dpi: int) -> pymupdf.Pixmap:
    zoom = dpi / 72.0
    matrix = pymupdf.Matrix(zoom, zoom)
    return page.get_pixmap(matrix=matrix, alpha=False)


def pixmap_to_ndarray(pix: pymupdf.Pixmap) -> np.ndarray:
    data = np.frombuffer(pix.samples, dtype=np.uint8)
    arr = data.reshape(pix.height, pix.width, pix.n)
    if pix.n == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr


def pdf_to_images(pdf: str | Path, type: str = "png", dpi: int = 300) -> list[np.ndarray]:
    doc = open_pdf(pdf)
    images: list[np.ndarray] = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        pix = page_to_pixmap(page, dpi=dpi)
        arr = pixmap_to_ndarray(pix)
        images.append(arr)
    doc.close()
    return images