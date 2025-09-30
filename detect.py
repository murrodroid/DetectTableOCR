# download to model weights: https://www.dropbox.com/s/57zjbwv6gh3srry/model_final.pth?dl=1
# TODO: point set registration

from typing import Dict, List, Tuple, Iterable, Optional
import os
import pymupdf
from PIL import Image, ImageDraw
import numpy as np
import torch
from transformers import AutoImageProcessor, TableTransformerForObjectDetection


def pdf_to_images(
    pdf_path: str,
    pages: Optional[Iterable[int]] = None,
    dpi: int = 200,
    color: bool = True,
) -> Dict[int, Image.Image]:
    """
    Render selected PDF pages to PIL Images. If pages=None, render ALL pages.
    Returns {page_index: PIL.Image}.
    """
    doc = pymupdf.open(pdf_path)
    try:
        if pages is None:
            pages = range(len(doc))

        zoom = dpi / 72.0
        mat = pymupdf.Matrix(zoom, zoom)
        images: Dict[int, Image.Image] = {}

        for p in pages:
            page = doc[p]
            pix = page.get_pixmap(matrix=mat, alpha=False,
                                  colorspace=(pymupdf.csRGB if color else pymupdf.csGRAY))
            if color:
                # Shape: (H, W, 3) -> mode inferred as RGB; avoid deprecated 'mode' arg
                arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
            else:
                # Shape: (H, W)
                arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w)
            img = Image.fromarray(arr)
            images[p] = img
        return images
    finally:
        doc.close()

def visualize_table_detections(
    pdf_path: str,
    results: Dict[int, List[dict]],
    out_dir: str = "visualizations",
    dpi: Optional[int] = None,  # pass the same dpi you used for detection; if None, we'll try to infer
    rect_thickness: int = 4,    # outline thickness in pixels (auto-scaled minimally)
) -> Dict[int, str]:
    """
    Draw red rectangles over detected tables and save one PNG per page.

    Args:
        pdf_path: Path to the source PDF.
        results:  {page_index: [ { "bbox_xyxy":[x1,y1,x2,y2], "image_size": {"width":W,"height":H}, ...}, ...], ... }
                  (Output format from detect_tables_on_pages)
        out_dir:  Directory to write output images.
        dpi:      If provided, render the PDF at this DPI. If None, try to infer scale from results['image_size'].
        rect_thickness: Base rectangle thickness in pixels.

    Returns:
        Dict mapping page_index -> saved image path.
    """
    os.makedirs(out_dir, exist_ok=True)
    saved = {}

    doc = pymupdf.open(pdf_path)
    try:
        for p in sorted(results.keys()):
            page = doc[p]

            # Pick a rendering scale:
            if dpi is not None:
                zoom = dpi / 72.0
                mat = pymupdf.Matrix(zoom, zoom)
            else:
                # Try to infer target image size from results (first det that has it)
                targetW = targetH = None
                for det in results[p]:
                    sz = det.get("image_size")
                    if sz and "width" in sz and "height" in sz:
                        targetW, targetH = float(sz["width"]), float(sz["height"])
                        break
                if targetW and targetH:
                    # page.rect is in points (1/72"). At 72 dpi, pixel width == page.rect.width
                    zoomx = targetW / float(page.rect.width)
                    zoomy = targetH / float(page.rect.height)
                    mat = pymupdf.Matrix(zoomx, zoomy)
                else:
                    # safe fallback
                    mat = pymupdf.Matrix(200/72.0, 200/72.0)

            pix = page.get_pixmap(matrix=mat, alpha=False, colorspace=pymupdf.csRGB)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
            img = Image.fromarray(arr)
            draw = ImageDraw.Draw(img)

            # Scale thickness a bit for very large pages
            auto_thick = max(rect_thickness, max(img.size) // 800)

            for det in results[p]:
                box = det.get("bbox_xyxy") or det.get("bbox")
                if not box or len(box) != 4:
                    continue
                x1, y1, x2, y2 = [float(v) for v in box]
                draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=auto_thick)

            out_path = os.path.join(out_dir, f"page_{p+1:03d}.png")
            img.save(out_path)
            saved[p] = out_path

    finally:
        doc.close()

    return saved

def detect_tables(
    pdf_path: str,
    page_indices: Optional[Tuple[int, ...]] = None,  # None => all pages
    score_threshold: float = 0.8,
    dpi: int = 300,
    device: Optional[str] = None,
) -> Dict[int, List[dict]]:
    """
    Returns: {page_index: [ {score, bbox_xyxy, corners, size, image_size}, ... ]}
    Coordinates are pixel-space at the chosen dpi.
    """
    # pick device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # IMPORTANT: use_fast=False to avoid the SizeDict / resize issues on some builds
    processor = AutoImageProcessor.from_pretrained(
        "microsoft/table-transformer-detection",
        use_fast=False
    )
    model = TableTransformerForObjectDetection.from_pretrained(
        "microsoft/table-transformer-detection"
    ).to(device).eval()

    imgs = pdf_to_images(pdf_path, pages=page_indices, dpi=dpi, color=True)

    results: Dict[int, List[dict]] = {}
    with torch.no_grad():
        for p, img in imgs.items():
            # pass a plain dict for size; this also silences the max_size deprecation warning
            inputs = processor(
                images=img,
                return_tensors="pt",
                size={"shortest_edge": 154,"longest_edge": 1536}  # reduce if VRAM is tight; increase for tiny tables
            ).to(device)

            outputs = model(**inputs)

            H, W = img.size[1], img.size[0]  # (height, width)
            post = processor.post_process_object_detection(
                outputs, threshold=score_threshold, target_sizes=[(H, W)]
            )[0]

            dets = []
            for bbox, score in zip(post["boxes"], post["scores"]):
                x1, y1, x2, y2 = [float(v) for v in bbox.tolist()]
                dets.append({
                    "score": float(score),
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "corners": {
                        "top_left":     [x1, y1],
                        "top_right":    [x2, y1],
                        "bottom_right": [x2, y2],
                        "bottom_left":  [x1, y2],
                    },
                    "size": {"width": x2 - x1, "height": y2 - y1},
                    "image_size": {"width": W, "height": H},
                })
            results[p] = dets

    return results

def main():
    pdf_path = '1880.pdf'
    dpi = 300
    results = detect_tables(pdf_path=pdf_path,page_indices=None, dpi=dpi)
    visualize_table_detections(pdf_path=pdf_path, results=results, dpi=dpi)


if __name__ == '__main__':
    main()