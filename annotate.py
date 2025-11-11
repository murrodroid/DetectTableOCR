# this is all chatgpt lol

"""
annotate.py — click-and-drag rectangle annotation for PDF pages.

Controls (focus the image window):
- Mouse: click, hold, and drag to draw a rectangle (live preview shown).
- u: undo last rectangle on current page
- d: delete (same as undo)
- r: reset current page annotations
- → / ← : next / previous page (autosaves)
- s: save now
- q: save & quit

JSON output (simple):
{
  "meta": { "pdf": "<path>", "dpi": <int> },
  "pages": {
    "<page_index>": { "size": [W, H], "rects": [[x1,y1,x2,y2], ...] },
    ...
  }
}
Coordinates are pixel-space at the rendered DPI.
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pymupdf  # pip install pymupdf
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector


# -------- PDF helpers --------
def render_page(pdf_path: str, page_index: int, dpi: int) -> Image.Image:
    doc = pymupdf.open(pdf_path)
    try:
        page = doc[page_index]
        zoom = dpi / 72.0
        mat = pymupdf.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False, colorspace=pymupdf.csRGB)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
        return Image.fromarray(arr)
    finally:
        doc.close()


def num_pages(pdf_path: str) -> int:
    doc = pymupdf.open(pdf_path)
    try:
        return len(doc)
    finally:
        doc.close()


# -------- Annotator --------
class RectAnnotator:
    def __init__(self, pdf_path: str, out_json: str, dpi: int, start_page: int):
        self.pdf_path = pdf_path
        self.out_json = out_json
        self.dpi = dpi
        self.N = num_pages(pdf_path)

        # Load or init annotations
        if os.path.exists(out_json):
            with open(out_json, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        else:
            self.data = {"meta": {"pdf": pdf_path, "dpi": dpi}, "pages": {}}

        # normalize meta
        self.data.setdefault("meta", {"pdf": pdf_path, "dpi": dpi})
        self.data["meta"]["pdf"] = pdf_path
        self.data["meta"]["dpi"] = dpi
        self.data.setdefault("pages", {})

        self.page_index = max(0, min(self.N - 1, start_page))
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.manager.set_window_title("PDF Rectangle Annotator")
        self.img_artist = None
        self.rs = None  # RectangleSelector

        self.live_rect_patch: Rectangle | None = None
        self._load_page()
        self._connect_events()

    # --- Page state helpers ---
    def _page_entry(self) -> Dict:
        key = str(self.page_index)
        if key not in self.data["pages"]:
            W, H = self.img.size
            self.data["pages"][key] = {"size": [float(W), float(H)], "rects": []}
        return self.data["pages"][key]

    def _get_rects(self) -> List[List[float]]:
        return self._page_entry()["rects"]

    def _set_rects(self, rects: List[List[float]]):
        self._page_entry()["rects"] = rects

    # --- UI wiring ---
    def _connect_events(self):
        self.cid_key = self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        # Live rectangle selector (drawtype="box" shows a box while dragging)
        self.rs = RectangleSelector(
            self.ax,
            onselect=self.on_select,
            useblit=True,
            button=[1],        # left mouse
            minspanx=2,
            minspany=2,
            spancoords="pixels",
            interactive=False,
            drag_from_anywhere=False,
        )

    def _clear_live_patch(self):
        if self.live_rect_patch is not None:
            try:
                self.live_rect_patch.remove()
            except Exception:
                pass
            self.live_rect_patch = None

    def _load_page(self):
        # Render
        self.img = render_page(self.pdf_path, self.page_index, self.dpi)
        self.ax.clear()
        self.ax.imshow(self.img)
        self.ax.set_title(f"Page {self.page_index+1}/{self.N} — drag to mark a table area")
        self.ax.axis("off")

        # Draw existing rects
        for x1, y1, x2, y2 in self._get_rects():
            self._draw_rect_patch(x1, y1, x2, y2, existing=True)

        self._clear_live_patch()
        self.fig.canvas.draw_idle()

    def _draw_rect_patch(self, x1, y1, x2, y2, existing=False):
        x = min(x1, x2)
        y = min(y1, y2)
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        rect = Rectangle((x, y), w, h,
                         fill=False,
                         linewidth=2 if existing else 1.5,
                         linestyle="-" if existing else "--")
        self.ax.add_patch(rect)
        if not existing:
            self.live_rect_patch = rect
        self.fig.canvas.draw_idle()
        return rect

    # --- RectangleSelector callback ---
    def on_select(self, eclick, erelease):
        # Clear live preview (RectangleSelector handles its own blit; we add a permanent patch)
        self._clear_live_patch()

        x1, y1 = float(eclick.xdata), float(eclick.ydata)
        x2, y2 = float(erelease.xdata), float(erelease.ydata)

        # Ignore tiny or invalid drags
        if x1 is None or y1 is None or x2 is None or y2 is None:
            return
        if abs(x2 - x1) < 2 or abs(y2 - y1) < 2:
            return

        rects = self._get_rects()
        rects.append([x1, y1, x2, y2])
        self._set_rects(rects)

        # Draw permanent rectangle
        self._draw_rect_patch(x1, y1, x2, y2, existing=True)

    # --- Keyboard controls ---
    def on_key(self, event):
        key = (event.key or "").lower()
        if key in ("right",):
            self._autosave()
            if self.page_index < self.N - 1:
                self.page_index += 1
                self._load_page()

        elif key in ("left",):
            self._autosave()
            if self.page_index > 0:
                self.page_index -= 1
                self._load_page()

        elif key in ("u", "d"):
            rects = self._get_rects()
            if rects:
                rects.pop()
                self._set_rects(rects)
                self._load_page()

        elif key == "r":
            self._set_rects([])
            self._load_page()

        elif key == "s":
            self._save()
            print(f"Saved to {self.out_json}")

        elif key == "q":
            self._autosave()
            plt.close(self.fig)

    # --- Persistence ---
    def _save(self):
        # Clean empty pages
        cleaned_pages: Dict[str, Dict] = {}
        for k, v in self.data["pages"].items():
            rects = v.get("rects", [])
            if rects:
                cleaned_pages[k] = {"size": v.get("size", [0.0, 0.0]), "rects": rects}
        payload = {"meta": self.data["meta"], "pages": cleaned_pages}

        tmp = self.out_json + ".tmp"
        os.makedirs(os.path.dirname(os.path.abspath(self.out_json)), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.out_json)

    def _autosave(self):
        try:
            self._save()
        except Exception as e:
            print(f"[warn] autosave failed: {e}")


def main():
    ap = argparse.ArgumentParser(description="Click-and-drag rectangle annotator for PDF tables.")
    ap.add_argument("pdf", help="Path to PDF")
    ap.add_argument("-o", "--out", default="annotations.json", help="Output JSON")
    ap.add_argument("--dpi", type=int, default=300, help="Render DPI")
    ap.add_argument("--page", type=int, default=1, help="Start at 1-based page index")
    args = ap.parse_args()

    start_page = max(1, args.page) - 1

    annot = RectAnnotator(args.pdf, args.out, args.dpi, start_page)
    print("Instructions:")
    print("  • Drag with the left mouse button to create a rectangle (live preview shown).")
    print("  • u / d: undo last rectangle, r: reset page, s: save, q: save & quit, ←/→: page nav")
    plt.show()

    annot._autosave()


if __name__ == "__main__":
    main()
