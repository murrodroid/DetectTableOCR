import os, json, matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import numpy as np, fitz

class Annot:
    def __init__(self, img, save_txt):
        self.img = img
        self.H, self.W = img.shape[:2]
        self.save_txt = save_txt
        self.boxes = []

    def on_select(self, eclick, erelease):
        x1, y1 = max(0, min(eclick.xdata, erelease.xdata)), max(0, min(eclick.ydata, erelease.ydata))
        x2, y2 = min(self.W, max(eclick.xdata, erelease.xdata)), min(self.H, max(eclick.ydata, erelease.ydata))
        xc, yc = (x1+x2)/2/self.W, (y1+y2)/2/self.H
        w, h = (x2-x1)/self.W, (y2-y1)/self.H
        self.boxes.append((0, xc, yc, w, h))

    def run(self):
        fig, ax = plt.subplots()
        ax.imshow(self.img)
        ax.set_axis_off()
        rs = RectangleSelector(ax, self.on_select, drawtype='box', useblit=True, interactive=True)
        plt.show()
        with open(self.save_txt, 'w') as f:
            for c,xc,yc,w,h in self.boxes: f.write(f'{c} {xc} {yc} {w} {h}\n')

def pdf_page_to_image(pdf_path, page_idx=0, dpi=300):
    d = fitz.open(pdf_path); p = d[page_idx]
    pix = p.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72), alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
    d.close()
    return img

def annotate_pdf_page(pdf_path, page_idx, out_txt):
    img = pdf_page_to_image(pdf_path, page_idx)
    Annot(img, out_txt).run()