# download to model weights: https://www.dropbox.com/s/57zjbwv6gh3srry/model_final.pth?dl=1
# TODO: point set registration

from ultralytics import YOLO
import fitz, numpy as np
import layoutparser as lp


def _table_class_ids(model):
    names = getattr(model.model, 'names', None) or model.names
    return [i for i,n in enumerate(names) if str(n).lower()=='table']

def pdf_to_images(pdf_path, dpi=300):
    doc = fitz.open(pdf_path)
    imgs = []
    for i in range(len(doc)):
        pix = doc[i].get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72), alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
        imgs.append(img)
    doc.close()
    return imgs

def detect_tables(weights_path, images, conf=0.25):
    m = YOLO(weights_path)
    keep = set(_table_class_ids(m))
    rs = m.predict(images, conf=conf, verbose=False)
    outs = []
    for r in rs:
        H, W = r.orig_shape
        boxes = []
        for b in r.boxes:
            c = int(b.cls[0].item())
            if c in keep:
                x1,y1,x2,y2 = b.xyxy[0].tolist()
                boxes.append((x1/W, y1/H, x2/W, y2/H, float(b.conf)))
        outs.append(boxes)
    return outs


def main():
    pass


if __name__ == '__main__':
    main()
