import glob, numpy as np
from detect import detect_tables, pdf_to_images
from ultralytics import YOLO

def pick_for_annotation(pdf_paths, model_path, k=10, conf=0.25):
    m = YOLO(model_path)
    pool = []
    for pdf in pdf_paths:
        imgs = pdf_to_images(pdf)
        rs = m.predict(imgs, conf=conf, verbose=False)
        for i, r in enumerate(rs):
            if r.boxes.shape[0] == 0 or float(r.boxes.conf.min()) < 0.3:
                pool.append((pdf, i, float(r.boxes.conf.min()) if r.boxes.shape[0] else 0.0))
    pool.sort(key=lambda x: x[2])
    return pool[:k]