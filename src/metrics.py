"""
metrics.py
Everything we need to MEASURE how well victims (swimmers) are found:
  * read ground-truth boxes
  * run the detector
  * match detections with ground truth and count hits / misses
Main metric = VICTIM RECALL = (swimmers found) / (swimmers that really exist)
"""
import os
import cv2
import numpy as np
import yaml


def load_names(data_yaml):
    d = yaml.safe_load(open(data_yaml))
    n = d['names']
    return n if isinstance(n, dict) else dict(enumerate(n))


def victim_ids_from_yaml(data_yaml, victim_words=('swimmer',)):
    """Which class ids count as 'victims'? Every class whose name contains 'swimmer'."""
    return [i for i, n in load_names(data_yaml).items() if any(w in n.lower() for w in victim_words)]


def read_gt_boxes(label_path, W, H, keep_ids=None, with_class=False):
    """Read a YOLO label file and return boxes in PIXELS: (x1, y1, x2, y2)."""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    for line in open(label_path):
        p = line.split()
        if len(p) != 5:
            continue
        c = int(float(p[0]))
        cx, cy, w, h = [float(v) for v in p[1:]]
        if keep_ids is not None and c not in keep_ids:
            continue
        b = ((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H)
        boxes.append((c,) + b if with_class else b)
    return boxes


def box_iou(a, b):
    """Intersection-over-Union of two boxes (x1, y1, x2, y2). 1 = identical, 0 = no overlap."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_boxes(gt, pred, iou_thr=0.5):
    """
    gt   : list of (x1,y1,x2,y2)             - real victims
    pred : list of (x1,y1,x2,y2,conf,...)    - what the detector found
    Returns (tp, fn, fp): found, missed, false alarms.
    """
    used, tp = set(), 0
    for p in sorted(pred, key=lambda b: -b[4]):        # most confident detections first
        best, best_iou = -1, iou_thr
        for gi, g in enumerate(gt):
            if gi in used:
                continue
            v = box_iou(g, p[:4])
            if v >= best_iou:
                best, best_iou = gi, v
        if best >= 0:
            used.add(best)
            tp += 1
    return tp, len(gt) - tp, len(pred) - tp


def summarize(tp, fn, fp):
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    return dict(tp=tp, fn=fn, fp=fp, victim_recall=round(rec, 4), victim_precision=round(prec, 4))


def detect(model, img_bgr, conf=0.25, imgsz=960, keep_ids=None):
    """Run the YOLO model on one image. Returns list of (x1, y1, x2, y2, confidence, class)."""
    res = model.predict(img_bgr, conf=conf, imgsz=imgsz, verbose=False)[0]
    out = []
    for b in res.boxes:
        c = int(b.cls[0])
        if keep_ids is not None and c not in keep_ids:
            continue
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
        out.append((x1, y1, x2, y2, float(b.conf[0]), c))
    return out


def list_items(yolo_dir, split='val'):
    """List (image_path, label_path) pairs of a split, sorted by file name."""
    img_dir, lab_dir = f'{yolo_dir}/images/{split}', f'{yolo_dir}/labels/{split}'
    items = []
    for f in sorted(os.listdir(img_dir)):
        if f.lower().endswith('.jpg'):
            items.append((f'{img_dir}/{f}', f'{lab_dir}/{os.path.splitext(f)[0]}.txt'))
    return items


def frames_with_victims(items, victim_ids, n):
    """Keep only frames that contain at least one victim, spread evenly, at most n frames."""
    keep = []
    for ip, lp in items:
        if not os.path.exists(lp):
            continue
        ids = [int(float(l.split()[0])) for l in open(lp) if l.strip()]
        if any(i in victim_ids for i in ids):
            keep.append((ip, lp))
    if len(keep) > n:
        step = len(keep) / n
        keep = [keep[int(i * step)] for i in range(n)]
    return keep


def evaluate_clean(model, items, victim_ids, conf=0.25, imgsz=960):
    """Victim recall on ORIGINAL (uncompressed) frames = the upper bound for every streaming method."""
    tp = fn = fp = 0
    for ip, lp in items:
        img = cv2.imread(ip)
        H, W = img.shape[:2]
        gt = read_gt_boxes(lp, W, H, victim_ids)
        pr = detect(model, img, conf, imgsz, victim_ids)
        t, f, p = match_boxes(gt, pr)
        tp, fn, fp = tp + t, fn + f, fp + p
    return summarize(tp, fn, fp)


def draw_boxes(img, gt=None, pred=None):
    """Draw ground truth (green) and detections (red) on a copy of the image."""
    out = img.copy()
    for b in (gt or []):
        cv2.rectangle(out, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
    for b in (pred or []):
        cv2.rectangle(out, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 2)
        cv2.putText(out, f'{b[4]:.2f}', (int(b[0]), max(int(b[1]) - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return out
