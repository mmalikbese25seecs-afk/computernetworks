"""
streaming.py
Simulates sending video frames from a drone to the ground station over a limited link.
BASELINES implemented here:
  1. Fixed quality  : every frame is JPEG-compressed with the same quality.
  2. Adaptive (ABR) : for every frame, pick the best JPEG quality that fits the CURRENT bandwidth budget.
                      (It does not care WHAT is in the picture - that is its weakness.)
"""
import cv2
import numpy as np
import metrics

LADDER = [10, 20, 30, 40, 50, 60, 70, 80, 90]      # JPEG quality levels we can choose from


def jpeg_encode(img, quality):
    """Compress an image as JPEG. Returns (size in bytes, the decoded image the receiver would see)."""
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return len(buf), cv2.imdecode(buf, cv2.IMREAD_COLOR)


def reference_bytes(items, q=70, n=30):
    """Typical frame size at JPEG quality q. We express the bandwidth as a fraction of this number."""
    sizes = []
    for ip, _ in items[:n]:
        sizes.append(jpeg_encode(cv2.imread(ip), q)[0])
    return float(np.median(sizes))


def make_bandwidth_trace(n_frames, seed=0, low=0.25, high=1.5):
    """
    Fake but realistic link: bandwidth wanders up and down and sometimes drops suddenly.
    Value 1.0 means 'exactly enough for a quality-70 frame'.
    """
    rng = np.random.RandomState(seed)
    level = rng.uniform(low, high)
    trace = np.zeros(n_frames)
    for i in range(n_frames):
        level += rng.normal(0, 0.08)                      # slow random walk
        if rng.rand() < 0.03:                             # sudden drop (e.g. drone flies behind a ship)
            level = rng.uniform(low, low + 0.15)
        level = float(np.clip(level, low, high))
        trace[i] = level
    return trace


def fixed_quality_encoder(q):
    def enc(i, img):
        n, dec = jpeg_encode(img, q)
        return dec, n, {'q': q}
    return enc


def abr_encoder(budgets):
    """Baseline 2: choose the highest quality whose size fits the byte budget of THIS frame."""
    def enc(i, img):
        budget = budgets[i]
        for q in reversed(LADDER):                        # try 90, 80, ... 10
            n, dec = jpeg_encode(img, q)
            if n <= budget:
                return dec, n, {'q': q, 'budget': budget}
        n, dec = jpeg_encode(img, LADDER[0])              # even the lowest quality is too big
        return dec, n, {'q': LADDER[0], 'budget': budget}
    return enc


def run_stream(model, items, encoder, victim_ids, conf=0.25, imgsz=960, budgets=None):
    """
    Push every frame through `encoder` (compression), then let the detector look at what the
    receiver would see. Count how many victims were found.
    """
    tp = fn = fp = over = 0
    sizes, infos = [], []
    for i, (ip, lp) in enumerate(items):
        img = cv2.imread(ip)
        H, W = img.shape[:2]
        gt = metrics.read_gt_boxes(lp, W, H, victim_ids)
        dec, nbytes, info = encoder(i, img)
        pr = metrics.detect(model, dec, conf, imgsz, victim_ids)
        t, f, p = metrics.match_boxes(gt, pr)
        tp, fn, fp = tp + t, fn + f, fp + p
        sizes.append(nbytes)
        infos.append(info)
        if budgets is not None and nbytes > budgets[i]:
            over += 1
    out = metrics.summarize(tp, fn, fp)
    out['mean_kb_per_frame'] = round(float(np.mean(sizes)) / 1000, 1)
    out['frames_over_budget'] = over
    return out, sizes, infos
