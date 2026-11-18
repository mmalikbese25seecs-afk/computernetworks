"""
roi_streaming.py  -  OUR PROPOSED IMPROVEMENT: victim-aware ROI streaming
Idea: the drone first runs a quick 'scout' pass on the raw frame and looks for possible victims.
Tiles around possible victims are sent in HIGH quality, the rest of the frame in LOW quality.
Same byte budget as the baseline, but the bytes are spent where the victims are.
"""
import cv2
import numpy as np
import metrics
import streaming

TILE = 64          # tile size in pixels (multiple of 16 so JPEG blocks line up)
MARGIN = 48        # extra pixels around every possible victim

# (quality of background, quality of ROI) from best to cheapest
ROI_LADDER = [(50, 90), (40, 90), (30, 90), (20, 85), (15, 85), (10, 80), (5, 75)]


def roi_mask_from_boxes(shape, boxes, margin=MARGIN):
    """1 = pixel belongs to a region of interest (tile-aligned), 0 = background."""
    H, W = shape[:2]
    mask = np.zeros((H, W), np.uint8)
    for b in boxes:
        x1, y1, x2, y2 = b[:4]
        a = max(int(x1) - margin, 0) // TILE * TILE
        t = max(int(y1) - margin, 0) // TILE * TILE
        c = (min(int(x2) + margin, W - 1) // TILE + 1) * TILE
        d = (min(int(y2) + margin, H - 1) // TILE + 1) * TILE
        mask[t:d, a:c] = 1
    return mask


def roi_encode(img, mask, q_lo, q_hi):
    """Returns (image the receiver sees, total bytes sent). Background low quality + ROI high quality."""
    n_lo, lo = streaming.jpeg_encode(img, q_lo)
    if mask.sum() == 0:
        return lo, n_lo
    hi_src = img * mask[:, :, None]                       # black everywhere except the ROI (costs almost nothing)
    n_hi, hi = streaming.jpeg_encode(hi_src, q_hi)
    out = np.where(mask[:, :, None] == 1, hi, lo).astype(np.uint8)
    return out, n_lo + n_hi


def roi_encoder(model, victim_ids, budgets, scout_conf=0.10, imgsz=960, cache=None):
    """Encoder with the same interface as streaming.abr_encoder, so run_stream can use it."""
    if cache is None:
        cache = {}

    def enc(i, img):
        if i not in cache:                                # the scout result does not depend on the budget
            cache[i] = metrics.detect(model, img, scout_conf, imgsz, victim_ids)
        scout = cache[i]
        mask = roi_mask_from_boxes(img.shape, scout)
        result = None
        for q_lo, q_hi in ROI_LADDER:
            dec, n = roi_encode(img, mask, q_lo, q_hi)
            result = (dec, n, {'q_lo': q_lo, 'q_hi': q_hi,
                               'roi_pct': round(100 * float(mask.mean()), 1), 'n_scout': len(scout)})
            if n <= budgets[i]:
                break
        return result
    return enc
