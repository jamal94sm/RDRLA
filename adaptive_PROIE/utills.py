import cv2
import numpy as np

def padding_img(img, padding=20):
    h, w = img.shape[:2]
    out = np.zeros((h + 2*padding, w + 2*padding, 3), dtype=img.dtype)
    out[padding:padding+h, padding:padding+w] = img
    return out

def _dist(mask):
    return cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2,
                                 cv2.DIST_MASK_PRECISE)

def find_circle_inform_hard(mask):
    dist_map = _dist(mask)
    _, max_val, _, center = cv2.minMaxLoc(dist_map)
    cx, cy = int(center[0]), int(center[1])
    good_region = dist_map.copy()
    good_region[good_region < max_val * 0.9] = 0
    good_region = good_region.astype(np.uint8)
    inner = _dist(good_region)
    radius = inner[cy, cx]
    good_copy = good_region.copy()
    cv2.circle(good_copy, (cx, cy), int(radius), 0, -1)
    black = np.zeros_like(good_copy)
    black[:cy] = good_copy[:cy]
    black[black > 0] = 255
    contours, _ = cv2.findContours(black.copy(), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if not contours:
        return (cx, cy), int(dist_map[cy, cx])
    best = max(contours, key=cv2.contourArea)
    M = cv2.moments(best)
    if M["m00"] == 0:
        return (cx, cy), int(dist_map[cy, cx])
    cx2 = int(M["m10"] / M["m00"])
    cy2 = int(M["m01"] / M["m00"])
    return (cx2, cy2), int(dist_map[cy2, cx2])

def circle_better(img, visualize_img, mask, rate=1.1):
    """Core FFARD constrained inscribed circle — Section III-B-2 of paper."""
    t1, t2 = 0.85, rate
    h, w = mask.shape[:2]

    dc = _dist(mask)
    _, max_r, _, pc = cv2.minMaxLoc(dc)
    x0, y0 = int(pc[0]), int(pc[1])

    # S: acceptable region
    s_mask = (dc > t1 * max_r).astype(np.uint8) * 255
    ds = _dist(s_mask)
    ds_at_pc = float(ds[y0, x0])

    # T: lower/outer band
    yy, xx = np.mgrid[0:h, 0:w]
    dist_to_pc = np.sqrt((xx - x0)**2 + (yy - y0)**2).astype(np.float32)
    t_mask = ((dist_to_pc > ds_at_pc) & (yy >= y0)).astype(np.uint8) * 255

    # S' = S ∩ T
    s_prime = cv2.bitwise_and(s_mask, t_mask)

    x0p, y0p = x0, y0
    contours, _ = cv2.findContours(s_prime, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if contours:
        best = max(contours, key=cv2.contourArea)
        M = cv2.moments(best)
        if M["m00"] != 0:
            x0p = int(M["m10"] / M["m00"])
            y0p = int(M["m01"] / M["m00"])

    final_r = int(t2 * float(dc[y0p, x0p]))
    if final_r <= 4:
        final_r = max(4, int(max_r))

    side = 2 * final_r
    src_x1 = max(0, x0p - final_r); src_x2 = min(w, x0p + final_r)
    src_y1 = max(0, y0p - final_r); src_y2 = min(h, y0p + final_r)
    dst_x1 = src_x1 - (x0p - final_r)
    dst_y1 = src_y1 - (y0p - final_r)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    canvas = np.zeros((side, side, 3), dtype=img.dtype)
    if src_x2 > src_x1 and src_y2 > src_y1:
        canvas[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2, src_x1:src_x2]

    vis = visualize_img.copy()
    cv2.circle(vis, (x0,  y0),  int(max_r), (0,200,0), 2)
    cv2.circle(vis, (x0p, y0p), final_r,    (0,0,220), 2)
    return canvas, vis