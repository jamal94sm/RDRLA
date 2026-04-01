#!/usr/bin/env python3
"""
run_roi_extraction.py
=====================
Single-script ROI extraction pipeline for MPDv2.
Combines Stage 1 (CHSST hand segmentation) and Stage 2 (LANet alignment + ROI crop).

Place this file at the RDRLA repo root and run:
    python run_roi_extraction.py
"""

# ─────────────────────────────────────────────────────────────────
#  CONFIG — edit these before running
# ─────────────────────────────────────────────────────────────────
MPD_RAW_DIR = "/home/pai-ng/Jamal/MPDv2"
CHSST_CKPT  = "CHSST/CHSST_checkpoints/EP7-iou0.951562-pacc0.977916_CHSST.pth"
LANET_CKPT  = "adaptive_PROIE/LANet_v1.pkl"
SEG_OUT_DIR = "/home/pai-ng/Jamal/MPDv2_segmented"
ROI_OUT_DIR = "/home/pai-ng/Jamal/MPDv2_ROI_FFARD"
VIS_OUT_DIR = "/home/pai-ng/Jamal/MPDv2_vis"
DEVICE      = "cuda"
# ─────────────────────────────────────────────────────────────────

import os
import sys
import math
import warnings
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import tqdm

warnings.filterwarnings("ignore")

# Add repo root to sys.path so CHSST.* imports resolve when torch.load unpickles the model
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ═══════════════════════════════════════════════════════════════════
#  UTILITIES — exact copy of adaptive_PROIE/utills.py
# ═══════════════════════════════════════════════════════════════════

def padding_img(img, padding=20):
    h, w = img.shape[:2]
    out = np.zeros((h + 2 * padding, w + 2 * padding, 3), dtype=img.dtype)
    out[padding:padding + h, padding:padding + w] = img
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
    dist_to_pc = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2).astype(np.float32)
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
    src_x1 = max(0, x0p - final_r);  src_x2 = min(w, x0p + final_r)
    src_y1 = max(0, y0p - final_r);  src_y2 = min(h, y0p + final_r)
    dst_x1 = src_x1 - (x0p - final_r)
    dst_y1 = src_y1 - (y0p - final_r)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    canvas = np.zeros((side, side, 3), dtype=img.dtype)
    if src_x2 > src_x1 and src_y2 > src_y1:
        canvas[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2, src_x1:src_x2]

    vis = visualize_img.copy()
    cv2.circle(vis, (x0,  y0),  int(max_r), (0, 200, 0), 2)
    cv2.circle(vis, (x0p, y0p), final_r,    (0, 0, 220), 2)
    return canvas, vis


# ═══════════════════════════════════════════════════════════════════
#  LANet — exact copy of adaptive_PROIE/LANet.py
# ═══════════════════════════════════════════════════════════════════

# Module-level vgg16 exactly as in LANet.py
vgg16 = torchvision.models.vgg16(pretrained=False)


class LAnet(nn.Module):
    def __init__(self, numclasses=1, ipt_size=56):
        super(LAnet, self).__init__()
        self.input_w = ipt_size
        self.input_h = ipt_size
        self.vgg_p3 = vgg16.features[:17]
        self.extra = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear((self.input_w * self.input_h * 256) // 64, 512),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=0.2),
            nn.Linear(512, 128),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=0.1),
            nn.Linear(128, numclasses),
            nn.Tanh(),
        )

    def forward(self, x):
        out = self.vgg_p3(x)
        out = self.extra(out)
        return out


# ═══════════════════════════════════════════════════════════════════
#  ROI_extraction.py helper functions — exact copy
# ═══════════════════════════════════════════════════════════════════

def generate_heatmap(keypoint_location, heatmap_size, variance):
    x, y = keypoint_location
    x_range = torch.arange(0, heatmap_size[1], 1)
    y_range = torch.arange(0, heatmap_size[0], 1)
    X, Y = torch.meshgrid(x_range, y_range)
    pos = torch.stack((X, Y), dim=2)
    heatmap = torch.exp(
        -(torch.sum((pos - torch.tensor([x, y])) ** 2, dim=2)) / (2.0 * variance ** 2)
    )
    return heatmap


def center_and_pad_image(input_img_cv2, kpts):
    height, width, _ = input_img_cv2.shape
    new_size = int(max(width, height))
    x_offset = (new_size - width) // 2
    y_offset = (new_size - height) // 2
    padded_image = np.zeros((new_size, new_size, 3), dtype=np.uint8)
    padded_image[y_offset:y_offset + height, x_offset:x_offset + width, :] = input_img_cv2
    kpts[:, 0] += x_offset
    kpts[:, 1] += y_offset
    return padded_image, kpts


def get_inter_square(img, Rotate_theta):
    h, w, _ = img.shape
    center = (w // 2, h // 2)
    width = math.sqrt(2) * (w / 2) / 2
    left   = center[0] - int(width)
    right  = center[0] + int(width)
    top    = center[1] - int(width)
    bottom = center[1] + int(width)
    mat = cv2.getRotationMatrix2D(center, Rotate_theta, scale=1)
    rotated_img = cv2.warpAffine(img, mat, (w, h))
    square_roi = rotated_img[top:bottom, left:right]
    return square_roi, rotated_img


def generate_net_ipt(seg_img):
    """
    Mirrors generate_net_ipt() from ROI_extraction.py.
    Takes the segmented palm image array directly instead of a file path.
    """
    img = seg_img
    mask = np.where((img == 0), 0, 255)[:, :, 2].astype(np.uint8)
    center, r = find_circle_inform_hard(mask)
    kpts = np.array([center], dtype=float)

    # Bounding box crop — same logic as ROI_extraction.py
    mask_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, mask_bin = cv2.threshold(mask_gray, 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask_bin.copy(), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, None, None
    m = 0
    m_area = 0
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > m_area:
            m = i
            m_area = area
    contour = contours[m].reshape(-1, 2)
    xs, ys = np.int0(np.min(contour, 0))
    xe, ye = np.int0(np.max(contour, 0))
    rotated_img = img[ys:ye, xs:xe]
    if rotated_img.size == 0:
        return None, None, None
    kpts[:, 0] -= xs
    kpts[:, 1] -= ys

    rotated_img, rotated_points = center_and_pad_image(rotated_img, kpts)
    center = rotated_points[0]
    h, w, _ = rotated_img.shape
    raw_img = rotated_img.copy()

    # Heatmap — exact same swap as ROI_extraction.py
    center_changed = np.array([int(center[1] * 56 / w), int(center[0] * 56 / h)])
    center_hmap = generate_heatmap(center_changed, (56, 56), 2)

    img_small = cv2.resize(rotated_img, (56, 56))
    img_t = np.transpose(img_small, (2, 0, 1)) / 255.
    img_t = torch.from_numpy(img_t.copy()).float()
    img_t[2] = center_hmap

    # r_center with same (cy, cx) swap convention as ROI_extraction.py
    return img_t, raw_img, (int(center[1]), int(center[0]))


# ═══════════════════════════════════════════════════════════════════
#  STAGE 1 — Hand Segmentation (palmSegmentation.py logic)
# ═══════════════════════════════════════════════════════════════════

def process_one_img(rawpth, model, device="cuda"):
    """Exact copy of process_one_img() from palmSegmentation.py."""
    img = cv2.imread(rawpth)
    if img is None:
        return None
    sx = rawpth.split("_")[-1][0]
    h, w, _ = img.shape
    if sx == "L":
        if w > h:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img = cv2.flip(img, 1)
    else:
        if w > h:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    imgIPT = img.copy()
    h, w, _ = img.shape
    inputimg = cv2.resize(imgIPT, (448, 448))
    indata = np.transpose(inputimg, (2, 0, 1)) / 255.
    indata = torch.tensor(indata).float().unsqueeze(0).to(device)
    output = model(indata)
    res = output[0].data.cpu().numpy()
    res = np.expand_dims(res.argmax(axis=0), 2) * 255
    res = np.repeat(res, 3, 2)
    h, w, _ = img.shape
    o = np.zeros((h, w * 2, 3))
    o[:, 0:w, :] = img
    o[:, w:2 * w, :] = cv2.resize(res, (w, h), interpolation=cv2.INTER_NEAREST)
    return o


def segment_one_image(rawpth, model, device="cuda"):
    """
    Runs segmentation on one raw image.
    Mirrors the loop body of segfile() from palmSegmentation.py.
    Returns cropped palm on black background, or None on failure.
    """
    o = process_one_img(rawpth, model, device)
    if o is None:
        return None
    h, w, _ = o.shape
    w = w // 2
    p_img = o[:, :w].astype(np.uint8)
    label = o[:, w:][:, :, 0].astype(np.uint8)
    contours, _ = cv2.findContours(label.copy(), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    best = max(contours, key=cv2.contourArea)
    if cv2.contourArea(best) < 100:
        return None
    filled = np.zeros_like(label)
    cv2.fillPoly(filled, [best], 255)
    filled_3ch = cv2.cvtColor(filled, cv2.COLOR_GRAY2BGR)
    palm = cv2.bitwise_and(p_img, filled_3ch)
    xs, ys = np.int0(np.min(best.reshape(-1, 2), 0))
    xe, ye = np.int0(np.max(best.reshape(-1, 2), 0))
    palm = palm[ys:ye, xs:xe]
    if palm.size == 0:
        return None
    return palm


# ═══════════════════════════════════════════════════════════════════
#  STAGE 2 — Alignment + ROI Extraction (ROI_extraction.py logic)
# ═══════════════════════════════════════════════════════════════════

def process_single_img(seg_palm, lanet, base_name, roi_dir, vis_dir):
    """
    Exact same logic as process_single_img_ipt() from ROI_extraction.py.
    Takes seg_palm image array directly instead of reading from file.
    Returns number of ROI files saved.
    """
    tensor_for_ipt, raw_img, r_center = generate_net_ipt(seg_palm)
    if tensor_for_ipt is None:
        return 0

    tensor_for_ipt = tensor_for_ipt.unsqueeze(0).to(DEVICE)
    theta = lanet(tensor_for_ipt).detach().cpu()[0]
    theta = theta.numpy()
    angle_degrees = np.degrees(theta * np.pi)
    angle_degrees = float(angle_degrees)

    rotation_matrix = cv2.getRotationMatrix2D(r_center, angle_degrees, scale=1.0)
    rawimg = cv2.warpAffine(raw_img, rotation_matrix,
                            (raw_img.shape[1], raw_img.shape[0]))
    rawimg = padding_img(rawimg, 20)
    raw_mask = np.where((rawimg <= 10), 0, 255)[:, :, 2].astype(np.uint8)
    visualize_img = rawimg.copy()

    final_circle, visualize_circle = circle_better(rawimg, visualize_img,
                                                    raw_mask, rate=1.1)

    # Save visualisation — exact same resize as ROI_extraction.py
    h, w, _ = visualize_circle.shape
    visualize_save_iname = base_name + ".jpg"
    visualize_circle_resized = cv2.resize(visualize_circle,
                                          (int(200 * w / h), 200))
    cv2.imwrite(os.path.join(vis_dir, visualize_save_iname),
                visualize_circle_resized)

    # 20 augmented 128×128 ROIs — exact same loop as ROI_extraction.py
    saved = 0
    try:
        for angle in range(-30, 30, 3):
            square_roi, circle_roi = get_inter_square(final_circle, angle)
            if square_roi is None or square_roi.size == 0:
                continue
            save_iname = base_name + "_" + str(angle) + ".jpg"
            square_roi_128 = cv2.resize(square_roi, (128, 128))
            cv2.imwrite(os.path.join(roi_dir, save_iname), square_roi_128)
            saved += 1
    except Exception as e:
        tqdm.tqdm.write(f"  [aug-error] {base_name}: {e}")

    return saved


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    for d in [SEG_OUT_DIR, ROI_OUT_DIR, VIS_OUT_DIR]:
        os.makedirs(d, exist_ok=True)

    # ── Load CHSST segmentation model ───────────────────────────
    print("Loading CHSST segmentation model ...")
    # training.py saves with torch.save(fcn_model, path) — full model object
    from CHSST.models.toptransformer.seaformer import Seaformernet
    seg_model = Seaformernet()
    state_dict = torch.load(CHSST_CKPT, map_location=DEVICE)
    # Strip 'module.' prefix if saved with DataParallel
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    seg_model.load_state_dict(state_dict, strict=False)
    seg_model.to(DEVICE)
    seg_model.eval()
    print("CHSST loaded.\n")

    # ── Load LANet ───────────────────────────────────────────────
    print("Loading LANet ...")
    lanet = LAnet(numclasses=4).to(DEVICE)
    # train_LANet.py saves as {"LANet": net.state_dict()}
    with open(LANET_CKPT, 'rb') as f:
        loaded_params = torch.load(f, map_location=DEVICE)
    lanet.load_state_dict(loaded_params["LANet"])
    lanet.eval()
    print("LANet loaded.\n")

    # ── Collect raw images ───────────────────────────────────────
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    raw_imgs = sorted(
        f for f in os.listdir(MPD_RAW_DIR)
        if os.path.splitext(f)[1].lower() in exts
    )
    print(f"Found {len(raw_imgs)} images in {MPD_RAW_DIR}\n")

    total_rois = 0
    seg_fail   = 0
    roi_fail   = 0

    for img_name in tqdm.tqdm(raw_imgs, desc="Extracting ROIs", unit="img"):
        raw_pth   = os.path.join(MPD_RAW_DIR, img_name)
        base_name = os.path.splitext(img_name)[0]
        seg_pth   = os.path.join(SEG_OUT_DIR, img_name)

        # ── Stage 1: Segmentation ────────────────────────────────
        try:
            with torch.no_grad():
                palm_seg = segment_one_image(raw_pth, seg_model, DEVICE)
        except Exception as e:
            tqdm.tqdm.write(f"  [seg-error] {img_name}: {e}")
            seg_fail += 1
            continue

        if palm_seg is None:
            tqdm.tqdm.write(f"  [seg-empty] {img_name}")
            seg_fail += 1
            continue

        # Save intermediate segmented palm (mirrors palmSegmentation.py)
        cv2.imwrite(seg_pth, palm_seg)

        # ── Stage 2: Alignment + ROI extraction ─────────────────
        try:
            with torch.no_grad():
                n = process_single_img(palm_seg, lanet, base_name,
                                       ROI_OUT_DIR, VIS_OUT_DIR)
        except Exception as e:
            tqdm.tqdm.write(f"  [roi-error] {img_name}: {e}")
            roi_fail += 1
            continue

        if n == 0:
            tqdm.tqdm.write(f"  [roi-empty] {img_name}")
            roi_fail += 1
        total_rois += n

    print(f"\n{'─' * 52}")
    print(f"  Input images   : {len(raw_imgs)}")
    print(f"  Seg failures   : {seg_fail}")
    print(f"  ROI failures   : {roi_fail}")
    print(f"  ROIs saved     : {total_rois}  "
          f"(~{total_rois // max(1, len(raw_imgs) - seg_fail)} per image)")
    print(f"  Seg output     : {SEG_OUT_DIR}")
    print(f"  ROI output     : {ROI_OUT_DIR}")
    print(f"{'─' * 52}")


if __name__ == "__main__":
    main()
