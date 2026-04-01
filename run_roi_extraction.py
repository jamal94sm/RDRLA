#!/usr/bin/env python3
"""
run_roi_extraction.py
=====================
Single-script ROI extraction pipeline for MPDv2.
Stage 1: CHSST hand segmentation  (palmSegmentation.py)
Stage 2: LANet alignment + ROI crop (ROI_extraction.py)

Run from RDRLA repo root:
    python run_roi_extraction.py
"""

# ─────────────────────────────────────────────────────────────────
#  CONFIG
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

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ═══════════════════════════════════════════════════════════════════
#  utills.py — padding_img, find_circle_inform_hard, circle_better
# ═══════════════════════════════════════════════════════════════════

def padding_img(img, padding=20):
    h, w = img.shape[:2]
    out = np.zeros((h + 2 * padding, w + 2 * padding, 3), dtype=img.dtype)
    out[padding:padding + h, padding:padding + w] = img
    return out


def _dist(mask):
    return cv2.distanceTransform(
        mask.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def find_circle_inform_hard(mask):
    dist_map = _dist(mask)
    _, max_val, _, center_flag = cv2.minMaxLoc(dist_map)
    good_val    = max_val * 0.9
    good_region = dist_map.copy()
    good_region[good_region < good_val] = 0
    good_region[good_region >= good_val] == 255          # exact from dataset.py
    _, radius, _, center = cv2.minMaxLoc(dist_map)
    center_x, center_y  = center
    good_region = good_region.astype(np.uint8)
    inner_dist_map = cv2.distanceTransform(
        good_region, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    radius = inner_dist_map[center_y, center_x]
    cv2.circle(good_region, center, int(radius), 0, -1)
    black_image = np.zeros_like(good_region)
    black_image[:center_y] = good_region[:center_y]
    black_image[black_image > 0] = 255
    contours, _ = cv2.findContours(
        black_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    m = 0
    m_area = 0
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > m_area:
            m = i
            m_area = area
    if not contours:
        return center, int(radius)
    contour = contours[m]
    M  = cv2.moments(contour)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    center = (cx, cy)
    r = int(dist_map[cy][cx])
    return center, r


def circle_better(img, visualize_img, mask, rate=1.1):
    t1, t2 = 0.85, rate
    h, w = mask.shape[:2]

    dc           = _dist(mask)
    _, max_r, _, pc = cv2.minMaxLoc(dc)
    x0, y0       = int(pc[0]), int(pc[1])

    s_mask   = (dc > t1 * max_r).astype(np.uint8) * 255
    ds       = _dist(s_mask)
    ds_at_pc = float(ds[y0, x0])

    yy, xx      = np.mgrid[0:h, 0:w]
    dist_to_pc  = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2).astype(np.float32)
    t_mask      = ((dist_to_pc > ds_at_pc) & (yy >= y0)).astype(np.uint8) * 255

    s_prime     = cv2.bitwise_and(s_mask, t_mask)

    x0p, y0p    = x0, y0
    contours, _ = cv2.findContours(
        s_prime, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        best = max(contours, key=cv2.contourArea)
        M    = cv2.moments(best)
        if M["m00"] != 0:
            x0p = int(M["m10"] / M["m00"])
            y0p = int(M["m01"] / M["m00"])

    final_r = int(t2 * float(dc[y0p, x0p]))
    if final_r <= 4:
        final_r = max(4, int(max_r))

    side    = 2 * final_r
    src_x1  = max(0, x0p - final_r);  src_x2 = min(w, x0p + final_r)
    src_y1  = max(0, y0p - final_r);  src_y2 = min(h, y0p + final_r)
    dst_x1  = src_x1 - (x0p - final_r)
    dst_y1  = src_y1 - (y0p - final_r)
    dst_x2  = dst_x1 + (src_x2 - src_x1)
    dst_y2  = dst_y1 + (src_y2 - src_y1)

    canvas  = np.zeros((side, side, 3), dtype=img.dtype)
    if src_x2 > src_x1 and src_y2 > src_y1:
        canvas[dst_y1:dst_y2, dst_x1:dst_x2] = \
            img[src_y1:src_y2, src_x1:src_x2]

    vis = visualize_img.copy()
    cv2.circle(vis, (x0,  y0),  int(max_r), (0, 200, 0), 2)
    cv2.circle(vis, (x0p, y0p), final_r,    (0, 0, 220), 2)
    return canvas, vis


# ═══════════════════════════════════════════════════════════════════
#  LANet.py — exact copy
#  numclasses=4: confirmed from checkpoint shape torch.Size([4, 128])
# ═══════════════════════════════════════════════════════════════════

vgg16 = torchvision.models.vgg16(pretrained=False)


class LAnet(nn.Module):
    def __init__(self, numclasses=1, ipt_size=56):
        super(LAnet, self).__init__()
        self.input_w = ipt_size
        self.input_h = ipt_size
        self.vgg_p3  = vgg16.features[:17]
        self.extra   = nn.Sequential(
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
#  ROI_extraction.py helpers — exact copy
# ═══════════════════════════════════════════════════════════════════

def generate_heatmap(keypoint_location, heatmap_size, variance):
    x, y    = keypoint_location
    x_range = torch.arange(0, heatmap_size[1], 1)
    y_range = torch.arange(0, heatmap_size[0], 1)
    X, Y    = torch.meshgrid(x_range, y_range)
    pos     = torch.stack((X, Y), dim=2)
    heatmap = torch.exp(
        -(torch.sum((pos - torch.tensor([x, y])) ** 2, dim=2))
        / (2.0 * variance ** 2))
    return heatmap


def center_and_pad_image(input_img_cv2, kpts):
    height, width, _ = input_img_cv2.shape
    new_size = int(max(width, height))
    x_offset = (new_size - width)  // 2
    y_offset = (new_size - height) // 2
    padded_image = np.zeros((new_size, new_size, 3), dtype=np.uint8)
    padded_image[y_offset:y_offset + height,
                 x_offset:x_offset + width, :] = input_img_cv2
    kpts[:, 0] += x_offset
    kpts[:, 1] += y_offset
    return padded_image, kpts


def get_inter_square(img, Rotate_theta):
    h, w, _ = img.shape
    center   = (w // 2, h // 2)
    width    = math.sqrt(2) * (w / 2) / 2
    left     = center[0] - int(width)
    right    = center[0] + int(width)
    top      = center[1] - int(width)
    bottom   = center[1] + int(width)
    mat         = cv2.getRotationMatrix2D(center, Rotate_theta, scale=1)
    rotated_img = cv2.warpAffine(img, mat, (w, h))
    square_roi  = rotated_img[top:bottom, left:right]
    return square_roi, rotated_img


def generate_net_ipt(root, imgn):
    """
    Exact copy of generate_net_ipt() from ROI_extraction.py.
    root/imgn: accepts either a file path split or a pre-loaded array.
    When imgn is None, root is treated as the image array directly.
    """
    if imgn is None:
        img = root                                   # array passed directly
    else:
        img = cv2.imread(os.path.join(root, imgn))

    mask   = np.where((img == 0), 0, 255)[:, :, 2].astype(np.uint8)
    center, r = find_circle_inform_hard(mask)
    h, w, _ = img.shape
    kpts   = np.array([center])

    # exact from ROI_extraction.py
    mask = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    m = 0
    m_area = 0
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > m_area:
            m = i
            m_area = area
    if not contours:
        return None, None, None
    contour = contours[m].reshape(-1, 2)
    xs, ys  = np.intp(np.min(contour, 0))   # np.int0 removed in numpy>=1.24
    xe, ye  = np.intp(np.max(contour, 0))
    rotated_img = img[ys:ye, xs:xe]
    if rotated_img.size == 0:
        return None, None, None
    kpts[:, 0] -= xs
    kpts[:, 1] -= ys

    rotated_img, rotated_points = center_and_pad_image(rotated_img, kpts)
    center   = rotated_points[0]
    h, w, _  = rotated_img.shape
    raw_img  = rotated_img.copy()

    # exact swap from ROI_extraction.py
    center_changed = np.array(
        [int(center[1] * 56 / w), int(center[0] * 56 / h)])
    center_hmap = generate_heatmap(center_changed, (56, 56), 2)

    img = cv2.resize(rotated_img, (56, 56))
    img = np.transpose(img, (2, 0, 1)) / 255.
    img = torch.from_numpy(img.copy()).float()
    img[2] = center_hmap

    # exact (cy, cx) swap from ROI_extraction.py
    return img, raw_img, (int(center[1]), int(center[0]))


# ═══════════════════════════════════════════════════════════════════
#  palmSegmentation.py — process_one_img + segfile loop body
#  Exact copy. Only change: np.intp for np.int0 (numpy>=1.24 compat)
# ═══════════════════════════════════════════════════════════════════

def process_one_img(rawpth, model, kpts, device="cuda"):
    """Exact copy of process_one_img() from palmSegmentation.py."""
    img  = cv2.imread(rawpth)
    if img is None:
        return None
    sx   = rawpth.split("_")[-1][0]
    h, w, _ = img.shape
    if sx == "L":
        if w > h:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img = cv2.flip(img, 1)
    else:
        if w > h:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    imgIPT     = img.copy()
    h, w, _    = img.shape
    inputimg   = cv2.resize(imgIPT, (448, 448))
    indata     = np.transpose(inputimg, (2, 0, 1)) / 255.
    indata     = torch.tensor(indata).float().unsqueeze(0).to(device)
    output     = model(indata)
    res        = output[0].data.cpu().numpy()
    res        = np.expand_dims(res.argmax(axis=0), 2) * 255
    res        = np.repeat(res, 3, 2)
    h, w, _    = img.shape
    o          = np.zeros((h, w * 2, 3))
    o[:, 0:w, :]     = img
    o[:, w:2 * w, :] = cv2.resize(
        res, (w, h), interpolation=cv2.INTER_NEAREST)
    return o


def segment_one_image(rawpth, model, device="cuda"):
    """
    Exact copy of the per-image body inside segfile() from
    palmSegmentation.py.
    np.intp replaces np.int0 (removed in numpy>=1.24, identical behavior).
    """
    o = process_one_img(rawpth, model, kpts=None, device=device)
    if o is None:
        return None

    h, w, _   = o.shape
    w         = int(w / 2)                       # exact: int(w/2)
    p_img     = o[:, :w].astype(np.uint8)
    raw_label = o[:, w:].astype(np.uint8)        # exact: kept as in segfile()
    label     = o[:, w:][:, :, 0].astype(np.uint8)

    contours, _ = cv2.findContours(
        label.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    m = 0
    m_area = 0
    for i in range(len(contours)):               # exact loop from segfile()
        area = cv2.contourArea(contours[i])
        if area > m_area:
            m = i
            m_area = area

    # exact: fresh blank mask + fillPoly = fills all interior holes
    label = np.zeros((p_img.shape[0], w)).astype('uint8')
    label = cv2.fillPoly(label, [contours[m]], 255)

    # exact: COLOR_GRAY2RGB as in segfile()
    label = cv2.cvtColor(label, cv2.COLOR_GRAY2RGB)

    xs, ys = np.intp(np.min(contours[m].reshape(-1, 2), 0))
    xe, ye = np.intp(np.max(contours[m].reshape(-1, 2), 0))

    palm = cv2.bitwise_and(p_img, label)
    palm = palm[ys:ye, xs:xe]

    if palm.size == 0:
        return None
    return palm


# ═══════════════════════════════════════════════════════════════════
#  ROI_extraction.py — process_single_img_ipt
#  Exact copy. Only change: theta.numpy()[0] because checkpoint has
#  numclasses=4 (confirmed: extra.7.weight shape torch.Size([4,128]))
#  so [0] selects the rotation angle — same scalar the original uses
#  when numclasses=1 and theta.numpy() is already length-1.
# ═══════════════════════════════════════════════════════════════════

def process_single_img_ipt(save_dir_visualize, save_dir_square,
                            seg_palm, imgn, detector):
    """
    Exact copy of process_single_img_ipt() from ROI_extraction.py.
    seg_palm: pre-loaded image array (skips file read).
    imgn:     base filename without extension.
    detector: LAnet instance (replaces ThetaPreDetector class).
    """
    tensor_for_ipt, raw_img, r_center = generate_net_ipt(seg_palm, None)
    if tensor_for_ipt is None:
        return 0

    tensor_for_ipt = tensor_for_ipt.unsqueeze(0).to(DEVICE)

    # exact from ROI_extraction.py:
    # ThetaPreDetector.forward returns (1, numclasses)
    # [0] → (numclasses,),  .numpy()[0] → scalar rotation angle
    theta        = detector(tensor_for_ipt).detach().cpu()[0]
    theta        = theta.numpy()[0]
    angle_degrees = np.degrees(theta * np.pi)
    angle_degrees = float(angle_degrees)

    rotation_matrix = cv2.getRotationMatrix2D(r_center, angle_degrees, scale=1.0)
    rawimg = cv2.warpAffine(
        raw_img, rotation_matrix, (raw_img.shape[1], raw_img.shape[0]))
    rawimg   = padding_img(rawimg, 20)
    raw_mask = np.where((rawimg <= 10), 0, 255)[:, :, 2].astype(np.uint8)

    visualize_img           = rawimg.copy()
    final_circle, visualize_circle = circle_better(
        rawimg, visualize_img, raw_mask, rate=1.1)

    # exact from ROI_extraction.py
    imgn_base = imgn.split(".")[0] if "." in imgn else imgn
    h, w, _   = visualize_circle.shape
    visualize_save_iname = imgn_base + ".jpg"
    visualize_circle     = cv2.resize(
        visualize_circle, (int(200 * w / h), 200))
    cv2.imwrite(
        os.path.join(save_dir_visualize, visualize_save_iname),
        visualize_circle)

    # exact augmentation loop from ROI_extraction.py
    saved = 0
    try:
        for angle in range(-30, 30, 3):
            square_roi, circle_roi = get_inter_square(final_circle, angle)
            save_iname             = imgn_base + "_" + str(angle) + ".jpg"
            square_roi_128         = cv2.resize(square_roi, (128, 128))
            cv2.imwrite(
                os.path.join(save_dir_square, save_iname),
                square_roi_128)
            saved += 1
    except Exception as e:
        print(f"  [aug-error] {imgn_base}: {e}")

    return saved


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    for d in [SEG_OUT_DIR, ROI_OUT_DIR, VIS_OUT_DIR]:
        os.makedirs(d, exist_ok=True)

    # ── Load CHSST ───────────────────────────────────────────────
    print("Loading CHSST segmentation model ...")
    from CHSST.models.toptransformer.seaformer import Seaformernet
    seg_model  = Seaformernet()
    state_dict = torch.load(CHSST_CKPT, map_location=DEVICE)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    seg_model.load_state_dict(state_dict, strict=False)
    seg_model.to(DEVICE).eval()
    print("CHSST loaded.\n")

    # ── Load LANet ───────────────────────────────────────────────
    print("Loading LANet ...")
    lanet = LAnet(numclasses=4).to(DEVICE)
    with open(LANET_CKPT, 'rb') as f:
        loaded_params = torch.load(f, map_location=DEVICE)
    lanet.load_state_dict(loaded_params["LANet"])
    lanet.eval()
    print("LANet loaded.\n")

    # ── Collect images ───────────────────────────────────────────
    exts      = {'.jpg', '.jpeg', '.png', '.bmp'}
    raw_imgs  = sorted(
        f for f in os.listdir(MPD_RAW_DIR)
        if os.path.splitext(f)[1].lower() in exts)
    total_imgs  = len(raw_imgs)
    seg_success = seg_fail = roi_success = roi_fail = total_rois = 0

    print(f"Found {total_imgs} images in {MPD_RAW_DIR}\n")

    for idx, img_name in enumerate(raw_imgs[:10], 1):
        raw_pth   = os.path.join(MPD_RAW_DIR, img_name)
        base_name = os.path.splitext(img_name)[0]
        seg_pth   = os.path.join(SEG_OUT_DIR, img_name)

        if idx % 100 == 0 or idx == 1:
            print(f"\n[{idx}/{total_imgs}] "
                  f"SEG ok={seg_success} fail={seg_fail} | "
                  f"ROI ok={roi_success} fail={roi_fail} | "
                  f"ROIs saved={total_rois}")

        # Stage 1: Segmentation
        print(f"  [{idx}/{total_imgs}] SEG  {img_name}", end=" ... ", flush=True)
        try:
            with torch.no_grad():
                palm_seg = segment_one_image(raw_pth, seg_model, DEVICE)
            if palm_seg is None:
                raise ValueError("empty mask")
            cv2.imwrite(seg_pth, palm_seg)
            seg_success += 1
            print("OK")
        except Exception as e:
            print(f"FAIL ({e})")
            seg_fail += 1
            continue

        # Stage 2: ROI extraction
        print(f"  [{idx}/{total_imgs}] ROI  {img_name}", end=" ... ", flush=True)
        try:
            with torch.no_grad():
                n = process_single_img_ipt(
                    VIS_OUT_DIR, ROI_OUT_DIR,
                    palm_seg, base_name, lanet)
            if n == 0:
                raise ValueError("0 ROIs produced")
            roi_success += 1
            total_rois  += n
            print(f"OK ({n} ROIs)")
        except Exception as e:
            print(f"FAIL ({e})")
            roi_fail += 1

    print(f"\n{'─' * 52}")
    print(f"  Total images   : {total_imgs}")
    print(f"  Seg  success   : {seg_success}")
    print(f"  Seg  failures  : {seg_fail}")
    print(f"  ROI  success   : {roi_success}")
    print(f"  ROI  failures  : {roi_fail}")
    print(f"  Total ROIs     : {total_rois}")
    print(f"  Per-image avg  : {total_rois // max(1, roi_success)}")
    print(f"  Seg output     : {SEG_OUT_DIR}")
    print(f"  ROI output     : {ROI_OUT_DIR}")
    print(f"{'─' * 52}")


if __name__ == "__main__":
    main()
