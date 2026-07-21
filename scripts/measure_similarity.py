from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.metrics import structural_similarity as ssim


def contain_on_white(img: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    scale = min(target_size[0] / img.width, target_size[1] / img.height)
    new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
    resized = img.resize(new_size, Image.LANCZOS)
    canvas = Image.new('RGB', target_size, (255, 255, 255))
    x = (target_size[0] - new_size[0]) // 2
    y = (target_size[1] - new_size[1]) // 2
    canvas.paste(resized, (x, y))
    return canvas


def parse_ignore_rect(value: str) -> tuple[int, int, int, int]:
    parts = [int(v.strip()) for v in value.split(',')]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError('ignore-rect must be x1,y1,x2,y2')
    return tuple(parts)


def foreground_mask(arr: np.ndarray, white_threshold: float = 0.97) -> np.ndarray:
    return np.any(arr < white_threshold, axis=2)


def apply_ignore(mask: np.ndarray, rects: list[tuple[int, int, int, int]]) -> np.ndarray:
    mask = mask.copy()
    h, w = mask.shape
    for x1, y1, x2, y2 in rects:
        x1 = max(0, min(w, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h, y1))
        y2 = max(0, min(h, y2))
        mask[y1:y2, x1:x2] = False
    return mask


def bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def bbox_similarity(a: tuple[int, int, int, int], b: tuple[int, int, int, int], w: int, h: int) -> float:
    av = np.array([a[0]/w, a[1]/h, a[2]/w, a[3]/h], dtype=np.float32)
    bv = np.array([b[0]/w, b[1]/h, b[2]/w, b[3]/h], dtype=np.float32)
    return float(max(0.0, 1.0 - np.mean(np.abs(av - bv))))


def hist_similarity(ref: np.ndarray, tst: np.ndarray, ref_fg: np.ndarray, tst_fg: np.ndarray) -> float:
    vals = []
    for c in range(3):
        a = ref[:, :, c][ref_fg]
        b = tst[:, :, c][tst_fg]
        if len(a) == 0 or len(b) == 0:
            vals.append(0.0)
            continue
        ha, _ = np.histogram(a, bins=32, range=(0, 1), density=True)
        hb, _ = np.histogram(b, bins=32, range=(0, 1), density=True)
        denom = np.sqrt(np.sum(ha) * np.sum(hb))
        vals.append(float(np.sum(np.sqrt(ha * hb)) / denom) if denom > 0 else 0.0)
    return float(np.mean(vals))


def make_overlay(ref_img: Image.Image, tst_img: Image.Image, rects: list[tuple[int,int,int,int]], out_path: Path) -> None:
    ref = np.asarray(ref_img).astype(np.float32)
    tst = np.asarray(tst_img).astype(np.float32)
    blend = (0.5 * ref + 0.5 * tst).clip(0,255).astype(np.uint8)
    out = Image.fromarray(blend)
    draw = ImageDraw.Draw(out)
    draw.text((20,20), 'Reference / Render Overlay v2', fill=(30,30,30))
    for rect in rects:
        draw.rectangle(rect, outline=(255,0,0), width=2)
    out.save(out_path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--reference', required=True)
    p.add_argument('--render', required=True)
    p.add_argument('--out-json', required=True)
    p.add_argument('--out-md', required=True)
    p.add_argument('--out-overlay', required=True)
    p.add_argument('--ignore-rect', action='append', type=parse_ignore_rect, default=[])
    args = p.parse_args()

    ref_img = Image.open(args.reference).convert('RGB')
    render_img = contain_on_white(Image.open(args.render).convert('RGB'), ref_img.size)

    ref = np.asarray(ref_img).astype(np.float32)/255.0
    tst = np.asarray(render_img).astype(np.float32)/255.0

    ref_fg = apply_ignore(foreground_mask(ref), args.ignore_rect)
    tst_fg = apply_ignore(foreground_mask(tst), args.ignore_rect)

    inter = np.logical_and(ref_fg, tst_fg).sum()
    union = np.logical_or(ref_fg, tst_fg).sum()
    silhouette_iou = float(inter/union) if union > 0 else 0.0

    ref_box = bbox(ref_fg)
    tst_box = bbox(tst_fg)
    box_sim = bbox_similarity(ref_box, tst_box, ref_img.width, ref_img.height)

    hsim = hist_similarity(ref, tst, ref_fg, tst_fg)

    ref_gray = rgb2gray(ref)
    tst_gray = rgb2gray(tst)
    ssim_score = float(ssim(ref_gray, tst_gray, data_range=1.0))

    ref_edge = apply_ignore(canny(ref_gray, sigma=2), args.ignore_rect)
    tst_edge = apply_ignore(canny(tst_gray, sigma=2), args.ignore_rect)
    e_union = np.logical_or(ref_edge, tst_edge).sum()
    edge_iou = float(np.logical_and(ref_edge, tst_edge).sum()/e_union) if e_union > 0 else 0.0

    weights = {
        'silhouette_iou': 0.30,
        'bbox_similarity': 0.15,
        'color_hist_similarity': 0.15,
        'ssim': 0.25,
        'edge_iou': 0.15,
    }
    scores = {
        'silhouette_iou': silhouette_iou,
        'bbox_similarity': box_sim,
        'color_hist_similarity': hsim,
        'ssim': ssim_score,
        'edge_iou': edge_iou,
    }
    combined = sum(scores[k] * weights[k] for k in weights)

    result = {
        'reference': args.reference,
        'render': args.render,
        'ignore_rects': args.ignore_rect,
        'aligned_render_size': list(ref_img.size),
        'scores': scores,
        'weights': weights,
        'combined_score_0_to_100': round(combined*100, 4),
        'interpretation': {
            '90_plus': 'Very close silhouette and detail match',
            '75_to_89': 'Strong resemblance, moderate visible differences',
            '60_to_74': 'Moderate resemblance, still clear gaps',
            'below_60': 'Large gap from reference',
        }
    }
    Path(args.out_json).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    md = f'''# Similarity Report v2\n\n- Reference: `{Path(args.reference).name}`\n- Render: `{Path(args.render).name}`\n- Combined score: **{combined*100:.2f} / 100**\n- Ignore rects: `{args.ignore_rect}`\n\n## Metric breakdown\n\n| Metric | Score | Weight |\n|---|---:|---:|\n| Silhouette IoU | {silhouette_iou:.4f} | {weights['silhouette_iou']:.2f} |\n| Bounding-box similarity | {box_sim:.4f} | {weights['bbox_similarity']:.2f} |\n| Color histogram similarity | {hsim:.4f} | {weights['color_hist_similarity']:.2f} |\n| SSIM | {ssim_score:.4f} | {weights['ssim']:.2f} |\n| Edge IoU | {edge_iou:.4f} | {weights['edge_iou']:.2f} |\n'''
    Path(args.out_md).write_text(md, encoding='utf-8')
    make_overlay(ref_img, render_img, args.ignore_rect, Path(args.out_overlay))
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
