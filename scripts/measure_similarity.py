from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, ImageDraw
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.metrics import structural_similarity as ssim


def contain_on_white(img: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    scale = min(target_size[0] / img.width, target_size[1] / img.height)
    new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
    resized = img.resize(new_size, Image.LANCZOS)
    canvas = Image.new("RGB", target_size, (255, 255, 255))
    x = (target_size[0] - new_size[0]) // 2
    y = (target_size[1] - new_size[1]) // 2
    canvas.paste(resized, (x, y))
    return canvas


def foreground_mask(arr: np.ndarray, white_threshold: float = 0.97) -> np.ndarray:
    return np.any(arr < white_threshold, axis=2)


def get_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def bbox_similarity(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int], w: int, h: int) -> float:
    a = np.array([box_a[0] / w, box_a[1] / h, box_a[2] / w, box_a[3] / h], dtype=np.float32)
    b = np.array([box_b[0] / w, box_b[1] / h, box_b[2] / w, box_b[3] / h], dtype=np.float32)
    return float(max(0.0, 1.0 - np.mean(np.abs(a - b))))


def bhattacharyya_hist_similarity(ref: np.ndarray, test: np.ndarray, ref_fg: np.ndarray, test_fg: np.ndarray) -> float:
    sims: list[float] = []
    for c in range(3):
        a = ref[:, :, c][ref_fg]
        b = test[:, :, c][test_fg]
        if len(a) == 0 or len(b) == 0:
            sims.append(0.0)
            continue
        ha, _ = np.histogram(a, bins=32, range=(0, 1), density=True)
        hb, _ = np.histogram(b, bins=32, range=(0, 1), density=True)
        denom = np.sqrt(np.sum(ha) * np.sum(hb))
        sims.append(float(np.sum(np.sqrt(ha * hb)) / denom) if denom > 0 else 0.0)
    return float(np.mean(sims))


def make_overlay(ref_img: Image.Image, test_img: Image.Image, out_path: Path) -> None:
    ref = np.asarray(ref_img).astype(np.float32)
    tst = np.asarray(test_img).astype(np.float32)
    blend = (0.5 * ref + 0.5 * tst).clip(0, 255).astype(np.uint8)
    out = Image.fromarray(blend, mode="RGB")
    draw = ImageDraw.Draw(out)
    draw.text((20, 20), "Reference / Render Overlay", fill=(20, 20, 20))
    out.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--render", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--out-overlay", required=True)
    args = parser.parse_args()

    reference_path = Path(args.reference)
    render_path = Path(args.render)
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_overlay = Path(args.out_overlay)

    ref_img = Image.open(reference_path).convert("RGB")
    render_img_raw = Image.open(render_path).convert("RGB")
    render_img = contain_on_white(render_img_raw, ref_img.size)

    ref = np.asarray(ref_img).astype(np.float32) / 255.0
    tst = np.asarray(render_img).astype(np.float32) / 255.0

    ref_fg = foreground_mask(ref)
    tst_fg = foreground_mask(tst)

    inter = np.logical_and(ref_fg, tst_fg).sum()
    union = np.logical_or(ref_fg, tst_fg).sum()
    silhouette_iou = float(inter / union) if union > 0 else 0.0

    ref_box = get_bbox(ref_fg)
    tst_box = get_bbox(tst_fg)
    box_sim = bbox_similarity(ref_box, tst_box, ref_img.width, ref_img.height)

    hist_sim = bhattacharyya_hist_similarity(ref, tst, ref_fg, tst_fg)

    ref_gray = rgb2gray(ref)
    tst_gray = rgb2gray(tst)
    ssim_score = float(ssim(ref_gray, tst_gray, data_range=1.0))

    ref_edge = canny(ref_gray, sigma=2)
    tst_edge = canny(tst_gray, sigma=2)
    e_union = np.logical_or(ref_edge, tst_edge).sum()
    edge_iou = float(np.logical_and(ref_edge, tst_edge).sum() / e_union) if e_union > 0 else 0.0

    weights = {
        "silhouette_iou": 0.30,
        "bbox_similarity": 0.15,
        "color_hist_similarity": 0.15,
        "ssim": 0.25,
        "edge_iou": 0.15,
    }
    scores = {
        "silhouette_iou": silhouette_iou,
        "bbox_similarity": box_sim,
        "color_hist_similarity": hist_sim,
        "ssim": ssim_score,
        "edge_iou": edge_iou,
    }
    combined = sum(scores[k] * weights[k] for k in weights)

    result = {
        "reference": str(reference_path),
        "render": str(render_path),
        "aligned_render_size": list(ref_img.size),
        "scores": scores,
        "weights": weights,
        "combined_score_0_to_100": round(combined * 100, 4),
        "interpretation": {
            "90_plus": "Very close silhouette and detail match",
            "75_to_89": "Strong resemblance, moderate visible differences",
            "60_to_74": "Moderate resemblance, still clear gaps",
            "below_60": "Large gap from reference",
        },
    }

    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    md = f"""# Similarity Report\n\n- Reference: `{reference_path.name}`\n- Render: `{render_path.name}`\n- Combined score: **{combined*100:.2f} / 100**\n\n## Metric breakdown\n\n| Metric | Score | Weight |\n|---|---:|---:|\n| Silhouette IoU | {silhouette_iou:.4f} | {weights['silhouette_iou']:.2f} |\n| Bounding-box similarity | {box_sim:.4f} | {weights['bbox_similarity']:.2f} |\n| Color histogram similarity | {hist_sim:.4f} | {weights['color_hist_similarity']:.2f} |\n| SSIM | {ssim_score:.4f} | {weights['ssim']:.2f} |\n| Edge IoU | {edge_iou:.4f} | {weights['edge_iou']:.2f} |\n\n## Interpretation\n\n- 90+: very close\n- 75-89: strong resemblance\n- 60-74: moderate resemblance\n- below 60: large gap\n"""
    out_md.write_text(md, encoding="utf-8")
    make_overlay(ref_img, render_img, out_overlay)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
