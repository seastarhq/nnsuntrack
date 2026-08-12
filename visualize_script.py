#!/usr/bin/env python
"""Visualize synthetic images in a grid with CNN-predicted centroids.

For each sampled image, runs the TFLite model and overlays a crosshair at
the predicted centroid, plus the confidence value and scene type. Useful
for eyeballing model behavior:
  - Do crosshairs land on the sun disk for clear-sun scenes?
  - Are reflections confidently rejected (low confidence, crosshair greyed)?
  - Do off-frame predictions point the right direction?

Grid dimensions are set by --grid_width (columns) and --grid_height (rows).
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless; save to PNG rather than display
import matplotlib.pyplot as plt

# Match deployment_script's import fallback.
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import ai_edge_litert.interpreter as tflite
    except ImportError:
        from tensorflow import lite as tflite


def infer(interp, image, input_details, output_details,
          input_width, input_height):
    """Run one inference.

    Returns (confidence, cx_px, cy_px, sun_present) in source pixels.
    """
    resized = cv2.resize(image, (input_width, input_height))
    x = resized.astype(np.float32) / 255.0
    x = x[np.newaxis, :, :, np.newaxis]
    interp.set_tensor(input_details['index'], x)
    interp.invoke()
    out = interp.get_tensor(output_details['index'])[0]
    logit, cxn, cyn = float(out[0]), float(out[1]), float(out[2])
    confidence = 1.0 / (1.0 + np.exp(-logit))
    sun_present = 1.0 / (1.0 + np.exp(-float(out[3])))
    h, w = image.shape
    return confidence, cxn * w, cyn * h, sun_present


def draw_crosshair(rgb, cx, cy, color, gap=6, arm=18, thickness=2):
    """Draw a '+' crosshair at (cx, cy).

    A small gap at the center leaves the target pixel visible. cv2 clips
    lines to the image bounds, so partially off-frame crosshairs render
    the arms that fit and the rest disappears.
    """
    cx, cy = int(round(cx)), int(round(cy))
    cv2.line(rgb, (cx - gap - arm, cy), (cx - gap, cy), color, thickness)
    cv2.line(rgb, (cx + gap, cy), (cx + gap + arm, cy), color, thickness)
    cv2.line(rgb, (cx, cy - gap - arm), (cx, cy - gap), color, thickness)
    cv2.line(rgb, (cx, cy + gap), (cx, cy + gap + arm), color, thickness)
    cv2.circle(rgb, (cx, cy), 2, color, -1)


def draw_off_frame_arrow(rgb, cx, cy, color):
    """If the centroid is outside the frame, draw an arrow from the nearest
    edge pointing toward it so the viewer can tell which way the model
    thinks the sun is."""
    h, w = rgb.shape[:2]
    if 0 <= cx < w and 0 <= cy < h:
        return
    tip_x = int(np.clip(cx, 4, w - 5))
    tip_y = int(np.clip(cy, 4, h - 5))
    dx, dy = cx - tip_x, cy - tip_y
    norm = max(np.hypot(dx, dy), 1e-9)
    dx, dy = dx / norm * 40, dy / norm * 40
    start = (int(tip_x - dx), int(tip_y - dy))
    cv2.arrowedLine(rgb, start, (tip_x, tip_y), color, 2, tipLength=0.4)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data_dir', type=str, default='./synthetic_images',
                   help='Dataset directory (contains metadata.csv and PNGs)')
    p.add_argument('--model_path', type=str, default='sun_detector_model.tflite')
    p.add_argument('--grid_width', type=int, default=4,
                   help='Number of columns in the grid')
    p.add_argument('--grid_height', type=int, default=3,
                   help='Number of rows in the grid')
    p.add_argument('--start_index', type=int, default=0,
                   help='Index of the first image to show (ignored if --random)')
    p.add_argument('--random', action='store_true',
                   help='Sample images randomly instead of sequentially')
    p.add_argument('--seed', type=int, default=0,
                   help='Random seed when --random is set')
    p.add_argument('--confidence_threshold', type=float, default=0.5,
                   help='Below this the fix is labelled occluded (the '
                        'crosshair is still drawn)')
    p.add_argument('--presence_threshold', type=float, default=0.5,
                   help='Below this, the crosshair is dimmed grey to show '
                        'the deployment NaN-centroid behavior')
    p.add_argument('--show_truth', action='store_true',
                   help='Also overlay the ground-truth centroid (green)')
    p.add_argument('--output', type=str, default='centroids_grid.png',
                   help='Output PNG path')
    p.add_argument('--num_threads', type=int, default=4)
    p.add_argument('--cell_size', type=float, default=3.5,
                   help='Matplotlib cell width in inches (height auto-scales '
                        'to image aspect ratio)')
    args = p.parse_args()

    df = pd.read_csv(os.path.join(args.data_dir, 'metadata.csv'))
    n_cells = args.grid_width * args.grid_height

    if args.random:
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(
            len(df), size=min(n_cells, len(df)), replace=False).tolist()
    else:
        indices = list(range(
            args.start_index,
            min(args.start_index + n_cells, len(df))))

    # Model.
    interp = tflite.Interpreter(
        model_path=args.model_path, num_threads=args.num_threads)
    interp.allocate_tensors()
    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    _, in_h, in_w, _ = in_d['shape']

    # Figure out source aspect ratio from the first image so the grid
    # cells don't waste vertical space.
    first_img = cv2.imread(
        os.path.join(args.data_dir, df.iloc[indices[0]]['image_filename']), 0)
    src_h, src_w = first_img.shape
    cell_height = args.cell_size * (src_h / src_w)

    fig, axes = plt.subplots(
        args.grid_height, args.grid_width,
        figsize=(args.grid_width * args.cell_size,
                 args.grid_height * cell_height + 0.4),
        squeeze=False)

    PRED_COLOR = (220, 40, 40)      # red
    TRUTH_COLOR = (40, 200, 40)     # green
    DIM_COLOR = (140, 140, 140)     # grey for below-threshold predictions

    for ax_idx, idx in enumerate(indices):
        row, col = divmod(ax_idx, args.grid_width)
        ax = axes[row][col]

        rec = df.iloc[idx]
        img_gray = cv2.imread(
            os.path.join(args.data_dir, rec['image_filename']), 0)
        rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

        confidence, pred_cx, pred_cy, sun_present = infer(
            interp, img_gray, in_d, out_d, in_w, in_h)
        # Presence decides whether there is a fix at all; confidence only
        # says how obscured it is. An occluded sun still gets a crosshair.
        detected = sun_present >= args.presence_threshold
        color = PRED_COLOR if detected else DIM_COLOR

        draw_crosshair(rgb, pred_cx, pred_cy, color)
        draw_off_frame_arrow(rgb, pred_cx, pred_cy, color)

        if args.show_truth and pd.notna(rec['centroid_x']):
            draw_crosshair(rgb, rec['centroid_x'], rec['centroid_y'],
                           TRUTH_COLOR)

        ax.imshow(rgb)
        if not detected:
            status = ' [no sun]'
        elif confidence < args.confidence_threshold:
            status = ' [occluded]'
        else:
            status = ''
        ax.set_title(
            f"#{idx}  p={sun_present:.2f} c={confidence:.2f}{status}\n"
            f"{rec['scene_type']}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

    # Unused cells (when len(indices) < n_cells).
    for ax_idx in range(len(indices), n_cells):
        row, col = divmod(ax_idx, args.grid_width)
        axes[row][col].axis('off')

    legend = "red + = CNN centroid   grey + = presence below threshold"
    if args.show_truth:
        legend += "   green + = ground truth"
    fig.suptitle(legend, fontsize=9, y=0.995)

    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(args.output, dpi=110)
    print(f"Saved {args.output}  "
          f"({args.grid_width}x{args.grid_height} grid, "
          f"{len(indices)} images)")


if __name__ == "__main__":
    main()
