#!/usr/bin/env python
"""Performance metrics for the sun detector CNN.

Evaluates a trained TFLite model against a labeled synthetic dataset and
reports the metrics that matter for the ROS2 control loop:

  Latency               Inference wall-clock time distribution. The 50 Hz
                        control loop has a 20 ms per-frame budget on the
                        RPi5; we report mean / median / p95 / p99 / max
                        plus the fraction of frames over budget.
  Detection             Binary precision / recall / F1 for "sun present"
                        vs "no sun" at the deployment confidence
                        threshold. The control node falls back to
                        ephemeris on false negatives.
  Confidence            Mean absolute error between predicted and true
  calibration           confidence, with reliability bins.
  Centroid accuracy     Euclidean pixel error on sun-present scenes.
                        Reported in px and in degrees at both 30 deg
                        (narrow) and 140 deg (fisheye) FOV so you can see
                        at a glance whether the 0.1 deg requirement is
                        being met for either camera type.
  Per-scene-type        Same metrics split across the six scene
  breakdown             categories written by synthetic_data_script.py.

Run on labeled synthetic data with known ground truth:

    python metrics_script.py \
        --data_dir ./synthetic_images \
        --model_path sun_detector_model.tflite

For a fair picture of generalization, evaluate against a *fresh* batch
generated with a different --num_images and random seed, not the set the
model was trained on.
"""
import argparse
import os
import time

import cv2
import numpy as np
import pandas as pd

# Match deployment_script's import fallback so the benchmark measures the
# same TFLite path the ROS2 node will use on the RPi5.
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import ai_edge_litert.interpreter as tflite
    except ImportError:
        from tensorflow import lite as tflite


def infer_raw(interp, image, input_details, output_details,
              input_width, input_height):
    """One inference. Returns raw (conf_logit, cx_norm, cy_norm).

    Bypasses deployment_script.run_inference so the NaN-on-low-confidence
    gate doesn't hide prediction error on below-threshold frames - we
    want every prediction for the metrics pass.
    """
    resized = cv2.resize(image, (input_width, input_height))
    x = resized.astype(np.float32) / 255.0
    x = x[np.newaxis, :, :, np.newaxis]
    interp.set_tensor(input_details['index'], x)
    interp.invoke()
    out = interp.get_tensor(output_details['index'])[0]
    return float(out[0]), float(out[1]), float(out[2])


def benchmark_latency(interp, sample_images, warmup, iters):
    """Time end-to-end per-frame inference (resize + normalize + invoke).

    Resize and normalize are counted deliberately: on the RPi5 those
    steps run inside the 20 ms frame budget too.
    """
    input_details = interp.get_input_details()[0]
    output_details = interp.get_output_details()[0]
    _, input_height, input_width, _ = input_details['shape']

    # Warm-up absorbs first-call allocator / thread-pool spin-up.
    for i in range(warmup):
        infer_raw(interp, sample_images[i % len(sample_images)],
                  input_details, output_details, input_width, input_height)

    times_ms = np.zeros(iters)
    for i in range(iters):
        img = sample_images[i % len(sample_images)]
        t0 = time.perf_counter()
        infer_raw(interp, img, input_details, output_details,
                  input_width, input_height)
        times_ms[i] = (time.perf_counter() - t0) * 1000.0
    return times_ms


def report_latency(times_ms, budget_ms):
    print("\n=== Inference latency ===")
    print(f"  samples:  {len(times_ms)}")
    print(f"  mean:     {times_ms.mean():7.2f} ms")
    print(f"  median:   {np.median(times_ms):7.2f} ms")
    print(f"  p95:      {np.percentile(times_ms, 95):7.2f} ms")
    print(f"  p99:      {np.percentile(times_ms, 99):7.2f} ms")
    print(f"  max:      {times_ms.max():7.2f} ms")
    over = int((times_ms > budget_ms).sum())
    pct = 100.0 * over / len(times_ms)
    print(f"  frames over {budget_ms:g} ms budget: {over}/{len(times_ms)} "
          f"({pct:.1f}%)")
    print(f"  steady-state throughput: {1000.0 / times_ms.mean():.1f} Hz")


def report_detection(true_conf, pred_conf, primary_threshold):
    """Binary precision / recall / F1 at several candidate thresholds."""
    # Sun present (clear or occluded) vs not present.
    positive = true_conf > 0

    print("\n=== Binary detection (sun present vs absent) ===")
    print(f"  {'threshold':>10s}  {'TP':>5s} {'FP':>5s} {'FN':>5s} {'TN':>5s}  "
          f"{'precision':>9s} {'recall':>7s} {'F1':>6s}")
    for thr in [0.1, 0.3, primary_threshold, 0.7, 0.9]:
        det = pred_conf >= thr
        tp = int((positive & det).sum())
        fp = int((~positive & det).sum())
        fn = int((positive & ~det).sum())
        tn = int((~positive & ~det).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        marker = "  *" if thr == primary_threshold else "   "
        print(f"{marker}{thr:8.2f}   {tp:5d} {fp:5d} {fn:5d} {tn:5d}  "
              f"{prec:9.3f} {rec:7.3f} {f1:6.3f}")
    print("  * = deployment threshold (see --confidence_threshold)")


def report_confidence_calibration(true_conf, pred_conf):
    """Reliability bins: for each predicted-confidence range, report the
    mean true confidence. A well-calibrated model has mean_true close to
    the bin midpoint."""
    print("\n=== Confidence calibration ===")
    print(f"  mean abs error (all samples): "
          f"{np.abs(true_conf - pred_conf).mean():.4f}")
    print(f"  {'pred range':<16s} {'n':>5s}  {'mean_true':>9s}  "
          f"{'mean_pred':>9s}")
    bin_edges = np.linspace(0, 1, 6)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (pred_conf >= lo) & (pred_conf < hi if hi < 1 else pred_conf <= hi)
        n = int(mask.sum())
        if n == 0:
            print(f"  [{lo:.2f}, {hi:.2f})      {n:5d}  {'-':>9s}  {'-':>9s}")
            continue
        print(f"  [{lo:.2f}, {hi:.2f})      {n:5d}  "
              f"{true_conf[mask].mean():9.3f}  {pred_conf[mask].mean():9.3f}")


def report_centroid_accuracy(err_px, src_width):
    """Centroid error in pixels, and in degrees at both FOV presets."""
    if len(err_px) == 0:
        print("\n=== Centroid accuracy ===\n  (no sun-present scenes)")
        return

    print(f"\n=== Centroid accuracy (sun-present scenes, n={len(err_px)}) ===")
    mae = err_px.mean()
    rmse = np.sqrt((err_px**2).mean())
    med = np.median(err_px)
    p95 = np.percentile(err_px, 95)
    print(f"  MAE:      {mae:7.2f} px")
    print(f"  RMSE:     {rmse:7.2f} px")
    print(f"  median:   {med:7.2f} px")
    print(f"  p95:      {p95:7.2f} px")
    print(f"  max:      {err_px.max():7.2f} px")

    # Degree equivalents + fraction within the 0.1-degree requirement for
    # both camera types this tracker uses.
    print(f"  {'FOV':>8s}  {'MAE deg':>8s}  {'req. px':>8s}  "
          f"{'within 0.1 deg':>15s}")
    for fov in (30.0, 140.0):
        deg_per_px = fov / src_width
        req_px = 0.1 / deg_per_px
        frac = float((err_px <= req_px).mean())
        print(f"  {fov:7.0f}°  {mae * deg_per_px:8.3f}  {req_px:8.2f}  "
              f"{100 * frac:14.1f}%")


def report_per_scene(scene_type, true_conf, pred_conf,
                     err_px_full, has_sun_mask):
    """err_px_full is NaN where has_sun_mask is False so we can index by
    scene type without re-aligning."""
    print("\n=== Per-scene-type breakdown ===")
    print(f"  {'scene_type':<32s} {'n':>5s}  {'conf_MAE':>9s}  "
          f"{'centroid_MAE_px':>16s}")
    for st in sorted(set(scene_type)):
        mask = scene_type == st
        n = int(mask.sum())
        conf_mae = np.abs(true_conf[mask] - pred_conf[mask]).mean()
        sm = mask & has_sun_mask
        if sm.any():
            cen_mae = f"{np.nanmean(err_px_full[sm]):16.2f}"
        else:
            cen_mae = f"{'-':>16s}"
        print(f"  {st:<32s} {n:5d}  {conf_mae:9.3f}  {cen_mae}")


def main():
    p = argparse.ArgumentParser(
        description="Performance metrics for the sun detector CNN.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data_dir', type=str, default='./synthetic_images',
                   help='Labeled dataset directory (contains metadata.csv)')
    p.add_argument('--model_path', type=str,
                   default='sun_detector_model.tflite',
                   help='Path to the TFLite model to evaluate')
    p.add_argument('--confidence_threshold', type=float, default=0.5,
                   help='Primary threshold for detection metrics '
                        '(should match deployment_script)')
    p.add_argument('--budget_ms', type=float, default=20.0,
                   help='Per-frame latency budget (50 Hz = 20 ms)')
    p.add_argument('--num_threads', type=int, default=4,
                   help='TFLite interpreter threads (RPi5 has 4 cores)')
    p.add_argument('--latency_iters', type=int, default=500,
                   help='Number of timed inferences for the latency test')
    p.add_argument('--latency_warmup', type=int, default=20,
                   help='Warm-up iterations before timing starts')
    p.add_argument('--latency_sample_size', type=int, default=16,
                   help='How many dataset images to cycle through during '
                        'the latency test (content does not affect compute)')
    args = p.parse_args()

    # --- Load dataset metadata ---
    df = pd.read_csv(os.path.join(args.data_dir, 'metadata.csv'))
    first = cv2.imread(
        os.path.join(args.data_dir, df.iloc[0]['image_filename']), 0)
    src_h, src_w = first.shape
    print(f"Dataset:    {len(df)} images at {src_w}x{src_h}  "
          f"({args.data_dir})")
    print(f"Model:      {args.model_path}  "
          f"(threads={args.num_threads})")

    # --- Load model ---
    interp = tflite.Interpreter(
        model_path=args.model_path, num_threads=args.num_threads)
    interp.allocate_tensors()
    _, in_h, in_w, _ = interp.get_input_details()[0]['shape']
    print(f"Model input: {in_w}x{in_h}")

    # --- Latency ---
    sample_count = min(args.latency_sample_size, len(df))
    sample_images = [cv2.imread(
        os.path.join(args.data_dir, df.iloc[i]['image_filename']), 0)
        for i in range(sample_count)]
    times_ms = benchmark_latency(interp, sample_images,
                                 warmup=args.latency_warmup,
                                 iters=args.latency_iters)
    report_latency(times_ms, args.budget_ms)

    # --- Accuracy pass over every labeled image ---
    input_details = interp.get_input_details()[0]
    output_details = interp.get_output_details()[0]

    n = len(df)
    pred_conf = np.zeros(n, dtype=np.float32)
    pred_cx = np.zeros(n, dtype=np.float32)
    pred_cy = np.zeros(n, dtype=np.float32)

    for i, fn in enumerate(df['image_filename']):
        img = cv2.imread(os.path.join(args.data_dir, fn), 0)
        logit, cxn, cyn = infer_raw(interp, img, input_details,
                                     output_details, in_w, in_h)
        # Numerically stable sigmoid.
        pred_conf[i] = 1.0 / (1.0 + np.exp(-logit))
        pred_cx[i] = cxn * src_w
        pred_cy[i] = cyn * src_h

    true_conf = df['confidence'].to_numpy(dtype=np.float32)
    true_cx = df['centroid_x'].to_numpy(dtype=np.float32)
    true_cy = df['centroid_y'].to_numpy(dtype=np.float32)
    scene_type = df['scene_type'].to_numpy()

    # --- Detection ---
    report_detection(true_conf, pred_conf, args.confidence_threshold)

    # --- Confidence calibration ---
    report_confidence_calibration(true_conf, pred_conf)

    # --- Centroid accuracy ---
    has_sun_mask = ~np.isnan(true_cx)
    err_px_full = np.full(n, np.nan, dtype=np.float32)
    err_px_full[has_sun_mask] = np.sqrt(
        (pred_cx[has_sun_mask] - true_cx[has_sun_mask])**2 +
        (pred_cy[has_sun_mask] - true_cy[has_sun_mask])**2)
    report_centroid_accuracy(err_px_full[has_sun_mask], src_w)

    # --- In-frame only centroid accuracy ---
    in_frame_mask = (has_sun_mask
                     & (true_cx >= 0) & (true_cx <= src_w)
                     & (true_cy >= 0) & (true_cy <= src_h))
    n_in = int(in_frame_mask.sum())
    n_off = int(has_sun_mask.sum()) - n_in
    print(f"\n=== In-frame centroid accuracy "
          f"(n={n_in}, excluding {n_off} off-frame) ===")
    if n_in > 0:
        err_in = err_px_full[in_frame_mask]
        print(f"  MAE:      {err_in.mean():7.2f} px")
        print(f"  RMSE:     {np.sqrt((err_in**2).mean()):7.2f} px")
        print(f"  median:   {np.median(err_in):7.2f} px")
        print(f"  p95:      {np.percentile(err_in, 95):7.2f} px")
        print(f"  max:      {err_in.max():7.2f} px")
        print(f"  {'FOV':>8s}  {'MAE deg':>8s}  {'req. px':>8s}  "
              f"{'within 0.1 deg':>15s}")
        for fov in (30.0, 140.0):
            deg_per_px = fov / src_w
            req_px = 0.1 / deg_per_px
            frac = float((err_in <= req_px).mean())
            print(f"  {fov:7.0f}°  {err_in.mean() * deg_per_px:8.3f}  "
                  f"{req_px:8.2f}  {100 * frac:14.1f}%")

    # --- Clean-sun only accuracy (sun_only scenes, in-frame) ---
    clean_mask = in_frame_mask & (scene_type == 'sun_only')
    n_clean = int(clean_mask.sum())
    print(f"\n=== Clean-sun only accuracy "
          f"(sun_only + in-frame, n={n_clean}) ===")
    if n_clean > 0:
        err_clean = err_px_full[clean_mask]
        conf_clean_mae = np.abs(true_conf[clean_mask]
                                - pred_conf[clean_mask]).mean()
        print(f"  conf MAE: {conf_clean_mae:7.4f}")
        print(f"  MAE:      {err_clean.mean():7.2f} px")
        print(f"  RMSE:     {np.sqrt((err_clean**2).mean()):7.2f} px")
        print(f"  median:   {np.median(err_clean):7.2f} px")
        print(f"  p95:      {np.percentile(err_clean, 95):7.2f} px")
        print(f"  max:      {err_clean.max():7.2f} px")
        print(f"  {'FOV':>8s}  {'MAE deg':>8s}  {'req. px':>8s}  "
              f"{'within 0.1 deg':>15s}")
        for fov in (30.0, 140.0):
            deg_per_px = fov / src_w
            req_px = 0.1 / deg_per_px
            frac = float((err_clean <= req_px).mean())
            print(f"  {fov:7.0f}°  {err_clean.mean() * deg_per_px:8.3f}  "
                  f"{req_px:8.2f}  {100 * frac:14.1f}%")

    # --- Per-scene-type breakdown ---
    report_per_scene(scene_type, true_conf, pred_conf,
                     err_px_full, has_sun_mask)


if __name__ == "__main__":
    main()
