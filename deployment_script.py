#!/usr/bin/env python
"""Sun detection inference on Raspberry Pi 5 via TFLite.

Loads a trained TFLite model and runs it on a single image. Prints the
detection result in a form suitable for manual inspection; the
`run_inference` function is structured to be imported by a ROS2 node that
publishes the result to the sun-position topic that the control loop
consumes.

Output semantics match the training labels:
  confidence: continuous in [0, 1]. 1.0 for a clear sun, lower when the
              sun is occluded (thin cloud, partial blockage), 0.0 when no
              sun is in the frame.
  cx, cy:     pixel coordinates of the sun center in the *source* image.
              May fall outside the image bounds when the sun is partially
              off-frame. Returned as NaN when confidence is below the
              configured threshold - the downstream ROS2 node treats NaN
              as "no fix" and falls back to the ephemeris-predicted
              position.
"""
import numpy as np
import cv2
import argparse

# Prefer the lightweight tflite_runtime package on the RPi5; fall back to
# the Interpreter shipped inside full TensorFlow when running on a
# workstation for local validation. The two APIs are identical for the
# methods we use.
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    from tensorflow import lite as tflite


def load_model(model_path, num_threads=4):
    """Load a TFLite interpreter and allocate tensors."""
    interp = tflite.Interpreter(model_path=model_path, num_threads=num_threads)
    interp.allocate_tensors()
    return interp


def run_inference(interp, image, confidence_threshold):
    """Run one forward pass.

    Args:
        interp: TFLite interpreter returned by `load_model`.
        image:  grayscale uint8 numpy array of any size. It is resized to
                the model's input shape on the fly.
        confidence_threshold: sigmoid-space cutoff. Below this the
                centroid is replaced with NaN.

    Returns (confidence, cx, cy). cx, cy are pixel coordinates in the
    source image frame (not the resized model input).
    """
    source_height, source_width = image.shape

    input_details = interp.get_input_details()[0]
    output_details = interp.get_output_details()[0]
    _, input_height, input_width, _ = input_details['shape']

    # Resize to match the model input. Note: this applies a straight
    # bilinear resize; if the source aspect ratio differs from the aspect
    # ratio the model was trained on, the sun shape is distorted in the
    # input and localization quality will suffer. Match training aspect
    # at deployment time (the default 4:3 matches the 800x600 Arducam).
    resized = cv2.resize(image, (input_width, input_height))
    x = resized.astype(np.float32) / 255.0
    x = x[np.newaxis, :, :, np.newaxis]  # (1, H, W, 1)

    interp.set_tensor(input_details['index'], x)
    interp.invoke()
    output = interp.get_tensor(output_details['index'])[0]  # shape (3,)

    conf_logit, cx_norm, cy_norm = output
    # The model emits a raw logit for confidence; apply sigmoid here (the
    # training loss applies it internally for numerical stability).
    confidence = float(1.0 / (1.0 + np.exp(-conf_logit)))

    if confidence >= confidence_threshold:
        # Centroid is trained as a fractional position in the source image,
        # so we un-normalize by the source dimensions. Values < 0 or > the
        # frame size indicate a sun whose center is off-frame.
        cx = float(cx_norm * source_width)
        cy = float(cy_norm * source_height)
    else:
        cx = float('nan')
        cy = float('nan')

    return confidence, cx, cy


def main():
    parser = argparse.ArgumentParser(
        description="Run sun detection inference using a TFLite model.")
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to input image (grayscale or color; '
                             'will be read as grayscale).')
    parser.add_argument('--model_path', type=str, default='sun_detector_model.tflite',
                        help='Path to TFLite model')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                        help='Minimum confidence to report a centroid. Below this, '
                             'centroid is NaN and the ROS2 control node should use '
                             'the ephemeris-predicted position instead.')
    parser.add_argument('--num_threads', type=int, default=4,
                        help='Number of threads for TFLite inference '
                             '(RPi5 has 4 Cortex-A76 cores).')
    args = parser.parse_args()

    image = cv2.imread(args.image_path, 0)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image_path}")

    interp = load_model(args.model_path, num_threads=args.num_threads)
    confidence, cx, cy = run_inference(interp, image, args.confidence_threshold)

    if np.isnan(cx):
        print(f"confidence={confidence:.3f} (below threshold); no sun detection")
    else:
        print(f"confidence={confidence:.3f} centroid=({cx:.2f}, {cy:.2f}) pixels")


if __name__ == "__main__":
    main()
