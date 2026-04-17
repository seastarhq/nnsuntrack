#!/usr/bin/env python
import pandas as pd
import numpy as np
import os
import cv2
import argparse
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, Flatten, Dense,
                                      GlobalAveragePooling2D, Concatenate)
from sklearn.model_selection import train_test_split


def create_model(input_shape):
    """Compact CNN sized for real-time inference on a Raspberry Pi 5.

    3 linear outputs: [confidence_logit, cx_norm, cy_norm]. Sigmoid is applied
    to the confidence channel in the loss (and at deployment). The two
    centroid outputs are unbounded so they can represent a sun whose center
    lies outside the image frame (normalized values < 0 or > 1).

    Design notes:
    - Stride-2 convs do the downsampling (cheaper than conv+maxpool and
      trimmed of the redundant full-resolution conv at each stage).
    - Feature depth grows as spatial dims shrink: 16 → 24 → 32 → 32.
    - Two output heads share the feature trunk:
        confidence: global average pool + linear. Spatial position doesn't
                    matter for "is there a sun in frame", so GAP is the right
                    aggregator - and it costs almost nothing.
        centroid:   1x1 conv squeezes 32 channels to 8, then flatten + two
                    dense layers. Position information is preserved through
                    the flatten so the regression head can read feature
                    activation at specific spatial locations, which is what
                    sub-pixel centroid accuracy needs.
    - With a 256x192 input this is ~70K params and ~20M MACs, vs. the
      previous ~25M params / ~268M MACs.
    """
    inputs = Input(shape=input_shape)

    # Shared feature extractor: four stride-2 conv blocks.
    # H x W reductions for a 192 x 256 input: 96x128, 48x64, 24x32, 12x16.
    x = Conv2D(16, 3, strides=2, padding='same', activation='relu')(inputs)
    x = Conv2D(24, 3, strides=2, padding='same', activation='relu')(x)
    x = Conv2D(32, 3, strides=2, padding='same', activation='relu')(x)
    x = Conv2D(32, 3, strides=2, padding='same', activation='relu')(x)

    # Confidence head (position-invariant).
    conf = GlobalAveragePooling2D()(x)
    conf = Dense(1, activation='linear')(conf)

    # Centroid head (position-sensitive). Channel-reduce to 8 so the flatten
    # is cheap (12*16*8 = 1536 features at the default 256x192 input).
    cen = Conv2D(8, 1, activation='relu')(x)
    cen = Flatten()(cen)
    cen = Dense(32, activation='relu')(cen)
    cen = Dense(2, activation='linear')(cen)

    outputs = Concatenate()([conf, cen])
    return Model(inputs=inputs, outputs=outputs)


def custom_loss(y_true, y_pred):
    """Continuous-confidence + centroid regression loss.

    y_true has 4 values per sample: [confidence, cx_norm, cy_norm, has_sun].
    The 4th channel is a mask: 1.0 when a sun is present in the scene
    (confidence > 0) and 0.0 otherwise. It gates the centroid loss so that
    no-sun scenes (which carry placeholder centroid zeros, not real labels)
    do not pull the centroid head toward (0, 0).

    The confidence target is continuous in [0, 1] - 1.0 for a clear sun,
    the occlusion factor for cloudy / partially occluded sun, 0.0 for no
    sun. Binary cross-entropy extends naturally to these soft targets and
    is better calibrated than MSE for probability-like outputs.
    """
    conf_true = y_true[:, 0]
    cx_true = y_true[:, 1]
    cy_true = y_true[:, 2]
    has_sun = y_true[:, 3]

    conf_pred_logit = y_pred[:, 0]
    cx_pred = y_pred[:, 1]
    cy_pred = y_pred[:, 2]

    conf_pred = tf.sigmoid(conf_pred_logit)
    bce = tf.keras.losses.binary_crossentropy(conf_true, conf_pred)

    mse_cx = tf.square(cx_true - cx_pred) * has_sun
    mse_cy = tf.square(cy_true - cy_pred) * has_sun

    return tf.reduce_mean(bce + mse_cx + mse_cy)


def load_dataset(data_dir, input_width, input_height):
    """Load images + labels from `data_dir` (expects metadata.csv and PNGs).

    Images are resized to (input_width, input_height) and normalized to
    [0, 1]. Centroid labels are normalized by the *original* image
    dimensions (inferred from the first image), so the model's centroid
    outputs are invariant to the training input resolution. Off-frame
    centroids produce normalized values outside [0, 1], which the
    unbounded linear head can represent.
    """
    df = pd.read_csv(os.path.join(data_dir, 'metadata.csv'))
    first = cv2.imread(os.path.join(data_dir, df.iloc[0]['image_filename']), 0)
    orig_height, orig_width = first.shape

    n = len(df)
    X = np.zeros((n, input_height, input_width, 1), dtype=np.float32)
    y = np.zeros((n, 4), dtype=np.float32)

    for i, row in df.iterrows():
        img = cv2.imread(os.path.join(data_dir, row['image_filename']), 0)
        img = cv2.resize(img, (input_width, input_height))
        X[i, :, :, 0] = img.astype(np.float32) / 255.0

        y[i, 0] = row['confidence']
        if pd.notna(row['centroid_x']):
            y[i, 1] = row['centroid_x'] / orig_width
            y[i, 2] = row['centroid_y'] / orig_height
            y[i, 3] = 1.0
        # else: leave centroid zeros and has_sun=0 to mask in the loss

    return X, y, (orig_width, orig_height)


def main():
    parser = argparse.ArgumentParser(description="Train CNN for sun detection.")
    parser.add_argument('--data_dir', type=str, default='./synthetic_images',
                        help='Directory with synthetic images and metadata.csv')
    parser.add_argument('--input_width', type=int, default=256,
                        help='Width to which images are resized before the CNN')
    parser.add_argument('--input_height', type=int, default=192,
                        help='Height to which images are resized before the CNN')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--validation_split', type=float, default=0.2)
    parser.add_argument('--output_model', type=str, default='sun_detector_model.keras',
                        help='Path to save the trained Keras model (native .keras format)')
    parser.add_argument('--output_tflite', type=str, default='sun_detector_model.tflite',
                        help='Path to save the TFLite model for RPi5 deployment')
    args = parser.parse_args()

    X, y, (orig_width, orig_height) = load_dataset(
        args.data_dir, args.input_width, args.input_height)
    print(f"Loaded {len(X)} samples. Source resolution: {orig_width}x{orig_height}. "
          f"Training input: {args.input_width}x{args.input_height}.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.validation_split, random_state=42)

    model = create_model((args.input_height, args.input_width, 1))
    model.summary()
    model.compile(optimizer='adam', loss=custom_loss)
    model.fit(X_train, y_train,
              validation_data=(X_test, y_test),
              epochs=args.epochs,
              batch_size=args.batch_size)

    model.save(args.output_model)

    # TFLite conversion: wrap the model in a concrete tf.function and convert
    # from that. `from_keras_model` tracing is broken under Keras 3, and the
    # SavedModel-based route trips a Python 3.14 bug in TrackableView.
    input_spec = tf.TensorSpec(
        shape=[1, args.input_height, args.input_width, 1], dtype=tf.float32)

    @tf.function(input_signature=[input_spec])
    def serve(x):
        return model(x)

    concrete = serve.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], model)
    tflite_model = converter.convert()
    with open(args.output_tflite, 'wb') as f:
        f.write(tflite_model)


if __name__ == "__main__":
    main()
