#!/usr/bin/env python
import pandas as pd
import numpy as np
import os
import cv2
import argparse
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, Dense,
                                      GlobalAveragePooling2D, Concatenate,
                                      Reshape, Activation, BatchNormalization,
                                      Lambda, UpSampling2D)
from sklearn.model_selection import train_test_split


def create_model(input_shape):
    """Compact CNN with high-resolution spatial soft-argmax for sub-pixel
    centroid accuracy.

    3 linear outputs: [confidence_logit, cx_norm, cy_norm]. Sigmoid is applied
    to the confidence channel in the loss (and at deployment). The two
    centroid outputs are unbounded so they can represent a sun whose center
    lies outside the image frame (normalized values < 0 or > 1).

    Design notes:
    - Stride-2 convs with BatchNorm do the downsampling. BN stabilizes
      training and is fused into the preceding conv at TFLite export
      (zero runtime cost). use_bias=False because BN's beta subsumes it.
    - Feature depth grows as spatial dims shrink: 16 → 24 → 32 → 32.
    - Two output heads share the feature trunk:
        confidence: global average pool + linear (position-invariant).
        centroid:   the centroid head branches from the 2nd conv block
                    (48x64 grid, ~12.5px cells) for high spatial resolution.
                    Deep features from the 4th block are upsampled and
                    concatenated so the head has both fine spatial detail
                    and semantic context. A 1x1 conv produces a heatmap,
                    a learnable temperature sharpens the softmax peak,
                    and the expected (x, y) coordinate is computed via
                    dot product with a fixed coordinate grid. A Dense(2)
                    affine layer stretches the output range for off-frame
                    centroids.
    - With a 256x192 input this is ~22K params and ~22M MACs.
    """
    input_height, input_width = input_shape[0], input_shape[1]
    cen_h = input_height // 4   # centroid head at 2nd block: stride 4
    cen_w = input_width // 4

    inputs = Input(shape=input_shape)

    # Shared feature extractor: four stride-2 conv blocks with BatchNorm.
    # H x W reductions for a 192 x 256 input: 96x128, 48x64, 24x32, 12x16.
    x = Conv2D(16, 3, strides=2, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(24, 3, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    shallow = x  # (B, 48, 64, 24) — high-res features for centroid head

    x = Conv2D(32, 3, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(32, 3, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    deep = x  # (B, 12, 16, 32) — semantic features

    # Confidence head (position-invariant) — uses full-depth features.
    conf = GlobalAveragePooling2D()(deep)
    conf = Dense(1, activation='linear')(conf)

    # Centroid head: multi-scale spatial soft-argmax.
    # Upsample deep features to match shallow resolution, then concat.
    deep_up = UpSampling2D(size=(4, 4), interpolation='bilinear')(deep)
    fused = Concatenate()([shallow, deep_up])  # (B, 48, 64, 24+32=56)

    # 1. Refinement conv to mix spatial + semantic info, then heatmap.
    fused = Conv2D(32, 3, padding='same', use_bias=False)(fused)
    fused = BatchNormalization()(fused)
    fused = Activation('relu')(fused)
    cen = Conv2D(1, 1, activation='linear')(fused)       # (B, cen_h, cen_w, 1)
    cen = Reshape((cen_h * cen_w,))(cen)                 # (B, cen_h*cen_w)

    # 2. Learnable temperature: sharper softmax → more precise peak.
    #    Initialized to 1.0 (neutral); the model learns to decrease it.
    log_temperature = tf.Variable(0.0, trainable=True, name='log_temperature')

    def tempered_softmax(logits):
        temperature = tf.exp(log_temperature)
        return tf.nn.softmax(logits / temperature, axis=1)

    cen = Lambda(tempered_softmax)(cen)                  # spatial probability

    # 3. Compute expected coordinates via dot product with a fixed grid.
    #    Cell-centered coordinates in [0, 1].
    gy = np.linspace(0.5 / cen_h, 1.0 - 0.5 / cen_h, cen_h)
    gx = np.linspace(0.5 / cen_w, 1.0 - 0.5 / cen_w, cen_w)
    grid_y, grid_x = np.meshgrid(gy, gx, indexing='ij')
    gx_flat = grid_x.reshape(-1).astype(np.float32)      # (cen_h*cen_w,)
    gy_flat = grid_y.reshape(-1).astype(np.float32)

    def soft_argmax(probs):
        cx = tf.reduce_sum(probs * tf.constant(gx_flat), axis=1, keepdims=True)
        cy = tf.reduce_sum(probs * tf.constant(gy_flat), axis=1, keepdims=True)
        return tf.concat([cx, cy], axis=1)

    cen = Lambda(soft_argmax)(cen)                        # (B, 2)

    # 4. Affine layer to allow unbounded output for off-frame suns.
    #    Initialized as identity so it starts as pass-through.
    cen = Dense(2, activation='linear',
                kernel_initializer='identity',
                bias_initializer='zeros')(cen)

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

    Centroid uses Huber (smooth L1) loss with delta=0.02 in normalized
    coords (~16 px at 800 px width). This is more robust than MSE to
    off-frame outliers while still penalizing small errors quadratically.
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

    LAMBDA_CENTROID = 20.0
    HUBER_DELTA = 0.1
    huber_cx = tf.keras.losses.huber(cx_true, cx_pred, delta=HUBER_DELTA) * has_sun
    huber_cy = tf.keras.losses.huber(cy_true, cy_pred, delta=HUBER_DELTA) * has_sun

    return tf.reduce_mean(bce + LAMBDA_CENTROID * (huber_cx + huber_cy))


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
    parser.add_argument('--epochs', type=int, default=50)
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

    steps_per_epoch = len(X_train) // args.batch_size
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-3,
        decay_steps=args.epochs * steps_per_epoch,
        alpha=1e-5)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(optimizer=optimizer, loss=custom_loss)
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
