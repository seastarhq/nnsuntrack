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
    """Compact CNN with coarse-to-fine spatial localization for sub-pixel
    centroid accuracy.

    4 linear outputs: [confidence_logit, cx_norm, cy_norm, presence_logit].
    Sigmoid is applied to the confidence and presence channels in the loss
    (and at deployment).

    Confidence and presence answer different questions, which is why both
    exist. Confidence is how much of the sun is getting through (1.0 clear,
    the transmission fraction when a cloud is over it). Presence is simply
    whether a sun is in the frame at all. A sun fully hidden behind a thick
    cloud edge has confidence near 0 but presence 1 - and its position is
    still known, so the tracker should keep following it rather than
    dropping to ephemeris.

    Design notes:
    - Stride-2 convs with BatchNorm do the downsampling. BN stabilizes
      training and is fused into the preceding conv at TFLite export
      (zero runtime cost). use_bias=False because BN's beta subsumes it.
    - Feature depth grows as spatial dims shrink: 16 → 24 → 32 → 32.
    - Two output heads share the feature trunk:
        confidence: global average pool + linear (position-invariant).
        centroid:   coarse-to-fine localization. The centroid head branches
                    from the 2nd conv block (60x80 grid at 320x240 input,
                    ~10px cells) for high spatial resolution. Deep features
                    from the 4th block are upsampled and concatenated, then
                    a 3x3 refinement conv fuses spatial + semantic info.
                    A tempered soft-argmax produces a coarse position, then
                    a context-aware offset MLP (GAP of fused features +
                    coarse position → Dense(16) → Dense(2)) predicts a
                    sub-grid-cell residual for fine adjustment.
    - With a 320x240 input this is ~25K params and ~35M MACs.
    """
    input_height, input_width = input_shape[0], input_shape[1]
    cen_h = input_height // 4   # centroid head at 2nd block: stride 4
    cen_w = input_width // 4

    inputs = Input(shape=input_shape)

    # Shared feature extractor: four stride-2 conv blocks with BatchNorm.
    # H x W reductions for a 240 x 320 input: 120x160, 60x80, 30x40, 15x20.
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

    # Confidence and presence heads (both position-invariant) share one
    # pooling of the full-depth features.
    pooled = GlobalAveragePooling2D()(deep)
    conf = Dense(1, activation='linear')(pooled)
    presence = Dense(1, activation='linear')(pooled)

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

    # 4. Coarse-to-fine offset refinement: the soft-argmax gives a coarse
    #    position limited by grid resolution. A small MLP takes the coarse
    #    (cx, cy) plus global context from the fused feature map and
    #    predicts a sub-grid-cell (dx, dy) residual. Initialized to zero
    #    so training starts from the coarse estimate.
    cen_context = GlobalAveragePooling2D()(fused)          # (B, 32)
    cen_input = Concatenate()([cen, cen_context])          # (B, 34)
    offset = Dense(16, activation='relu')(cen_input)
    offset = Dense(2, activation='linear',
                   kernel_initializer='zeros',
                   bias_initializer='zeros')(offset)       # (B, 2)
    cen = Lambda(lambda args: args[0] + args[1])([cen, offset])

    outputs = Concatenate()([conf, cen, presence])
    return Model(inputs=inputs, outputs=outputs)


def custom_loss(y_true, y_pred):
    """Continuous-confidence + centroid regression loss.

    y_true has 4 values per sample: [confidence, cx_norm, cy_norm, sun_present].

    The 4th channel does double duty. It gates the centroid loss, so no-sun
    scenes (which carry placeholder centroid zeros, not real labels) do not
    pull the centroid head toward (0, 0) - and it is the target for the
    presence output. Those are the same quantity: a sun is present exactly
    when the scene has a real centroid to regress toward, including when
    the sun is fully occluded.

    The confidence target is continuous in [0, 1] - 1.0 for a clear sun,
    the occlusion factor for cloudy / partially occluded sun, 0.0 for no
    sun. Binary cross-entropy extends naturally to these soft targets and
    is better calibrated than MSE for probability-like outputs.

    Centroid uses Huber (smooth L1) loss with delta=0.1 in normalized
    coords (~80 px at 800 px width). Errors within delta are penalized
    quadratically; larger errors get linear gradients to limit the
    influence of extreme outliers.
    """
    conf_true = y_true[:, 0]
    cx_true = y_true[:, 1]
    cy_true = y_true[:, 2]
    has_sun = y_true[:, 3]

    cx_pred = y_pred[:, 1]
    cy_pred = y_pred[:, 2]

    conf_pred = tf.sigmoid(y_pred[:, 0])
    bce = tf.keras.losses.binary_crossentropy(conf_true, conf_pred)

    presence_pred = tf.sigmoid(y_pred[:, 3])
    bce_presence = tf.keras.losses.binary_crossentropy(has_sun, presence_pred)

    LAMBDA_CENTROID = 20.0
    LAMBDA_PRESENCE = 1.0
    HUBER_DELTA = 0.1
    huber_cx = tf.keras.losses.huber(cx_true, cx_pred, delta=HUBER_DELTA) * has_sun
    huber_cy = tf.keras.losses.huber(cy_true, cy_pred, delta=HUBER_DELTA) * has_sun

    return tf.reduce_mean(bce + LAMBDA_PRESENCE * bce_presence
                          + LAMBDA_CENTROID * (huber_cx + huber_cy))


def load_dataset(data_dir, input_width, input_height, normalize=False):
    """Load images + labels from `data_dir` (expects metadata.csv and PNGs).

    Images are resized to (input_width, input_height). With `normalize`
    they come back as float32 in [0, 1], ready to hand straight to
    `model.fit`; otherwise they stay uint8 and `make_dataset` does the
    scaling one batch at a time. A 20K-image set is 1.5 GB as uint8
    versus 6.1 GB as float32, and the whole set is held in host RAM at
    once, so the uint8 form is the default.

    Centroid labels are normalized by the *original* image dimensions
    (inferred from the first image), so the model's centroid outputs are
    invariant to the training input resolution.

    Off-frame suns (normalized centroid outside [0, 1]) have their
    centroid targets clamped to the nearest frame edge. This teaches the
    model to point toward the correct edge when the sun is just outside
    the frame — the right behavior for the tracker's control loop.
    """
    df = pd.read_csv(os.path.join(data_dir, 'metadata.csv'))
    first = cv2.imread(os.path.join(data_dir, df.iloc[0]['image_filename']), 0)
    orig_height, orig_width = first.shape

    n = len(df)
    n_clamped = 0
    has_presence = 'sun_present' in df.columns
    X = np.zeros((n, input_height, input_width, 1),
                 dtype=np.float32 if normalize else np.uint8)
    y = np.zeros((n, 4), dtype=np.float32)

    for i, row in df.iterrows():
        img = cv2.imread(os.path.join(data_dir, row['image_filename']), 0)
        img = cv2.resize(img, (input_width, input_height))
        X[i, :, :, 0] = img.astype(np.float32) / 255.0 if normalize else img

        y[i, 0] = row['confidence']
        if pd.notna(row['centroid_x']):
            cx_norm = row['centroid_x'] / orig_width
            cy_norm = row['centroid_y'] / orig_height
            if not (0.0 <= cx_norm <= 1.0 and 0.0 <= cy_norm <= 1.0):
                n_clamped += 1
            y[i, 1] = np.clip(cx_norm, 0.0, 1.0)
            y[i, 2] = np.clip(cy_norm, 0.0, 1.0)
            # sun_present, when the generator recorded it. Datasets made
            # before that column existed fall back to "has a centroid",
            # which is the same thing.
            y[i, 3] = row['sun_present'] if has_presence else 1.0

    print(f"Off-frame suns clamped to frame edge: {n_clamped}/{n} "
          f"({100 * n_clamped / n:.1f}%)")
    return X, y, (orig_width, orig_height)


def main():
    parser = argparse.ArgumentParser(description="Train CNN for sun detection.")
    parser.add_argument('--data_dir', type=str, default='./synthetic_images',
                        help='Directory with synthetic images and metadata.csv')
    parser.add_argument('--input_width', type=int, default=320,
                        help='Width to which images are resized before the CNN')
    parser.add_argument('--input_height', type=int, default=240,
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

    best_weights = args.output_model.replace('.keras', '_best.weights.h5')
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        best_weights, monitor='val_loss', save_best_only=True,
        save_weights_only=True, verbose=1)

    model.fit(X_train, y_train,
              validation_data=(X_test, y_test),
              epochs=args.epochs,
              batch_size=args.batch_size,
              callbacks=[checkpoint])

    # Reload the best weights for export (avoids Lambda deserialization issues).
    model.load_weights(best_weights)
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
