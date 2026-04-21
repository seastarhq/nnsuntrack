#!/usr/bin/env python
import numpy as np
import cv2
import random
import os
import argparse
import pandas as pd


def draw_sun(image, cx, cy, major_axis, min_ellipticity, max_ellipticity, brightness):
    """Draw a circular or elliptical sun of the given major-axis radius.

    The caller picks the size so that it can also be used to set the centroid
    margin (how far outside the frame the centroid may fall).
    """
    is_circle = random.random() < 0.5

    if is_circle:
        radius = int(round(major_axis))
        cv2.circle(image, (int(round(cx)), int(round(cy))), radius, int(brightness), -1)
    else:
        ellipticity = random.uniform(min_ellipticity, max_ellipticity)
        minor_axis = major_axis / ellipticity
        axes = (int(round(major_axis)), int(round(minor_axis)))
        angle = random.uniform(0, 360)
        cv2.ellipse(image, (int(round(cx)), int(round(cy))), axes, angle, 0, 360,
                    int(brightness), -1)


def draw_reflection(image, image_width, image_height, min_size, max_size, brightness):
    """Draw a non-sun polygonal bright object representing a specular reflection."""
    shape_types = ['triangle', 'square', 'rectangle', 'polygon5', 'polygon6', 'polygon7']
    shape_type = random.choice(shape_types)

    x = random.uniform(0, image_width)
    y = random.uniform(0, image_height)

    if shape_type == 'square':
        side = random.uniform(min_size, max_size)
        top_left = (int(x - side / 2), int(y - side / 2))
        bottom_right = (int(x + side / 2), int(y + side / 2))
        cv2.rectangle(image, top_left, bottom_right, int(brightness), -1)
    elif shape_type == 'rectangle':
        width = random.uniform(min_size, max_size)
        height = random.uniform(min_size, max_size)
        while abs(width - height) < min_size * 0.1:
            height = random.uniform(min_size, max_size)
        top_left = (int(x - width / 2), int(y - height / 2))
        bottom_right = (int(x + width / 2), int(y + height / 2))
        cv2.rectangle(image, top_left, bottom_right, int(brightness), -1)
    else:
        n_sides = 3 if shape_type == 'triangle' else int(shape_type[-1])
        radius = random.uniform(min_size, max_size)
        offset_angle = random.uniform(0, 2 * np.pi)
        points = []
        for i in range(n_sides):
            theta = 2 * np.pi * i / n_sides + offset_angle
            px = x + radius * np.cos(theta)
            py = y + radius * np.sin(theta)
            points.append((int(px), int(py)))
        points = np.array(points, np.int32)
        cv2.fillPoly(image, [points], color=int(brightness))


def apply_partial_occlusion(image, cx, cy, sun_radius, visible_fraction,
                            background_brightness, image_width, image_height):
    """Occlude part of the sun with a half-plane that simulates a cloud edge.

    visible_fraction in [0, 1] controls how much of the sun disk remains visible:
    1.0 = no occlusion, 0.5 = half occluded, 0.0 = fully occluded.
    """
    angle = random.uniform(0, 2 * np.pi)
    # (dx, dy) is the inward normal of the cloud edge, pointing into the
    # occluded half-plane. d is the signed distance from the sun center to the
    # edge along that normal: d < 0 puts the center on the visible side
    # (most of the sun visible), d > 0 puts it on the occluded side.
    # visible_fraction = 1.0 -> edge at the far side of the disk (d = -r)
    # visible_fraction = 0.5 -> edge through center                 (d =  0)
    # visible_fraction = 0.0 -> edge at the near side of the disk   (d = +r)
    d = sun_radius * (1 - 2 * visible_fraction)

    dx, dy = np.cos(angle), np.sin(angle)
    ex, ey = cx + d * dx, cy + d * dy

    # Perpendicular direction along the edge.
    px, py = -dy, dx
    # Large enough to cover the whole image past the edge from any orientation.
    far = max(image_width, image_height) * 2

    p1 = (int(ex + far * px), int(ey + far * py))
    p2 = (int(ex - far * px), int(ey - far * py))
    p3 = (int(ex - far * px - far * dx), int(ey - far * py - far * dy))
    p4 = (int(ex + far * px - far * dx), int(ey + far * py - far * dy))

    occluder = np.array([p1, p2, p3, p4], np.int32)
    cv2.fillPoly(image, [occluder], int(background_brightness))


def generate_scene(image_width, image_height,
                   sun_min_size, sun_max_size,
                   refl_min_size, refl_max_size,
                   min_ellipticity, max_ellipticity,
                   sun_brightness, background_brightness, noise_stddev):
    """Generate one synthetic scene and its label metadata.

    Scenes are sampled from six categories covering the operational cases the
    tracker will encounter: clear sun, sun among reflections, occluded sun
    (thin cloud or partial blockage), reflections only, and empty sky.

    Returns (image, metadata) where metadata has:
      confidence:  1.0 for a clear sun, visible_fraction or cloud_factor for
                   occluded sun, 0.0 when no sun is present
      centroid_x, centroid_y: pixel coords of the sun's true center, or NaN
                              when there is no sun
      scene_type:  which category was sampled (for debugging/analysis)
    """
    scene_type = random.choices(
        ['sun_only', 'reflection_only', 'sun_and_reflections',
         'occluded_sun', 'occluded_sun_and_reflections', 'empty'],
        weights=[0.30, 0.15, 0.20, 0.15, 0.10, 0.10],
    )[0]

    # numpy shape is (rows, cols) = (height, width)
    image = np.full((image_height, image_width), background_brightness, dtype=np.uint8)
    confidence = 0.0
    centroid_x = float('nan')
    centroid_y = float('nan')

    has_sun = scene_type in ('sun_only', 'sun_and_reflections',
                             'occluded_sun', 'occluded_sun_and_reflections')
    has_reflections = scene_type in ('reflection_only', 'sun_and_reflections',
                                     'occluded_sun_and_reflections')
    is_occluded = scene_type in ('occluded_sun', 'occluded_sun_and_reflections')

    if has_sun:
        # Pick the sun's major-axis radius first so we can use it to set the
        # centroid margin - the centroid is allowed to fall outside the frame
        # by up to one major axis, which covers "sun partially clipped by the
        # frame edge" through "sun just barely off-frame" cases.
        sun_radius = random.uniform(sun_min_size, sun_max_size)
        centroid_x = random.uniform(-sun_radius, image_width + sun_radius)
        centroid_y = random.uniform(-sun_radius, image_height + sun_radius)

        if is_occluded:
            occlusion_type = random.choice(['thin_cloud', 'partial'])

            if occlusion_type == 'thin_cloud':
                # Thin cloud: draw sun dimmer, then blur to simulate diffuse
                # scattering. The cloud_factor becomes the confidence label.
                cloud_factor = random.uniform(0.2, 0.7)
                effective_brightness = sun_brightness * cloud_factor
                draw_sun(image, centroid_x, centroid_y,
                         sun_radius, min_ellipticity, max_ellipticity,
                         effective_brightness)
                blur_ksize = max(3, int(sun_radius * random.uniform(0.3, 0.8)) * 2 + 1)
                image = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), 0)
                confidence = cloud_factor
            else:
                # Partial occlusion: draw the full sun then cut it with a
                # half-plane. visible_fraction becomes the confidence label.
                draw_sun(image, centroid_x, centroid_y,
                         sun_radius, min_ellipticity, max_ellipticity,
                         sun_brightness)
                visible_fraction = random.uniform(0.3, 0.8)
                apply_partial_occlusion(image, centroid_x, centroid_y, sun_radius,
                                        visible_fraction, background_brightness,
                                        image_width, image_height)
                confidence = visible_fraction
        else:
            draw_sun(image, centroid_x, centroid_y,
                     sun_radius, min_ellipticity, max_ellipticity,
                     sun_brightness)
            confidence = 1.0

    if has_reflections:
        num_reflections = random.randint(1, 3)
        for _ in range(num_reflections):
            # Reflections vary in brightness - they can be as bright as the sun
            # or noticeably dimmer depending on the reflecting surface.
            refl_brightness = sun_brightness * random.uniform(0.5, 1.0)
            draw_reflection(image, image_width, image_height,
                            refl_min_size, refl_max_size, refl_brightness)

    # Gaussian noise. Use int16 intermediate to avoid uint8 wraparound on
    # negative samples.
    noise = np.random.normal(0, noise_stddev, image.shape).astype(np.int16)
    noisy_image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    metadata = {
        'centroid_x': centroid_x,
        'centroid_y': centroid_y,
        'confidence': confidence,
        'scene_type': scene_type,
    }

    return noisy_image, metadata


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic images for sun detection.")
    parser.add_argument('--num_images', type=int, default=1000,
                        help='Number of images to generate')
    parser.add_argument('--image_width', type=int, default=800,
                        help='Image width in pixels (default 800, matches Arducam)')
    parser.add_argument('--image_height', type=int, default=600,
                        help='Image height in pixels (default 600, matches Arducam)')
    parser.add_argument('--sun_min_size', type=float, default=20,
                        help='Minimum sun major-axis radius in pixels')
    parser.add_argument('--sun_max_size', type=float, default=100,
                        help='Maximum sun major-axis radius in pixels')
    parser.add_argument('--refl_min_size', type=float, default=20,
                        help='Minimum reflection size in pixels')
    parser.add_argument('--refl_max_size', type=float, default=100,
                        help='Maximum reflection size in pixels')
    parser.add_argument('--min_ellipticity', type=float, default=0.8,
                        help='Minimum ellipticity ratio (major/minor)')
    parser.add_argument('--max_ellipticity', type=float, default=1.2,
                        help='Maximum ellipticity ratio (major/minor)')
    parser.add_argument('--sun_brightness', type=int, default=200,
                        help='Brightness of the sun [0-255]')
    parser.add_argument('--background_brightness', type=int, default=30,
                        help='Brightness of the background [0-255]')
    parser.add_argument('--noise_stddev', type=float, default=20,
                        help='Standard deviation of Gaussian noise')
    parser.add_argument('--output_dir', type=str, default='./synthetic_images',
                        help='Directory to save generated images and metadata.csv')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    metadata_list = []
    for i in range(args.num_images):
        image, metadata = generate_scene(
            image_width=args.image_width,
            image_height=args.image_height,
            sun_min_size=args.sun_min_size,
            sun_max_size=args.sun_max_size,
            refl_min_size=args.refl_min_size,
            refl_max_size=args.refl_max_size,
            min_ellipticity=args.min_ellipticity,
            max_ellipticity=args.max_ellipticity,
            sun_brightness=args.sun_brightness,
            background_brightness=args.background_brightness,
            noise_stddev=args.noise_stddev,
        )
        image_filename = f'image_{i}.png'
        cv2.imwrite(os.path.join(args.output_dir, image_filename), image)
        metadata['image_filename'] = image_filename
        metadata_list.append(metadata)

    df = pd.DataFrame(metadata_list)
    df.to_csv(os.path.join(args.output_dir, 'metadata.csv'), index=False)


if __name__ == "__main__":
    main()
