#!/usr/bin/env python
"""Generate synthetic training images for the sun tracker.

Scenes are built in linear radiance (float32, arbitrary units) and then
pushed through a model of the imaging chain, in the order light actually
encounters it:

    scene radiance -> optics (bloom, flare, vignetting)
                   -> exposure/gain
                   -> sensor (PRNU, hot pixels/rows, shot + read noise)
                   -> 8-bit quantization

Working in radiance before clipping is what makes the bright cases come
out right: the sun is ~1e4 times brighter than the sky, so it saturates
and its point-spread wings bloom into a disk far larger than the sun's
true angular size. Drawing a flat-topped disk straight into 8-bit space
cannot reproduce that.

Geometry is driven by a camera model rather than free pixel sizes. The
sun subtends 0.53 degrees, which is ~14 px across on the narrow camera
but only ~3 px on the 140-degree fisheye - so most of what the detector
sees of the sun is bloom, not disk. Images are generated in the camera's
native projection (raw fisheye, not undistorted), and centroid labels are
in that same distorted pixel space.

Cloud edges near the sun are deliberately brighter than the sun itself:
forward-scattered sunlight through a thin edge is the brightest thing in
a real sky, so "track the brightest blob" is exactly the wrong policy.
The sun stays labeled through these, including when fully occluded.

Labels written to metadata.csv:
  centroid_x, centroid_y  sun center in pixels, NaN when no sun is present
  confidence              1.0 clear sun, mean transmission when occluded,
                          0.0 when no sun (unchanged semantics)
  sun_present             1.0 whenever a sun is in the scene, even fully
                          hidden - lets the model separate "no sun here"
                          from "sun present but occluded"
  occlusion               0.0 clear .. 1.0 fully hidden, NaN when no sun
  scene_type              which category was sampled, for analysis
"""
import numpy as np
import cv2
import os
import json
import argparse
import multiprocessing as mp
import pandas as pd

# Radiance units: the saturation level of the sensor is 255.0, so a
# radiance of 255 exposes to full scale at unit gain. Everything below is
# expressed against that reference.
SATURATION = 255.0

SUN_ANGULAR_DIAMETER_DEG = 0.53

# Camera presets for the two Arducams. Every field is overridable from the
# command line so a hardware change is a flag, not an edit.
CAMERA_PRESETS = {
    'narrow': {
        'fov_deg': 30.0,
        'projection': 'rectilinear',
        'image_width': 800,
        'image_height': 600,
    },
    'fisheye': {
        'fov_deg': 140.0,
        'projection': 'equidistant',
        'image_width': 800,
        'image_height': 600,
    },
}


class Camera:
    """Projection model mapping sky angles to pixels.

    `fov_deg` is the horizontal field of view, so the focal length is set
    to make that FOV span the image width.

    rectilinear  r = f*tan(theta)   - normal lens
    equidistant  r = f*theta        - the usual fisheye construction
    """

    def __init__(self, width, height, fov_deg, projection):
        self.width = width
        self.height = height
        self.fov = np.radians(fov_deg)
        self.projection = projection
        self.cx = width / 2.0
        self.cy = height / 2.0

        half = self.fov / 2.0
        if projection == 'rectilinear':
            self.f = (width / 2.0) / np.tan(half)
        elif projection == 'equidistant':
            self.f = (width / 2.0) / half
        else:
            raise ValueError(f"unknown projection: {projection}")

    def theta_at_radius(self, r):
        """Inverse projection: pixel radius from center -> angle off-axis."""
        if self.projection == 'rectilinear':
            return np.arctan(r / self.f)
        return r / self.f

    def scales_at(self, theta):
        """Local pixel-per-radian scale at angle `theta` off-axis.

        Returns (radial, tangential). These differ off-axis, which is why
        the sun's image is not circular away from the optical center: a
        rectilinear lens stretches it radially, an equidistant fisheye
        stretches it tangentially.
        """
        if self.projection == 'rectilinear':
            c = np.cos(theta)
            return self.f / (c * c), self.f / c
        # Equidistant. tangential = f*theta/sin(theta) -> f as theta -> 0.
        if theta < 1e-6:
            return self.f, self.f
        return self.f, self.f * theta / np.sin(theta)


def build_sensor(rng, width, height, hot_pixel_rate, hot_row_count,
                 prnu_stddev, dark_offset):
    """Fixed-pattern sensor defects, generated once and shared by every
    image in a dataset.

    This is the point of fixed-pattern noise: it sits at the same pixels
    in every frame, so a detector can learn to ignore it. Re-rolling it
    per image would turn it into ordinary noise and teach nothing.
    """
    n_hot = int(hot_pixel_rate * width * height)
    hot_y = rng.integers(0, height, n_hot)
    hot_x = rng.integers(0, width, n_hot)
    hot_level = rng.uniform(0.3, 1.0, n_hot).astype(np.float32) * SATURATION

    hot_pixels = np.zeros((height, width), np.float32)
    hot_pixels[hot_y, hot_x] = hot_level

    # Hot rows: a whole readout row sitting high, as seen with a bad
    # row-select or a damaged column amplifier.
    hot_rows = np.zeros((height, width), np.float32)
    for _ in range(hot_row_count):
        row = rng.integers(0, height)
        hot_rows[row, :] = rng.uniform(0.05, 0.25) * SATURATION

    # Photo-response non-uniformity: per-pixel gain variation, multiplicative.
    prnu = (1.0 + rng.normal(0.0, prnu_stddev, (height, width))).astype(np.float32)

    return {
        'hot_pixels': hot_pixels,
        'hot_rows': hot_rows,
        'prnu': prnu,
        'dark_offset': dark_offset,
    }


def fbm(rng, height, width, octaves=5, persistence=0.55):
    """Fractional Brownian motion noise, for cloud structure.

    Built by upsampling successively finer random grids - cheap, and the
    bicubic interpolation gives the soft irregular edges that a
    half-plane occluder cannot.
    """
    field = np.zeros((height, width), np.float32)
    amplitude = 1.0
    total = 0.0
    size = 3
    for _ in range(octaves):
        grid = rng.random((size, size)).astype(np.float32)
        layer = cv2.resize(grid, (width, height), interpolation=cv2.INTER_CUBIC)
        field += amplitude * layer
        total += amplitude
        amplitude *= persistence
        size *= 2
    return field / total


def smoothstep(x, lo, hi):
    t = np.clip((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def draw_sun_disk(layer, cx, cy, a_radial, a_tangential, angle, radiance,
                  supersample=8):
    """Draw the solar disk into `layer` with sub-pixel accuracy.

    The disk can be ~3 px across on the fisheye while the accuracy target
    is 0.57 px, so a hard-edged rasterization would quantize the label
    itself. Rendering a small ROI at `supersample` resolution and box-
    downsampling keeps the rendered centroid faithful to the float
    coordinates.
    """
    height, width = layer.shape
    reach = int(np.ceil(max(a_radial, a_tangential))) + 2
    x0, x1 = int(np.floor(cx)) - reach, int(np.ceil(cx)) + reach
    y0, y1 = int(np.floor(cy)) - reach, int(np.ceil(cy)) + reach
    x0c, y0c = max(x0, 0), max(y0, 0)
    x1c, y1c = min(x1, width), min(y1, height)
    if x0c >= x1c or y0c >= y1c:
        return  # entirely off-frame

    roi_w, roi_h = x1c - x0c, y1c - y0c
    big = np.zeros((roi_h * supersample, roi_w * supersample), np.float32)
    cv2.ellipse(
        big,
        center=(int(round((cx - x0c) * supersample)),
                int(round((cy - y0c) * supersample))),
        axes=(max(int(round(a_radial * supersample)), 1),
              max(int(round(a_tangential * supersample)), 1)),
        angle=np.degrees(angle), startAngle=0, endAngle=360,
        color=float(radiance), thickness=-1, lineType=cv2.LINE_8)
    small = cv2.resize(big, (roi_w, roi_h), interpolation=cv2.INTER_AREA)
    layer[y0c:y1c, x0c:x1c] += small


def apply_bloom(sun_layer, core_sigma, glare_weights, glare_sigmas):
    """Spread the sun layer by the optical point-spread function.

    A real PSF is a sharp core on top of very broad wings (veiling glare
    from scattering in the lens stack). The wings are what turn a 3 px
    solar disk into a large saturated blob. Approximated as a sum of
    Gaussians; the broad ones are computed on a downsampled copy, which
    is where nearly all the cost would otherwise be.
    """
    out = cv2.GaussianBlur(sun_layer, (0, 0), core_sigma)
    height, width = sun_layer.shape

    for weight, sigma in zip(glare_weights, glare_sigmas):
        if sigma <= 8.0:
            out += weight * cv2.GaussianBlur(sun_layer, (0, 0), sigma)
            continue
        # Downsample, blur cheaply at reduced sigma, upsample. Energy is
        # preserved because the mean is unchanged by resize + blur.
        factor = 4 if sigma <= 32.0 else 8
        small = cv2.resize(sun_layer, (width // factor, height // factor),
                           interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), sigma / factor)
        out += weight * cv2.resize(small, (width, height),
                                   interpolation=cv2.INTER_LINEAR)
    return out


def add_flare(layer, rng, sun_x, sun_y, camera, sun_radiance,
              ghost_gain, streak_gain, n_blades, starburst):
    """Lens flare: ghost images, and optionally an aperture starburst.

    Ghosts are inter-element reflections, so they land on the line from
    the sun through the optical center, mirrored and scaled.

    The starburst is diffraction at the edges of a blade diaphragm (an odd
    blade count gives 2N rays, an even count gives N). The Arducam modules
    have a fixed aperture and no blades, so they do not produce one - it
    is off by default and only worth enabling for a lens that has an iris.
    """
    dx, dy = sun_x - camera.cx, sun_y - camera.cy

    n_ghosts = rng.integers(3, 7)
    for _ in range(n_ghosts):
        t = rng.uniform(-1.0, 0.9)
        gx, gy = camera.cx + t * dx, camera.cy + t * dy
        radius = rng.uniform(6.0, 40.0)
        brightness = sun_radiance * ghost_gain * rng.uniform(0.2, 1.0)
        ghost = np.zeros_like(layer)
        cv2.circle(ghost, (int(round(gx)), int(round(gy))),
                   int(radius), float(brightness), -1, lineType=cv2.LINE_AA)
        # Ghosts are defocused images of the aperture, so they are soft.
        layer += cv2.GaussianBlur(ghost, (0, 0), radius * 0.25)

    if not starburst:
        return

    n_rays = n_blades if n_blades % 2 == 0 else 2 * n_blades
    streak = np.zeros_like(layer)
    length = max(camera.width, camera.height) * rng.uniform(0.2, 0.5)
    phase = rng.uniform(0, np.pi)
    for i in range(n_rays):
        angle = phase + 2.0 * np.pi * i / n_rays
        ex = sun_x + length * np.cos(angle)
        ey = sun_y + length * np.sin(angle)
        cv2.line(streak, (int(round(sun_x)), int(round(sun_y))),
                 (int(round(ex)), int(round(ey))),
                 float(sun_radiance * streak_gain), 1, lineType=cv2.LINE_AA)
    # Taper the rays so they fade with distance from the sun, rather than
    # running at full brightness clear across the frame.
    yy, xx = np.mgrid[0:layer.shape[0], 0:layer.shape[1]].astype(np.float32)
    falloff = np.exp(-np.hypot(xx - sun_x, yy - sun_y) / (length * 0.3))
    streak = cv2.GaussianBlur(streak, (0, 0), 1.5) * falloff
    layer += streak


def draw_reflection(layer, rng, width, height, min_size, max_size, radiance):
    """A specular glint off the robot's own structure.

    These are the hard negatives: bright, saturating, and shaped like
    everything except a disk. Kept polygonal for exactly that reason, but
    placed toward the frame edges where the robot frame actually intrudes.
    """
    shape_type = rng.choice(['triangle', 'square', 'rectangle',
                             'polygon5', 'polygon6', 'polygon7'])

    edge_bias = rng.random() < 0.6
    if edge_bias:
        if rng.random() < 0.5:
            x = rng.choice([rng.uniform(0, width * 0.2),
                            rng.uniform(width * 0.8, width)])
            y = rng.uniform(0, height)
        else:
            x = rng.uniform(0, width)
            y = rng.choice([rng.uniform(0, height * 0.2),
                            rng.uniform(height * 0.8, height)])
    else:
        x, y = rng.uniform(0, width), rng.uniform(0, height)

    if shape_type == 'square':
        side = rng.uniform(min_size, max_size)
        cv2.rectangle(layer, (int(x - side / 2), int(y - side / 2)),
                      (int(x + side / 2), int(y + side / 2)),
                      float(radiance), -1)
    elif shape_type == 'rectangle':
        w = rng.uniform(min_size, max_size)
        h = rng.uniform(min_size, max_size)
        while abs(w - h) < min_size * 0.1:
            h = rng.uniform(min_size, max_size)
        cv2.rectangle(layer, (int(x - w / 2), int(y - h / 2)),
                      (int(x + w / 2), int(y + h / 2)), float(radiance), -1)
    else:
        n_sides = 3 if shape_type == 'triangle' else int(shape_type[-1])
        radius = rng.uniform(min_size, max_size)
        offset = rng.uniform(0, 2 * np.pi)
        pts = np.array([
            (int(x + radius * np.cos(2 * np.pi * i / n_sides + offset)),
             int(y + radius * np.sin(2 * np.pi * i / n_sides + offset)))
            for i in range(n_sides)], np.int32)
        cv2.fillPoly(layer, [pts], color=float(radiance))


def vignette_map(camera):
    """Natural (cos^4) illumination falloff for a rectilinear lens.

    Fisheye lenses are designed against this, so they get a much gentler
    falloff rather than the cos^4 law.
    """
    yy, xx = np.mgrid[0:camera.height, 0:camera.width].astype(np.float32)
    r = np.hypot(xx - camera.cx, yy - camera.cy)
    theta = camera.theta_at_radius(r)
    if camera.projection == 'rectilinear':
        return np.cos(theta) ** 4
    return 1.0 - 0.25 * (theta / (camera.fov / 2.0)) ** 2


def generate_scene(rng, camera, sensor, vignette, args):
    """Build one scene and its labels."""
    scene_type = rng.choice(
        ['sun_only', 'reflection_only', 'sun_and_reflections',
         'occluded_sun', 'occluded_sun_and_reflections', 'bright_cloud_edge',
         'empty'],
        p=[0.24, 0.12, 0.16, 0.13, 0.09, 0.16, 0.10])

    has_sun = scene_type in ('sun_only', 'sun_and_reflections', 'occluded_sun',
                             'occluded_sun_and_reflections', 'bright_cloud_edge')
    has_reflections = scene_type in ('reflection_only', 'sun_and_reflections',
                                     'occluded_sun_and_reflections')
    has_cloud = scene_type in ('occluded_sun', 'occluded_sun_and_reflections',
                               'bright_cloud_edge')

    height, width = camera.height, camera.width

    # --- Sky background: a gradient, not a flat fill ---
    sky_level = rng.uniform(*args.sky_radiance_range)
    yy = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    gradient = 1.0 + args.sky_gradient * (0.5 - yy)
    scene = np.full((height, width), sky_level, np.float32) * gradient

    centroid_x = float('nan')
    centroid_y = float('nan')
    confidence = 0.0
    occlusion = float('nan')
    sun_present = 0.0

    sun_layer = np.zeros((height, width), np.float32)
    sun_x = sun_y = None
    sun_radiance = 0.0
    a_radial = a_tangential = 0.0

    if has_sun:
        sun_present = 1.0
        sun_radiance = rng.uniform(*args.sun_radiance_range)

        # Place the sun by pixel position so the frame is well covered,
        # then invert the projection to get the true off-axis angle. The
        # margin lets the center fall outside the frame - a partially
        # clipped sun is a case the tracker has to survive.
        margin = args.sun_margin_px
        sun_x = rng.uniform(-margin, width + margin)
        sun_y = rng.uniform(-margin, height + margin)
        centroid_x, centroid_y = sun_x, sun_y

        r = np.hypot(sun_x - camera.cx, sun_y - camera.cy)
        theta = camera.theta_at_radius(r)
        phi = np.arctan2(sun_y - camera.cy, sun_x - camera.cx)
        radial_scale, tangential_scale = camera.scales_at(theta)

        alpha = np.radians(args.sun_angular_diameter * args.sun_size_scale) / 2.0
        if args.sun_radius_px is not None:
            a_radial = a_tangential = args.sun_radius_px
        else:
            a_radial = radial_scale * alpha
            a_tangential = tangential_scale * alpha

        draw_sun_disk(sun_layer, sun_x, sun_y, a_radial, a_tangential, phi,
                      sun_radiance)

    # --- Cloud ---
    if has_cloud:
        density = fbm(rng, height, width, octaves=args.cloud_octaves)

        if has_sun:
            # Put the cloud edge right at the sun. Choosing the threshold
            # from the density sampled at the sun's position guarantees a
            # transition passes through it, rather than hoping one lands
            # there by chance.
            sx = int(np.clip(sun_x, 0, width - 1))
            sy = int(np.clip(sun_y, 0, height - 1))
            at_sun = float(density[sy, sx])
            if scene_type == 'bright_cloud_edge':
                # Edge adjacent to the sun but not over it.
                lo = at_sun + rng.uniform(0.01, 0.06)
            else:
                lo = at_sun - rng.uniform(0.0, 0.12)
        else:
            lo = float(np.percentile(density, rng.uniform(35, 65)))

        cloud = smoothstep(density, lo, lo + args.cloud_edge_softness)

        # Optical depth -> transmission through the cloud.
        tau = args.cloud_optical_depth * rng.uniform(0.6, 1.4)
        transmission = np.exp(-tau * cloud)
        sun_layer *= transmission

        # Forward scattering. Sunlight scattered through a thin cloud edge
        # near the solar direction is the brightest thing in the sky -
        # brighter than the disk itself. It peaks at moderate density
        # (thick cloud blocks it, clear sky has nothing to scatter) and
        # falls off with angular distance from the sun.
        if has_sun:
            yy2, xx2 = np.mgrid[0:height, 0:width].astype(np.float32)
            dist = np.hypot(xx2 - sun_x, yy2 - sun_y)
            falloff = np.exp(-dist / args.forward_scatter_scale)
            edge_weight = cloud * np.exp(-args.cloud_optical_depth * cloud)
            scene += (sun_radiance * args.forward_scatter_gain
                      * edge_weight * falloff)

        # Ambient cloud brightness away from the sun. Viewed from below the
        # cloud is backlit, so thin parts glow and thick cores go dark -
        # that variation is what gives a cloud internal structure instead
        # of a flat gray fill.
        thickness = smoothstep(density, lo, lo + args.cloud_thickness_scale)
        shade = args.cloud_shade_range[1] - (
            args.cloud_shade_range[1] - args.cloud_shade_range[0]) * thickness
        scene += cloud * sky_level * rng.uniform(1.5, 4.0) * shade

        if has_sun:
            # Occlusion label: mean transmission over the solar disk, so a
            # thin edge and a thick core give meaningfully different
            # numbers instead of a binary flag.
            reach = max(int(np.ceil(max(a_radial, a_tangential))), 1)
            x0 = int(np.clip(sun_x - reach, 0, width - 1))
            x1 = int(np.clip(sun_x + reach + 1, 1, width))
            y0 = int(np.clip(sun_y - reach, 0, height - 1))
            y1 = int(np.clip(sun_y + reach + 1, 1, height))
            if x1 > x0 and y1 > y0:
                mean_t = float(np.mean(transmission[y0:y1, x0:x1]))
            else:
                mean_t = 1.0  # sun off-frame; nothing to occlude
            occlusion = 1.0 - mean_t
            confidence = mean_t
        else:
            confidence = 0.0
    elif has_sun:
        occlusion = 0.0
        confidence = 1.0

    # --- Specular glints off the robot frame ---
    refl_layer = np.zeros((height, width), np.float32)
    if has_reflections:
        for _ in range(rng.integers(1, 4)):
            radiance = SATURATION * rng.uniform(*args.reflection_radiance_range)
            draw_reflection(refl_layer, rng, width, height,
                            args.refl_min_size, args.refl_max_size, radiance)

    # --- Optics ---
    # One PSF for everything. Bloom is a property of the lens, not of the
    # object, so the same convolution applies to the sun and to glints -
    # the sun blooms more only because it is far brighter. Glints are not
    # attenuated by cloud (the robot frame is in front of the sky), which
    # is why they are combined only after transmission is applied.
    if has_sun or has_reflections:
        scene += apply_bloom(sun_layer + refl_layer, args.psf_core_sigma,
                             args.psf_glare_weights, args.psf_glare_sigmas)

    if has_sun and args.flare and 0 <= sun_x < width and 0 <= sun_y < height:
        add_flare(scene, rng, sun_x, sun_y, camera, sun_radiance,
                  args.flare_ghost_gain, args.flare_streak_gain,
                  args.aperture_blades, args.starburst)

    scene *= vignette

    # --- Exposure ---
    def auto_gain():
        """Center-weighted average metering, as a real camera does it.

        Metering off a high percentile would let a single saturating
        glint drive the gain to its floor and crush the whole sky to
        black. Averaging over the scene means small bright objects blow
        out - which is correct - while a large bright cloud still pulls
        the exposure down and can leave the sun no longer the brightest
        thing in the frame.
        """
        level = np.percentile(scene, args.auto_exposure_percentile)
        return float(np.clip(args.auto_exposure_target / max(float(level), 1e-3),
                             *args.gain_range))

    if args.exposure == 'auto':
        gain, mode = auto_gain(), 'auto'
    elif args.exposure == 'fixed':
        gain, mode = rng.uniform(*args.fixed_gain_range), 'fixed'
    elif rng.random() < 0.5:  # 'both' - mix the two regimes
        gain, mode = auto_gain(), 'auto'
    else:
        gain, mode = rng.uniform(*args.fixed_gain_range), 'fixed'

    signal = scene * gain

    # --- Sensor ---
    signal *= sensor['prnu']
    signal += sensor['dark_offset']
    signal += sensor['hot_pixels']
    signal += sensor['hot_rows']

    # Shot noise is Poisson in photoelectrons, so its standard deviation
    # grows as sqrt(signal) - unlike the uniform Gaussian this previously
    # used, it is strongest exactly where the scene is brightest.
    if args.full_well > 0:
        electrons = np.clip(signal, 0, None) * (args.full_well / SATURATION)
        noisy = rng.poisson(np.clip(electrons, 0, 1e9)).astype(np.float32)
        signal = noisy * (SATURATION / args.full_well)

    signal += rng.normal(0.0, args.read_noise, signal.shape).astype(np.float32)

    image = np.clip(signal, 0, SATURATION).astype(np.uint8)

    metadata = {
        'centroid_x': centroid_x,
        'centroid_y': centroid_y,
        'confidence': confidence,
        'sun_present': sun_present,
        'occlusion': occlusion,
        'scene_type': scene_type,
        'exposure_mode': mode,
        'exposure_gain': gain,
    }
    return image, metadata


# Set once per worker process by _init_worker. The camera model, sensor
# pattern and vignette map are identical for every image in a dataset, so
# they are shipped to each worker once instead of pickled per task.
_WORKER = {}


def _init_worker(camera, sensor, vignette, args, output_dir, seed):
    # OpenCV multithreads its own primitives, which would oversubscribe the
    # machine against the process pool and make it slower than serial.
    cv2.setNumThreads(1)
    _WORKER.update(camera=camera, sensor=sensor, vignette=vignette,
                   args=args, output_dir=output_dir, seed=seed)


def _generate_one(index):
    """Render, write and label image `index`.

    The RNG is seeded from (base seed, image index) rather than drawn from
    one shared stream, so output is reproducible and identical regardless
    of --jobs: the worker count changes scheduling, not the random stream
    any given image sees.
    """
    w = _WORKER
    rng = np.random.default_rng([w['seed'], index])
    image, metadata = generate_scene(rng, w['camera'], w['sensor'],
                                     w['vignette'], w['args'])
    filename = f'image_{index}.png'
    cv2.imwrite(os.path.join(w['output_dir'], filename), image)
    metadata['image_filename'] = filename
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic images for sun detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--num_images', type=int, default=1000)
    parser.add_argument('--output_dir', type=str, default='./synthetic_images')
    parser.add_argument('--seed', type=int, default=0,
                        help='Base RNG seed; also seeds the fixed-pattern sensor')
    parser.add_argument('--jobs', type=int, default=0,
                        help='Worker processes; 0 uses every core, 1 runs '
                             'serially (easier to profile and debug)')

    # --- Camera geometry ---
    parser.add_argument('--camera', choices=sorted(CAMERA_PRESETS),
                        default='narrow',
                        help='Preset for FOV, projection and resolution')
    parser.add_argument('--fov_deg', type=float, default=None,
                        help='Horizontal FOV; overrides the preset')
    parser.add_argument('--projection', choices=['rectilinear', 'equidistant'],
                        default=None, help='Overrides the preset')
    parser.add_argument('--image_width', type=int, default=None)
    parser.add_argument('--image_height', type=int, default=None)

    # --- Sun ---
    parser.add_argument('--sun_angular_diameter', type=float,
                        default=SUN_ANGULAR_DIAMETER_DEG,
                        help='Angular diameter in degrees (0.53 is physical)')
    parser.add_argument('--sun_size_scale', type=float, default=1.0,
                        help='Multiplier on the angular diameter, for '
                             'experimenting with apparent size')
    parser.add_argument('--sun_radius_px', type=float, default=None,
                        help='Force a fixed disk radius in pixels, ignoring '
                             'the camera geometry')
    parser.add_argument('--sun_radiance_range', type=float, nargs=2,
                        default=[3000.0, 12000.0],
                        help='Solar radiance, in units where 255 saturates')
    parser.add_argument('--sun_margin_px', type=float, default=60.0,
                        help='How far outside the frame the sun center may fall')

    # --- Sky and cloud ---
    parser.add_argument('--sky_radiance_range', type=float, nargs=2,
                        default=[12.0, 45.0])
    parser.add_argument('--sky_gradient', type=float, default=0.35,
                        help='Top-to-bottom relative brightness variation')
    parser.add_argument('--cloud_octaves', type=int, default=6)
    parser.add_argument('--cloud_edge_softness', type=float, default=0.18,
                        help='Density width of the cloud edge transition')
    parser.add_argument('--cloud_thickness_scale', type=float, default=0.45,
                        help='Density range over which a cloud goes from '
                             'thin/glowing to thick/dark')
    parser.add_argument('--cloud_shade_range', type=float, nargs=2,
                        default=[0.6, 1.6],
                        help='Brightness multiplier at thick core and thin '
                             'edge respectively (backlit cloud)')
    parser.add_argument('--cloud_optical_depth', type=float, default=4.0)
    parser.add_argument('--forward_scatter_gain', type=float, default=0.02,
                        help='Brightness of forward-scattered light at a '
                             'cloud edge, as a fraction of solar radiance. '
                             'Large enough that the edge outshines the sun.')
    parser.add_argument('--forward_scatter_scale', type=float, default=120.0,
                        help='Falloff distance in pixels from the sun')

    # --- Reflections ---
    # Glints off a robot frame are small at 30 deg FOV - the previous
    # 20-100 px defaults produced shapes filling much of the frame.
    parser.add_argument('--refl_min_size', type=float, default=5.0)
    parser.add_argument('--refl_max_size', type=float, default=45.0)
    parser.add_argument('--reflection_radiance_range', type=float, nargs=2,
                        default=[1.2, 8.0],
                        help='Glint radiance as a multiple of saturation')

    # --- Optics ---
    parser.add_argument('--psf_core_sigma', type=float, default=0.9)
    parser.add_argument('--psf_glare_weights', type=float, nargs='+',
                        default=[0.10, 0.030, 0.010, 0.004])
    parser.add_argument('--psf_glare_sigmas', type=float, nargs='+',
                        default=[3.0, 10.0, 30.0, 90.0])
    parser.add_argument('--flare', action='store_true', default=True)
    parser.add_argument('--no_flare', dest='flare', action='store_false')
    parser.add_argument('--flare_ghost_gain', type=float, default=6e-4)
    # Off by default: the Arducam has a fixed aperture with no diaphragm
    # blades, so it produces no diffraction spikes. Only enable for a lens
    # that actually has an iris.
    parser.add_argument('--starburst', action='store_true', default=False,
                        help='Draw aperture diffraction spikes')
    parser.add_argument('--flare_streak_gain', type=float, default=4e-3)
    parser.add_argument('--aperture_blades', type=int, default=6)

    # --- Exposure ---
    parser.add_argument('--exposure', choices=['fixed', 'auto', 'both'],
                        default='both',
                        help="'both' mixes fixed and auto-exposure scenes")
    parser.add_argument('--gain_range', type=float, nargs=2,
                        default=[0.2, 6.0],
                        help='Clip limits on the auto-exposure gain')
    parser.add_argument('--fixed_gain_range', type=float, nargs=2,
                        default=[0.7, 1.8],
                        help='Gain sampled for fixed-exposure scenes. Narrow '
                             'on purpose: a real fixed exposure is set for '
                             'the sky, not drawn from a 10x spread.')
    parser.add_argument('--auto_exposure_target', type=float, default=110.0,
                        help='Level the metered statistic is driven toward')
    parser.add_argument('--auto_exposure_percentile', type=float, default=70.0,
                        help='Scene percentile used for metering. Broad by '
                             'design: a high percentile would let one bright '
                             'glint black out the whole frame.')

    # --- Sensor ---
    parser.add_argument('--hot_pixel_rate', type=float, default=2e-5,
                        help='Fraction of pixels that are hot')
    parser.add_argument('--hot_row_count', type=int, default=2)
    parser.add_argument('--prnu_stddev', type=float, default=0.01)
    parser.add_argument('--dark_offset', type=float, default=2.0)
    parser.add_argument('--full_well', type=float, default=8000.0,
                        help='Full-well capacity in electrons, setting the '
                             'shot-noise level. 0 disables shot noise.')
    parser.add_argument('--read_noise', type=float, default=1.5)

    args = parser.parse_args()

    preset = CAMERA_PRESETS[args.camera]
    fov_deg = args.fov_deg if args.fov_deg is not None else preset['fov_deg']
    projection = args.projection or preset['projection']
    width = args.image_width or preset['image_width']
    height = args.image_height or preset['image_height']

    camera = Camera(width, height, fov_deg, projection)
    rng = np.random.default_rng(args.seed)
    sensor = build_sensor(rng, width, height, args.hot_pixel_rate,
                          args.hot_row_count, args.prnu_stddev,
                          args.dark_offset)
    vignette = vignette_map(camera)

    sun_deg = args.sun_angular_diameter * args.sun_size_scale
    disk_px = camera.f * np.radians(sun_deg)  # on-axis, where scale == f
    print(f"Camera: {args.camera}  {width}x{height}  {fov_deg:.1f} deg "
          f"{projection}  (f={camera.f:.1f} px)")
    print(f"Solar disk on axis: {disk_px:.2f} px across ({sun_deg:.3f} deg)")

    os.makedirs(args.output_dir, exist_ok=True)

    jobs = args.jobs if args.jobs > 0 else (os.cpu_count() or 1)
    jobs = max(1, min(jobs, args.num_images))
    init_args = (camera, sensor, vignette, args, args.output_dir, args.seed)
    print(f"Generating {args.num_images} images on {jobs} process(es)")

    def report(done):
        step = max(args.num_images // 10, 1)
        if done % step == 0 or done == args.num_images:
            print(f"  {done}/{args.num_images}", flush=True)

    metadata_list = []
    if jobs == 1:
        _init_worker(*init_args)
        for i in range(args.num_images):
            metadata_list.append(_generate_one(i))
            report(i + 1)
    else:
        # imap preserves order, so metadata rows stay aligned with the
        # image_N.png numbering regardless of completion order.
        with mp.Pool(jobs, initializer=_init_worker,
                     initargs=init_args) as pool:
            for done, metadata in enumerate(
                    pool.imap(_generate_one, range(args.num_images),
                              chunksize=8), start=1):
                metadata_list.append(metadata)
                report(done)

    pd.DataFrame(metadata_list).to_csv(
        os.path.join(args.output_dir, 'metadata.csv'), index=False)

    # Record how this dataset was made. Training and deployment need the
    # camera geometry, and it is otherwise unrecoverable from the PNGs.
    params = dict(vars(args))
    params.update({'fov_deg': fov_deg, 'projection': projection,
                   'image_width': width, 'image_height': height,
                   'focal_length_px': camera.f})
    with open(os.path.join(args.output_dir, 'generation_params.json'), 'w') as f:
        json.dump(params, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
