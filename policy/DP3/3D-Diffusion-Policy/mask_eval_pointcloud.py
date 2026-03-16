import numpy as np


def _to_01(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32)
    if rgb.min() < -0.2:
        rgb = (rgb + 1.0) / 2.0
    return np.clip(rgb, 0.0, 1.0)


def classify_color_points(
    point_cloud: np.ndarray,
    sat_thresh: float = 0.18,
    red_margin: float = 0.08,
    green_margin: float = 0.06,
    blue_margin: float = 0.06,
):
    """
    point_cloud: [N, 6] -> [x,y,z,r,g,b]
    returns boolean masks: is_red, is_green, is_blue, is_neutral
    """
    assert point_cloud.ndim == 2 and point_cloud.shape[1] >= 6, \
        f"Expected [N,6+], got {point_cloud.shape}"

    rgb = _to_01(point_cloud[:, 3:6])
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    rgb_max = np.max(rgb, axis=1)
    rgb_min = np.min(rgb, axis=1)
    sat = rgb_max - rgb_min

    is_colored = sat > sat_thresh

    is_red = is_colored & (r > g + red_margin) & (r > b + red_margin)
    is_green = is_colored & (g > r + green_margin) & (g > b + green_margin)
    is_blue = is_colored & (b > r + blue_margin) & (b > g + blue_margin)

    is_neutral = ~(is_red | is_green | is_blue)
    return is_red, is_green, is_blue, is_neutral


def mask_green_blue_points(
    point_cloud: np.ndarray,
    keep_red: bool = True,
    keep_neutral: bool = True,
    resample_to_original_size: bool = True,
    rng: np.random.Generator = None,
):
    """
    Remove green/blue colored object points, keep red + neutral points.
    point_cloud: [N, 6]
    """
    if rng is None:
        rng = np.random.default_rng()

    N = point_cloud.shape[0]
    is_red, is_green, is_blue, is_neutral = classify_color_points(point_cloud)

    keep_mask = np.zeros(N, dtype=bool)
    if keep_red:
        keep_mask |= is_red
    if keep_neutral:
        keep_mask |= is_neutral

    kept = point_cloud[keep_mask]

    if len(kept) == 0:
        return point_cloud.copy()

    if not resample_to_original_size:
        return kept

    if len(kept) >= N:
        idx = rng.choice(len(kept), size=N, replace=False)
    else:
        idx = rng.choice(len(kept), size=N, replace=True)

    return kept[idx]


def mask_green_blue_sequence(point_cloud_seq: np.ndarray, **kwargs):
    """
    Supports:
      [N, 6]
      [T, N, 6]
      [B, T, N, 6]
    """
    if point_cloud_seq.ndim == 2:
        return mask_green_blue_points(point_cloud_seq, **kwargs)

    elif point_cloud_seq.ndim == 3:
        return np.stack([
            mask_green_blue_points(point_cloud_seq[t], **kwargs)
            for t in range(point_cloud_seq.shape[0])
        ], axis=0)

    elif point_cloud_seq.ndim == 4:
        return np.stack([
            np.stack([
                mask_green_blue_points(point_cloud_seq[b, t], **kwargs)
                for t in range(point_cloud_seq.shape[1])
            ], axis=0)
            for b in range(point_cloud_seq.shape[0])
        ], axis=0)

    else:
        raise ValueError(f"Unsupported point cloud shape: {point_cloud_seq.shape}")


def debug_color_stats(point_cloud: np.ndarray, prefix: str = ""):
    is_red, is_green, is_blue, is_neutral = classify_color_points(point_cloud)
    print(
        f"{prefix} total={len(point_cloud)} "
        f"red={is_red.sum()} green={is_green.sum()} "
        f"blue={is_blue.sum()} neutral={is_neutral.sum()}"
    )