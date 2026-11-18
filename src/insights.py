"""
insights.py: numbers behind the data visualisation page.

For a sample of images per class it measures four simple features
(brightness, greenness, yellow ratio, contrast, see preprocessing.image_statistics)
and writes a short interpretation for each, based on the actual numbers.
Results are cached in memory and refreshed when the training data changes.
"""

import numpy as np

from src.config import CLASS_NAMES, TRAIN_DIR
from src.preprocessing import image_statistics, list_labelled_images

_cache = {"key": None, "data": None}
SAMPLE_PER_CLASS = 60

FEATURES = {
    "brightness": "Brightness (mean grayscale pixel, 0 to 255)",
    "greenness": "Greenness, G / (R + G + B)",
    "yellow_ratio": "Yellow / brown pixel share",
    "contrast": "Contrast (grayscale standard deviation)",
}


def _interpret(feature, means):
    hi = max(means, key=means.get)
    lo = min(means, key=means.get)
    h, l = means[hi], means[lo]
    healthy = means.get("Healthy")
    if feature == "brightness":
        return (f"{hi} photos are the brightest on average ({h:.0f}) and {lo} the darkest ({l:.0f}). "
                "Brightness mostly reflects lighting and background in the field, not the disease, "
                "so the model should not rely on it. This is why every image is rescaled the same "
                "way and why mixed lighting in new uploads is useful training data.")
    if feature == "greenness":
        txt = (f"{hi} leaves are the greenest ({h:.3f}) and {lo} the least green ({l:.3f}). ")
        if healthy is not None:
            below = [c for c, v in means.items() if c != "Healthy" and v < healthy]
            if below:
                txt += (f"{', '.join(below)} {'is' if len(below) == 1 else 'are'} less green than Healthy "
                        f"({healthy:.3f}), which fits disease destroying chlorophyll. ")
        txt += "The differences are small, so colour alone cannot separate the classes."
        return txt
    if feature == "yellow_ratio":
        txt = (f"{hi} has the largest share of yellow or brown pixels ({h:.1%}) and {lo} the smallest "
               f"({l:.1%}). ")
        if healthy is not None:
            txt += (f"Healthy leaves sit at {healthy:.1%}. Yellowing and browning are visible symptoms "
                    "of mosaic, mottle and streak diseases, so this feature carries real signal. ")
        txt += "Some yellow also comes from soil and dry grass in the background."
        return txt
    if feature == "contrast":
        return (f"{hi} images have the most contrast ({h:.1f}) and {lo} the least ({l:.1f}). "
                "Higher contrast means more texture: spots, streaks and patchy colour. Mosaic and "
                "blight produce patchy leaves, so texture is one of the patterns MobileNetV2 "
                "picks up, which colour averages alone miss.")
    return ""


def compute_insights(force=False):
    paths, labels = list_labelled_images(TRAIN_DIR)
    key = (len(paths), paths[-1].name if paths else None)
    if not force and _cache["key"] == key:
        return _cache["data"]

    counts = {c: int((labels == i).sum()) for i, c in enumerate(CLASS_NAMES)}
    rng = np.random.default_rng(0)
    per_class = {c: {f: [] for f in FEATURES} for c in CLASS_NAMES}
    for i, c in enumerate(CLASS_NAMES):
        idx = np.where(labels == i)[0]
        if len(idx) > SAMPLE_PER_CLASS:
            idx = rng.choice(idx, SAMPLE_PER_CLASS, replace=False)
        for j in idx:
            try:
                stats = image_statistics(paths[j])
            except Exception:
                continue
            for f in FEATURES:
                per_class[c][f].append(stats[f])

    features = {}
    for f, title in FEATURES.items():
        means = {c: float(np.mean(v[f])) for c, v in per_class.items() if v[f]}
        if not means:
            continue
        features[f] = {
            "title": title,
            "mean": means,
            "std": {c: float(np.std(per_class[c][f])) for c in means},
            "values": {c: [round(x, 4) for x in per_class[c][f]] for c in means},
            "interpretation": _interpret(f, means),
        }

    total = sum(counts.values())
    top = max(counts, key=counts.get) if total else None
    low = min(counts, key=counts.get) if total else None
    if total and counts[top] == counts[low]:
        sample_txt = f"This sample has {total} training images, the same number for every class. "
    elif total:
        sample_txt = (f"This sample has {total} training images. {top} is the most common class "
                      f"({counts[top]}) and {low} the rarest ({counts[low]}). ")
    data = {
        "class_counts": counts,
        "class_interpretation": (
            sample_txt + "In the full Kaggle dataset "
            "CMD is about 61% of all 21,397 images and CBB only about 5%. Because of this "
            "imbalance, the model uses class weights and is judged on macro F1, which treats "
            "every class equally, instead of accuracy alone." if total else "No training images yet."),
        "features": features,
    }
    _cache.update(key=key, data=data)
    return data
