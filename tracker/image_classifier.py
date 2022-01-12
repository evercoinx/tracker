import re
from glob import glob
from typing import List, Optional, Tuple

import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from tracker.error import DatasetImageError


class ImageClassifier:
    PLACEHOLDER_LABEL = "0x"

    dataset_path: str
    image_format: str
    model: Optional[KNeighborsClassifier]

    def __init__(self, dataset_path: str, image_format: str) -> None:
        self.dataset_path = dataset_path
        self.image_format = image_format
        self.model = None

    def train(self) -> None:
        image_path_pattern = re.compile(
            r"/([\d\w]{2})_\d." + re.escape(self.image_format) + r"$",
            flags=re.IGNORECASE,
        )
        image_paths = glob(
            f"{self.dataset_path}/*.{self.image_format}", recursive=False
        )

        features: List[np.ndarray] = []
        labels: List[str] = []

        for p in sorted(image_paths):
            img = cv2.imread(p)
            if img is None:
                raise DatasetImageError(f"unable to read table card image path {p}")

            feat = self.extract_feature(img)
            features.append(feat)

            matches = re.findall(image_path_pattern, p)
            if not matches:
                raise DatasetImageError(f"unable to parse table card image path {p}")
            labels.append(matches[0])

        self.model = KNeighborsClassifier(n_neighbors=1, n_jobs=1)
        self.model.fit(features, labels)

    def predict(self, image: np.ndarray) -> str:
        feat = self.extract_feature(image)
        labels = self.model.predict([feat])
        return "" if labels[0] == ImageClassifier.PLACEHOLDER_LABEL else labels[0]

    @staticmethod
    def extract_feature(image: np.ndarray, bins: Tuple = (8,)) -> np.ndarray:
        hist = cv2.calcHist([image], [0], None, bins, [0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()
