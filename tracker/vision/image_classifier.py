import re
from glob import glob
from typing import ClassVar, List, Tuple

import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from typing_extensions import Literal


class ImageClassifier:
    """Classifies board images based on a k-NN model"""

    placeholder_label: ClassVar[Literal["0x"]] = "0x"

    dataset_path: str
    image_format: str
    model: KNeighborsClassifier

    def __init__(self, dataset_path: str, image_format: str) -> None:
        self.dataset_path = dataset_path
        self.image_format = image_format
        self.model = KNeighborsClassifier(n_neighbors=1, n_jobs=1)

    def train(self) -> None:
        image_path_pattern = re.compile(
            r"/([\d\w]{2})_\d." + re.escape(self.image_format) + r"$",
            flags=re.IGNORECASE,
        )
        board_image_paths = glob(
            f"{self.dataset_path}/board/*.{self.image_format}", recursive=False
        )

        features: List[np.ndarray] = []
        labels: List[str] = []

        for p in sorted(board_image_paths):
            img = cv2.imread(p)
            if img is None:
                raise OSError(f"Unable to read board dataset image at {p}")

            feat = self._extract_feature(img)
            features.append(feat)

            matches = re.findall(image_path_pattern, p)
            if not matches:
                raise Exception(f"Unable to parse board dataset image at {p}")
            labels.append(matches[0])

        self.model.fit(features, labels)

    def classify(self, image: np.ndarray) -> str:
        feat = self._extract_feature(image)
        labels = self.model.predict([feat])
        return "" if labels[0] == type(self).placeholder_label else labels[0]

    @staticmethod
    def _extract_feature(image: np.ndarray, bins: Tuple[int] = (8,)) -> np.ndarray:
        hist = cv2.calcHist([image], [0], None, bins, [0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()
