import random

import cv2

__all__ = [
    'BorderValueMixin', 'FillValueMixin',
]


class BorderValueMixin:

    @property
    def pad_mode(self):
        return random.choice([
            cv2.BORDER_CONSTANT,
            cv2.BORDER_REPLICATE,
        ])

    @property
    def border_mode(self):
        return random.choice([
            cv2.BORDER_CONSTANT,
            cv2.BORDER_REPLICATE,
        ])

    @property
    def value(self):
        return [random.randint(0, 255) for _ in range(3)]

    @pad_mode.setter
    def pad_mode(self, x):
        return None

    @border_mode.setter
    def border_mode(self, x):
        return None

    @value.setter
    def value(self, x):
        return None


class FillValueMixin:

    @property
    def fill_value(self):
        return [random.randint(0, 255) for _ in range(3)]

    @fill_value.setter
    def fill_value(self, x):
        return None
