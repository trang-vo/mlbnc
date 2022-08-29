from .subtour import SubtourStateExtractor, PriorSubtourStateExtractor
from .cycle import *

STATE_EXTRACTOR_NAME = {
    "subtour": {
        "default": SubtourStateExtractor,
        "SubtourStateExtractor": SubtourStateExtractor,
        "PriorSubtourStateExtractor": PriorSubtourStateExtractor,
    },
    "cycle": {
        "default": CycleStateExtractor,
        "CycleStateExtractor": CycleStateExtractor,
        "PriorCycleStateExtractor": PriorCycleStateExtractor,
    },
}