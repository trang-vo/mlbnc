from .subtour import SubtourStateExtractor
from .cycle import CycleStateExtractor

STATE_EXTRACTOR_NAME = {
    "subtour": SubtourStateExtractor,
    "cycle": CycleStateExtractor,
}
