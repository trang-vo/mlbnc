from .feature_extractors import *


EXTRACTOR_NAME = {
    "GNNGraphExtractor": GNNGraphExtractor,
    "GINEGraphExtractor": GINEGraphExtractor,
    "MLP": MLP,
}
