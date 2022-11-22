from .feature_extractors import *


FEATURE_EXTRACTOR_NAME = {
    "GNNGraphExtractor": GNNGraphExtractor,
    "GINEGraphExtractor": GINEGraphExtractor,
    "MLP": MLP,
    "MLPOneLayer": MLPOneLayer
}
