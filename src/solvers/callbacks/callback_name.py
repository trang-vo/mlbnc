from .base import *
from .env_callback import *


CALLBACK_NAME = {
    "BaseUserCallback": BaseUserCallback,
    "RLUserCallback": RLUserCallback,
    "EnvUserCallback": EnvUserCallback,
    "RandomUserCallback": RandomUserCallback,
    "PreprocessUserCallback": PreprocessUserCallback,
    "ArbitraryStartEnvUserCallback": ArbitraryStartEnvUserCallback,
    "MLUserCallback": MLUserCallback,
}
