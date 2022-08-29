from .base import *
from .env_callback import EnvUserCallback


CALLBACK_NAME = {
    "BaseUserCallback": BaseUserCallback,
    "RLUserCallback": RLUserCallback,
    "EnvUserCallback": EnvUserCallback,
}
