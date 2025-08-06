from . import *
from .world_model import WorldModel
from .beam_world_model import BeamSearchWorldModel

WORLD_MODELS = {
    'beam_search': BeamSearchWorldModel,
    'mcts' : WorldModel,
    }

def get_world_model(world_model_name):
    assert world_model_name in WORLD_MODELS.keys(), f"World model {world_model_name} is not supported."
    return WORLD_MODELS[world_model_name]