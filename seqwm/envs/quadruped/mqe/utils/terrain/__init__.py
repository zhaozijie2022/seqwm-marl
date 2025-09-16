import importlib

terrain_registry = dict(
    Terrain= "seqwm.envs.quadruped.mqe.utils.terrain.terrain:Terrain",
    BarrierTrack= "seqwm.envs.quadruped.mqe.utils.terrain.barrier_track:BarrierTrack",
    TerrainPerlin= "seqwm.envs.quadruped.mqe.utils.terrain.perlin:TerrainPerlin",
)

def get_terrain_cls(terrain_cls):
    entry_point = terrain_registry[terrain_cls]
    module, class_name = entry_point.rsplit(":", 1)
    module = importlib.import_module(module)
    return getattr(module, class_name)
