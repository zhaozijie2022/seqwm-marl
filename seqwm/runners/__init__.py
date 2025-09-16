"""Runner registry."""
from seqwm.runners.world_model_runner import WorldModelRunner

RUNNER_REGISTRY = {
    "seqwm": WorldModelRunner,
}
