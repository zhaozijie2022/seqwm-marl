"""Train an algorithm."""
import isaacgym
import argparse
import json
import seqwm.envs.quadruped.mqe.envs.configs.go1_gate_config


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="M3W training script")
    parser.add_argument(
        "--load-config",
        type=str,
        default="configs/dexhands/ShadowHandBottleCap/seqwm/config.json",
        help="Path to config file"
    )

    with open(args["load_config"], encoding="utf-8") as file:
        all_config = json.load(file)
    args["algo"] = all_config["main_args"]["algo"]
    args["env"] = all_config["main_args"]["env"]
    args["exp_name"] = all_config["main_args"]["exp_name"]
    args["load_config"] = all_config["main_args"]["load_config"]
    algo_args = all_config["algo_args"]
    env_args = all_config["env_args"]

    from seqwm.runners import RUNNER_REGISTRY

    runner = RUNNER_REGISTRY[args["algo"]](args, algo_args, env_args)
    runner.run()
    runner.close()


if __name__ == "__main__":
    main()
