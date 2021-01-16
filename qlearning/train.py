"""
TODO:
- Discretize action space (source file claims on/off is possible).
  - Steer: left, no, right.
  - Gas: no, yes.
  - Brake: no, yes.
- Implement model, 96x96x4 input should be ok (4 BW frames)
"""
import argparse

from qlearning.config import ConfigParams


def do_parsing():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Q-Learning PyTorch training script")
    parser.add_argument("--config_file", required=True, type=str, help="Path to the config file")
    args = parser.parse_args()
    return args


def main():
    args = do_parsing()
    print(args)

    config = ConfigParams(args.config_file)

    

if __name__ == "__main__":
    main()
