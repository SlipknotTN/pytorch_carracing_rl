import argparse

def do_parsing():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Policy gradient PyTorch training script")
    parser.add_argument("--config_file", required=True, type=str, help="Output dir for training artifacts")
    parser.add_argument("--output_dir", required=True, type=str, help="Output directory ")
    parser.add_argument("--env_render", action="store_true", help="Render environment in GUI")
    parser.add_argument("--debug_state", action="store_true", help="Show last state frame in GUI")
    args = parser.parse_args()
    return args


def main():
    args = do_parsing()
    print(args)


if __name__ == "__main__":
    main()
