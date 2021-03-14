"""
Visualize recorded experience.
Format: (s, a, r, s', exp_weight) tuples
"""
import argparse
import pickle

import cv2
import numpy as np


def do_parsing():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Experience memory analyzer script")
    parser.add_argument("--experience_file", required=True, type=str, help="Experience pkl filepath")
    args = parser.parse_args()
    return args


def main():
    args = do_parsing()
    print(args)

    with open(args.experience_file, "rb") as in_fp:
        experience_buffer = pickle.load(in_fp)

    for ix in range(0, 1000):
        experience = experience_buffer.sample(batch_size=1)[0]
        state = experience[0]
        weight = experience[-1]
        last_frame = state[-1]
        last_frame_np = last_frame.cpu().data.numpy()[0]
        print(f"Frame mean value after processing: {np.mean(last_frame_np)}")
        print(f"Frame weight: {weight}")
        last_frame_np += 1.0
        last_frame_np /= 2.0
        cv2.imshow("Experience frame", last_frame_np)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
