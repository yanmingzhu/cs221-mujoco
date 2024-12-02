import argparse
from util import train_agent, replay_agent
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=str,
        help="",
    )
    parser.add_argument(
        "--mode",
        type=str,
        help="",
    )
    parser.add_argument(
        "-n",
        type=str,
        help="",
    )
    args = parser.parse_args()

    mode = "train"
    if args.mode is not None:
        mode = args.mode
    
    max_steps = 5e6 if args.n is None else int(args.n)
    seed = random.randint(1, 500) if args.seed is None else int(args.seed)
    if mode == "train":
        rewards = train_agent(seed, max_steps)
    else:
        replay_agent()