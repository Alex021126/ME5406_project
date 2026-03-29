import argparse

from src.training import resume_sac


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume SAC training from an existing checkpoint.")
    parser.add_argument("checkpoint_path")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--obstacles", type=int, default=1)
    parser.add_argument("--output", default="")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    path = resume_sac(
        checkpoint_path=args.checkpoint_path,
        total_timesteps=args.timesteps,
        obstacle_count=args.obstacles,
        output_path=args.output or None,
        device=args.device,
    )
    print(f"Saved resumed model to {path}")
