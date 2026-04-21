from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import time
import zipfile

import numpy as np
import torch
from stable_baselines3 import SAC

from src.envs.obstacle_avoidance_env import EnvConfig, ObstacleAvoidanceArmEnv


def model_uses_goal_conditioning(model_path: str) -> bool:
    with zipfile.ZipFile(model_path) as archive:
        data = archive.read("data").decode("utf-8")
    return "MultiInputPolicy" in data or "HerReplayBuffer" in data


def evaluate_sac(
    model_path: str,
    episodes: int = 20,
    obstacle_count: int = 3,
    device: str | None = None,
) -> dict:
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {resolved_device}")
    goal_conditioned = model_uses_goal_conditioning(model_path)
    env = ObstacleAvoidanceArmEnv(
        config=EnvConfig(obstacle_count=obstacle_count),
        goal_conditioned=goal_conditioned,
    )
    model = SAC.load(model_path, env=env, device=resolved_device)

    successes = 0
    collisions = 0
    returns = []
    steps = []
    planning_latencies = []
    action_variances = []
    action_delta_variances = []

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_return = 0.0
        episode_steps = 0
        first_action_latency_sec = 0.0
        actions: list[np.ndarray] = []

        while not done and not truncated:
            inference_t0 = time.perf_counter()
            action, _ = model.predict(obs, deterministic=True)
            if episode_steps == 0:
                first_action_latency_sec = time.perf_counter() - inference_t0
            obs, reward, done, truncated, info = env.step(action)
            episode_return += reward
            episode_steps += 1
            actions.append(np.asarray(action, dtype=np.float64))

        successes += int(info["success"])
        collisions += int(info["collision"])
        returns.append(episode_return)
        steps.append(episode_steps)
        planning_latencies.append(first_action_latency_sec)
        action_array = np.asarray(actions, dtype=np.float64)
        action_variances.append(float(np.mean(np.var(action_array, axis=0))) if len(action_array) else 0.0)
        if len(action_array) > 1:
            delta_array = np.diff(action_array, axis=0)
            action_delta_variances.append(float(np.mean(np.var(delta_array, axis=0))))
        else:
            action_delta_variances.append(0.0)

    env.close()
    return {
        "model_path": model_path,
        "episodes": episodes,
        "success_rate": successes / episodes,
        "collision_rate": collisions / episodes,
        "mean_return": float(np.mean(returns)),
        "mean_episode_steps": float(np.mean(steps)),
        "return_std": float(np.std(returns)),
        "step_std": float(np.std(steps)),
        "mean_planning_latency_sec": float(np.mean(planning_latencies)),
        "mean_action_variance": float(np.mean(action_variances)),
        "mean_action_delta_variance": float(np.mean(action_delta_variances)),
        "goal_conditioned": goal_conditioned,
        "config": asdict(env.config),
    }


def save_metrics(metrics: dict, output_path: str) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return output
