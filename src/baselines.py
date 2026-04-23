from __future__ import annotations

import time
from statistics import mean, pstdev

import mujoco
import numpy as np

from src.envs.obstacle_avoidance_env import EnvConfig, ObstacleAvoidanceArmEnv


def ik_velocity_baseline(
    env: ObstacleAvoidanceArmEnv,
    target_position: np.ndarray | None = None,
    gain: float = 1.5,
) -> np.ndarray:
    if target_position is None:
        target_position = env.data.mocap_pos[0]
    rel_target = target_position - env.data.site_xpos[env.ee_site_id]
    jac_pos = np.zeros((3, env.model.nv))
    mujoco.mj_jacSite(env.model, env.data, jac_pos, None, env.ee_site_id)
    dq = gain * np.linalg.pinv(jac_pos[:, : env.model.nu]) @ rel_target
    return np.clip(dq, -1.0, 1.0)


def _action_trace_metrics(actions: list[np.ndarray]) -> tuple[float, float]:
    if not actions:
        return 0.0, 0.0
    action_array = np.asarray(actions, dtype=np.float64)
    action_variance = float(np.mean(np.var(action_array, axis=0)))
    if len(action_array) < 2:
        return action_variance, 0.0
    delta_array = np.diff(action_array, axis=0)
    action_delta_variance = float(np.mean(np.var(delta_array, axis=0)))
    return action_variance, action_delta_variance


def run_ik_episode(obstacle_count: int = 3, max_steps: int = 200) -> dict:
    env = ObstacleAvoidanceArmEnv(config=EnvConfig(obstacle_count=obstacle_count))
    env.reset()
    done = False
    truncated = False
    total_reward = 0.0
    steps = 0
    info = {}
    actions: list[np.ndarray] = []
    first_action_latency_sec = 0.0

    while not done and not truncated and steps < max_steps:
        t0 = time.perf_counter()
        action = ik_velocity_baseline(env)
        if steps == 0:
            first_action_latency_sec = time.perf_counter() - t0
        _, reward, done, truncated, info = env.step(action)
        actions.append(action.copy())
        total_reward += reward
        steps += 1

    action_variance, action_delta_variance = _action_trace_metrics(actions)
    env.close()
    return {
        "success": bool(info.get("success", False)),
        "collision": bool(info.get("collision", False)),
        "return": total_reward,
        "steps": steps,
        "planning_latency_sec": first_action_latency_sec,
        "action_variance": action_variance,
        "action_delta_variance": action_delta_variance,
        "planner_solved": True,
    }


def _segment_sphere_intersection(a: np.ndarray, b: np.ndarray, center: np.ndarray, radius: float) -> bool:
    segment = b - a
    denom = float(np.dot(segment, segment))
    if denom <= 1e-12:
        return float(np.linalg.norm(center - a)) <= radius
    t = float(np.dot(center - a, segment) / denom)
    t = min(1.0, max(0.0, t))
    closest = a + t * segment
    return float(np.linalg.norm(center - closest)) <= radius


def _segment_free(a: np.ndarray, b: np.ndarray, obstacles: list[np.ndarray], radius: float) -> bool:
    for center in obstacles:
        if _segment_sphere_intersection(a, b, center, radius):
            return False
    return True


def _rrt_star_plan_cartesian(
    start: np.ndarray,
    goal: np.ndarray,
    config: EnvConfig,
    obstacle_centers: list[np.ndarray],
    obstacle_radius: float,
    max_iter: int = 1800,
    step_size: float = 0.08,
    rewire_radius: float = 0.18,
    goal_bias: float = 0.2,
) -> tuple[list[np.ndarray], bool]:
    low = np.array(
        [
            min(config.target_x_bounds[0], config.obstacle_x_bounds[0]),
            min(config.target_y_bounds[0], config.obstacle_y_bounds[0]),
            min(config.target_z_bounds[0], config.obstacle_z_bounds[0]),
        ],
        dtype=np.float64,
    )
    high = np.array(
        [
            max(config.target_x_bounds[1], config.obstacle_x_bounds[1]),
            max(config.target_y_bounds[1], config.obstacle_y_bounds[1]),
            max(config.target_z_bounds[1], config.obstacle_z_bounds[1]),
        ],
        dtype=np.float64,
    )
    safety_radius = obstacle_radius + 0.02
    goal_tolerance = config.goal_tolerance
    rng = np.random.default_rng()

    nodes = [
        {
            "pos": start.copy(),
            "parent": -1,
            "cost": 0.0,
        }
    ]
    goal_idx = -1

    for _ in range(max_iter):
        if rng.random() < goal_bias:
            sample = goal.copy()
        else:
            sample = rng.uniform(low=low, high=high).astype(np.float64)

        node_positions = np.asarray([node["pos"] for node in nodes])
        nearest_idx = int(np.argmin(np.linalg.norm(node_positions - sample, axis=1)))
        nearest_pos = nodes[nearest_idx]["pos"]
        delta = sample - nearest_pos
        dist = float(np.linalg.norm(delta))
        if dist < 1e-9:
            continue
        new_pos = nearest_pos + delta / dist * min(step_size, dist)
        if not _segment_free(nearest_pos, new_pos, obstacle_centers, safety_radius):
            continue

        nearby = np.where(np.linalg.norm(node_positions - new_pos, axis=1) <= rewire_radius)[0]
        parent = nearest_idx
        best_cost = float(nodes[nearest_idx]["cost"] + np.linalg.norm(new_pos - nearest_pos))
        for idx in nearby:
            parent_pos = nodes[int(idx)]["pos"]
            if not _segment_free(parent_pos, new_pos, obstacle_centers, safety_radius):
                continue
            candidate_cost = float(nodes[int(idx)]["cost"] + np.linalg.norm(new_pos - parent_pos))
            if candidate_cost < best_cost:
                parent = int(idx)
                best_cost = candidate_cost

        nodes.append({"pos": new_pos, "parent": parent, "cost": best_cost})
        new_idx = len(nodes) - 1

        for idx in nearby:
            idx = int(idx)
            if idx == new_idx:
                continue
            candidate_cost = float(best_cost + np.linalg.norm(nodes[idx]["pos"] - new_pos))
            if candidate_cost >= float(nodes[idx]["cost"]):
                continue
            if not _segment_free(new_pos, nodes[idx]["pos"], obstacle_centers, safety_radius):
                continue
            nodes[idx]["parent"] = new_idx
            nodes[idx]["cost"] = candidate_cost

        if float(np.linalg.norm(new_pos - goal)) <= goal_tolerance and _segment_free(
            new_pos, goal, obstacle_centers, safety_radius
        ):
            nodes.append(
                {
                    "pos": goal.copy(),
                    "parent": new_idx,
                    "cost": float(best_cost + np.linalg.norm(goal - new_pos)),
                }
            )
            goal_idx = len(nodes) - 1
            break

    if goal_idx == -1:
        return [start.copy(), goal.copy()], False

    path = []
    cursor = goal_idx
    while cursor != -1:
        path.append(nodes[cursor]["pos"].copy())
        cursor = int(nodes[cursor]["parent"])
    path.reverse()
    return path, True


def run_rrt_star_episode(obstacle_count: int = 3, max_steps: int = 200) -> dict:
    env = ObstacleAvoidanceArmEnv(config=EnvConfig(obstacle_count=obstacle_count))
    env.reset()
    start = env.data.site_xpos[env.ee_site_id].copy()
    goal = env.data.mocap_pos[0].copy()
    active = min(env.config.obstacle_count, len(env.obstacle_body_ids))
    obstacle_centers = [env.data.mocap_pos[1 + i].copy() for i in range(active)]

    t_plan = time.perf_counter()
    cartesian_path, solved = _rrt_star_plan_cartesian(
        start=start,
        goal=goal,
        config=env.config,
        obstacle_centers=obstacle_centers,
        obstacle_radius=env.config.obstacle_radius,
    )
    planning_latency_sec = time.perf_counter() - t_plan

    done = False
    truncated = False
    total_reward = 0.0
    steps = 0
    info = {}
    waypoint_idx = 1
    actions: list[np.ndarray] = []
    first_action_latency_sec = 0.0

    while not done and not truncated and steps < max_steps:
        if solved and waypoint_idx < len(cartesian_path):
            current_ee = env.data.site_xpos[env.ee_site_id].copy()
            if float(np.linalg.norm(current_ee - cartesian_path[waypoint_idx])) <= env.config.goal_tolerance:
                waypoint_idx += 1
            target_waypoint = cartesian_path[min(waypoint_idx, len(cartesian_path) - 1)]
        else:
            target_waypoint = env.data.mocap_pos[0].copy()

        t0 = time.perf_counter()
        action = ik_velocity_baseline(env, target_position=target_waypoint, gain=1.2)
        if steps == 0:
            first_action_latency_sec = time.perf_counter() - t0
        _, reward, done, truncated, info = env.step(action)
        actions.append(action.copy())
        total_reward += reward
        steps += 1

    action_variance, action_delta_variance = _action_trace_metrics(actions)
    env.close()
    return {
        "success": bool(info.get("success", False)),
        "collision": bool(info.get("collision", False)),
        "return": total_reward,
        "steps": steps,
        "planning_latency_sec": planning_latency_sec,
        "first_action_latency_sec": first_action_latency_sec,
        "action_variance": action_variance,
        "action_delta_variance": action_delta_variance,
        "planner_solved": solved,
    }


def evaluate_ik_baseline(episodes: int = 50, obstacle_count: int = 3) -> dict:
    runs = [run_ik_episode(obstacle_count=obstacle_count) for _ in range(episodes)]
    returns = [run["return"] for run in runs]
    steps = [run["steps"] for run in runs]
    latencies = [run["planning_latency_sec"] for run in runs]
    action_variances = [run["action_variance"] for run in runs]
    action_delta_variances = [run["action_delta_variance"] for run in runs]
    return {
        "episodes": episodes,
        "obstacle_count": obstacle_count,
        "success_rate": mean(int(run["success"]) for run in runs),
        "collision_rate": mean(int(run["collision"]) for run in runs),
        "mean_return": mean(returns),
        "return_std": pstdev(returns) if len(returns) > 1 else 0.0,
        "mean_episode_steps": mean(steps),
        "step_std": pstdev(steps) if len(steps) > 1 else 0.0,
        "mean_planning_latency_sec": mean(latencies),
        "mean_action_variance": mean(action_variances),
        "mean_action_delta_variance": mean(action_delta_variances),
    }


def evaluate_rrt_star_baseline(episodes: int = 50, obstacle_count: int = 3) -> dict:
    runs = [run_rrt_star_episode(obstacle_count=obstacle_count) for _ in range(episodes)]
    returns = [run["return"] for run in runs]
    steps = [run["steps"] for run in runs]
    planning_latencies = [run["planning_latency_sec"] for run in runs]
    first_action_latencies = [run["first_action_latency_sec"] for run in runs]
    action_variances = [run["action_variance"] for run in runs]
    action_delta_variances = [run["action_delta_variance"] for run in runs]
    solved_rate = mean(int(run["planner_solved"]) for run in runs)
    return {
        "episodes": episodes,
        "obstacle_count": obstacle_count,
        "success_rate": mean(int(run["success"]) for run in runs),
        "collision_rate": mean(int(run["collision"]) for run in runs),
        "mean_return": mean(returns),
        "return_std": pstdev(returns) if len(returns) > 1 else 0.0,
        "mean_episode_steps": mean(steps),
        "step_std": pstdev(steps) if len(steps) > 1 else 0.0,
        "mean_planning_latency_sec": mean(planning_latencies),
        "mean_first_action_latency_sec": mean(first_action_latencies),
        "mean_action_variance": mean(action_variances),
        "mean_action_delta_variance": mean(action_delta_variances),
        "planner_solved_rate": solved_rate,
    }
