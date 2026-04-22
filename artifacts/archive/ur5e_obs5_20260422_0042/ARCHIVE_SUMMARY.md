# UR5e Obstacle 5 Archive

Archived on 2026-04-22 00:42 Asia/Singapore.

## Model

- `models/sac_her_ur5e_obs5_seed42.zip`
- Robot: Universal Robots UR5e from MuJoCo Menagerie
- Algorithm: SAC with HER replay
- Obstacles: 5
- Seed: 42

## Evaluation

50 episodes:

- Success rate: 0.76
- Collision rate: 0.28
- Mean return: -10.8203
- Mean episode steps: 4.76
- Mean planning latency: 0.000168 s
- Mean action variance: 0.3007
- Mean action delta variance: 0.6836

## Included Files

- `results/eval_ur5e_obs5.json`
- `results/rollout_ur5e_obs5_slow.gif`
- `models/sac_her_ur5e_obs5_seed42.zip`
- `config/ur5e_obstacle_scene.xml`
- `config/obstacle_avoidance_env.py`
- `config/training.py`
- `config/policies.py`
