# UR5e Obstacle 3 Archive

Archived on 2026-04-21 21:53 Asia/Singapore.

## Model

- `models/sac_her_ur5e_obs3_seed42.zip`
- Robot: Universal Robots UR5e from MuJoCo Menagerie
- Algorithm: SAC with HER replay
- Obstacles: 3
- Seed: 42

## Evaluation

50 episodes:

- Success rate: 0.80
- Collision rate: 0.20
- Mean return: -7.8123
- Mean episode steps: 4.22
- Mean planning latency: 0.000464 s
- Mean action variance: 0.2050
- Mean action delta variance: 0.3568

## Included Files

- `results/eval_ur5e_obs3.json`
- `results/rollout_ur5e_obs3_slow.gif`
- `models/sac_her_ur5e_obs3_seed42.zip`
- `config/ur5e_obstacle_scene.xml`
- `config/obstacle_avoidance_env.py`
- `config/training.py`
- `config/policies.py`
