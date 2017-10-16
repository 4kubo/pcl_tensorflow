# Path Consistency Learning in Tensorflow

This is Tensorflow (partially using keras) implementation of PCL as described in [Bridging the Gap Between Value and Policy Based Reinforcement Learning](https://arxiv.org/abs/1702.08892).

## Requirement
Verified on the following envrironment,
- python 2.7
- tensorflow 1.3.0
- gym 0.9.2
- keras 2.0.5

## Usage
`python main.py --target_task CartPole-v0`

If you want to render, please add "-v" argument

`python main.py --target_task CartPole-v0 -v`

Verified work on the task "Copy-v0", but not as much as their report.

`python main --target_task Copy-v0 --tau 0.005 --gamma 0.9 --d  10 -b 400 --step_to_report 100 -r 5e-5 --start_at 2000  -c 0.5 --with_lstm`