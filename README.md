# Path Consistency Learning in Tensorflow

This is Tensorflow implementation of PCL as described in [Bridging the Gap Between Value and Policy Based Reinforcement Learning](https://arxiv.org/abs/1702.08892).

## Requirement
Verified on the following envrironment,
- python 2.7
- tensorflow 1.3.0
- gym 0.9.2

## Usage
Currently, in this implementation, only the task "CartPole-v0" was verified to work

`python main.py --target_task CartPole-v0`

If you want to render, please add "-v" argument

`python main.py --target_task CartPole-v0 -v`