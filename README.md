## About The Project

Basic Actor Critic implementation in JAX
Code for the blog post https://gabrielkeith.dev/blog/basic-actor-critic

Train agents to solve simple gym tasks!

[rl-video-episode-0.mp4](/assets/rl-video-episode-0.mp4)

## Prerequisites
* Python 3.12
* Poetry 1.8.3

## Install
```bash
poetry install
```

## Train an agent
```bash
train
```

You should now see a reward printed to the terminal for the cart pole problem!
This should only take a laptop cpu around 10 minutes to finish.

When training is finished, a video should be recorded to the output folder
