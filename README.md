## Policy Gradient
Experiment with Policy Gradient methods ([description](https://spinningup.openai.com/en/latest/algorithms/vpg.html)), as well as variance reduction.

Current implementation:
- Continuous and discrete environments
- Baseline network for variance reduction

## Usage

#### Setup environment
```
$ conda env create -f [environment.yml | environment_cuda.yml]
$ conda activate [policy_grad | policy_grad_cuda]
```

#### Run training

```
$ python main.py --config_filename config_filename
```