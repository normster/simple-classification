Simple ImageNet classification script.

Developed for PyTorch 1.8.1 but should be compatible with several earlier versions.
[submitit](https://github.com/facebookincubator/submitit) is only required for multi-node training on [slurm](https://slurm.schedmd.com/quickstart.html).
[pytorch-image-models](https://github.com/rwightman/pytorch-image-models) can be replaced with torchvision.models

# Instructions

Launch single-node, single-gpu training with:

```
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py [--additional --flags --here]
```

Launch single-node, multi-gpu training with: 

```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py [--additional --flags --here]
```

Launch multi-node multi-gpu training (using submitit for slurm) with:

```
python run_with_submitit.py --nodes 2 --gpus 8 [--additional --flags --here]
```

Multi-node multi-gpu training is also possible via `main.py` by setting the appropriate `--rank` + `--world-size` flags or `RANK` + `WORLD_SIZE` environment variables and running `python -m torch.distributed.launch` once per node.