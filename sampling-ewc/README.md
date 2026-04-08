# Experiments on the mid-angle sampling and EWC-Replay methods.

1. Mid-angle sampling vs. original sampling across various CL replay frameworks:

```
python main.py --model er --dataset seq-cifar10 --buffer_size 1000 --lr 0.1 --seed 99 --device 0 --enable_other_metrics 1
python main.py --model er_mid_angle --dataset seq-cifar10 --buffer_size 1000 --lr 0.1 --seed 99 --device 0 --enable_other_metrics 1
python main.py --model gem --dataset seq-cifar10 --model_config best --buffer_size 1000 --seed 99 --device 0 --enable_other_metrics 1
python main.py --model gem_mid --dataset seq-cifar10 --model_config best --buffer_size 1000 --device 0 --enable_other_metrics 1
python main.py --model agem --dataset seq-cifar10 --model_config best --buffer_size 1000 --seed 99 --device 0 --enable_other_metrics 1
python main.py --model agem_mid --dataset seq-cifar10 --model_config best --buffer_size 1000 --seed 99 --device 0 --enable_other_metrics 1
python main.py --model rpc --dataset seq-cifar10 --model_config best --buffer_size 1000 --seed 99 --device 0 --enable_other_metrics 1
python main.py --model rpc_mid --dataset seq-cifar10 --model_config best --buffer_size 1000 --seed 99 --device 0 --enable_other_metrics 1
python main.py --model fdr --dataset seq-cifar10 --model_config best --buffer_size 1000 --seed 99 --device 0 --enable_other_metrics 1
python main.py --model fdr_mid --dataset seq-cifar10 --model_config best --buffer_size 1000 --seed 99 --device 0 --enable_other_metrics 1
```

2. Different replay buffer sampling methods within ER framework:

```
python main.py --model er --dataset seq-cifar10 --buffer_size 2000 --lr 0.1 --seed 99 --device 0 --enable_other_metrics 1
python main.py --model er_maxent --dataset seq-cifar10 --buffer_size 2000 --lr 0.1 --seed 99 --device 0 --enable_other_metrics 1
python main.py --model er_ipm --dataset seq-cifar10 --buffer_size 2000 --lr 0.1 --seed 99 --device 0 --enable_other_metrics 1
python main.py --model er_gss --dataset seq-cifar10 --buffer_size 2000 --lr 0.1 --seed 99 --device 0 --enable_other_metrics 1
python main.py --model er_lars --dataset seq-cifar10 --buffer_size 2000 --lr 0.1 --seed 99 --device 0 --enable_other_metrics 1
python main.py --model er_big_angle --dataset seq-cifar10 --buffer_size 2000 --lr 0.1 --seed 99 --device 0 --enable_other_metrics 1
python main.py --model er_small_angle --dataset seq-cifar10 --buffer_size 2000 --lr 0.1 --seed 99 --device 0 --enable_other_metrics 1
python main.py --model er_mid_angle --dataset seq-cifar10 --buffer_size 2000 --lr 0.1 --seed 99 --device 0 --enable_other_metrics 1
```

3. EWC-Replay method vs. original replay method across various CL replay frameworks:

```
python main.py --model er --dataset seq-cifar10 --buffer_size 1000 --lr 0.1 --seed 99 --device 0 --enable_other_metrics 1
python main.py --model er_ewc --dataset seq-cifar10 --model_config best --buffer_size 1000 --seed 99 --device 0 --enable_other_metrics 1
python main.py --model gss --dataset seq-cifar10 --model_config best --buffer_size 1000 --seed 99 --device 0 --enable_other_metrics 1
python main.py --model gss_ewc --dataset seq-cifar10 --model_config best --buffer_size 1000 --seed 99 --device 0 --enable_other_metrics 1
python main.py --model agem --dataset seq-cifar10 --model_config best --buffer_size 1000 --seed 99 --device 0 --enable_other_metrics 1
python main.py --model agem_ewc --dataset seq-cifar10 --model_config best --buffer_size 1000 --seed 99 --device 0 --enable_other_metrics 1
python main.py --model rpc --dataset seq-cifar10 --model_config best --buffer_size 1000 --seed 99 --device 0 --enable_other_metrics 1
python main.py --model rpc_ewc --dataset seq-cifar10 --model_config best --buffer_size 1000 --seed 99 --device 0 --enable_other_metrics 1
python main.py --model fdr --dataset seq-cifar10 --model_config best --buffer_size 1000 --seed 99 --device 0 --enable_other_metrics 1
python main.py --model fdr_ewc --dataset seq-cifar10 --model_config best --buffer_size 1000 --seed 99 --device 0 --enable_other_metrics 1
```

- `--model`: the name of the model
- `--dataset`: the name of the dataset
- `--buffer_size`: the size of the buffer
- `--enable_other_metrics`: including the average forgetting metric.