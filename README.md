
## Reproduce networks

Reproduce networks and train in cifar10

### Requirements

my experiment environments

* python3.6
* pytorch1.5
* torchsummary

### Usage

* train

```
./train.sh
```

* test

```
python test.py
```

### Training Details

* input size: 224x224
* lr_scheduler: MultiStepLR with init lr 0.01
* epoch: 100
* optimizer: SGD with 0.95 momentum

### Results.

| models    | batch_size | test_accuracy | params |
| --------- | ---------- | ------------- | ------ |
| resnet18  | 64         | 92.144        | 11.2M  |
| resnet34  | 64         | 92.711        | 21.2M  |
| resnet50  | 32         | 92.133        | 23.5M  |
| resnet101 | 16         | 91.711        | 42.5M  |
| resnet152 |            | TODO          |        |
| TODO      |            | TODO          |        |

