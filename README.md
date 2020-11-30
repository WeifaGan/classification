
## Reproduce networks

Reproduce networks and train in cifar10

### Requirements

my experiment environments

* python3.6
* pytorch1.5

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
* batch size: 64
* lr_scheduler:MultiStepLR with init lr 0.001
* epoch: 200
* optimizer: SGD with 0.95 momentum

### Results.

| models    | accuracy |
| --------- | -------- |
| resnet18  | 91.056   |
| resnet34  | TODO     |
| resnet50  | TODO     |
| resnet101 | TODO     |
| resnet152 | TODO     |
| TODO      | TODO     |

