
## Installation

> [!IMPORTANT]
> It is recommended that you first install the version of `torch` 2.2.3 suitable to your platform by following the instructions [here](https://pytorch.org/get-started/previous-versions/#v230), or the command below may not work.
> When installing using the above link, add `torchtext==0.18.0` to the installation command. For example
> ```conda install pytorch==2.3.0 torchvision==0.18.0 torchtext==0.18.0 torchaudio==2.3.0 cpuonly -c pytorch```

```
pip install -e .
```

## Running the models

On CPU:
```
python3 run.py -d cpu -m  Transformer LSTM 
```

On Nvidia GPU:
```
python3 run.py -d cuda -m  Transformer LSTM 
```

You can exclude any of the models in the commands above and it will not be run.
