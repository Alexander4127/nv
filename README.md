# Mel2Speech Project

## Report

The introduction, pipeline details, experiments, and results are presented in the [wandb report](https://wandb.ai/practice-cifar/nv_project/reports/Vocoder-Project--Vmlldzo2MTMwNDM4).

## Installation guide

To get started install the requirements
```shell
pip install -r ./requirements.txt
```

Then download train data ([LJSpeech](https://keithito.com/LJ-Speech-Dataset/) dataset)
```shell
sudo apt install axel
bash loader.sh
```

## Model training

This project implements [HiFiGAN](https://arxiv.org/abs/2010.05646) model for speech synthesis.

To train model from scratch run
```shell
python3 train.py -c nv/configs/train.json
```

For fine-tuning pretrained model from checkpoint, `--resume` parameter is applied.
For example, continuing training model with `train.json` config organized as follows
```shell
python3 train.py -c nv/configs/train.json -r saved/models/final/<run_id>/<any_checkpoint>.pth
```

## Inference stage

Checkpoint should be located in `default_test_model` directory. Pretrained model can be downloaded by running python code
```python3
import gdown
gdown.download("https://drive.google.com/uc?id=1I5qPDu6Bsc_xm6u6U35e867RRNeqi0v3", "default_test_model/checkpoint.pth")
```

Model evaluation is executed by command
```shell
python3 test.py \
   -i default_test_model/text.txt \
   -r default_test_model/checkpoint.pth \
   -o output \
   -l False
```

- `-i` (`--input-dir`) provide the path to directory with input `.wav` files. Additionally, one `text.txt` file can be located there. In this case it will be read by rows (one row for each audio).
- `-r` (`--resume`) provide the path to model checkpoint. Note that config file is expected to be in the same dir with name `config.json`.
- `-o` (`--output`) specify output directory path, where `.wav` files will be saved.
- `-l` (`--log-wandb`) determine log results to wandb project or not. If `True`, authorization in command line is needed. Name of project can be changed in the config file. Lines from `text.txt` file are also logged, if it is provided.

Running with default parameters
```shell
python3 test.py
```

## Credits

The code of model is based on a [notebook](https://github.com/XuMuK1/dla2023/blob/2023/week07/seminar07.ipynb).
