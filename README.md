# FedALG

The implementation of some federated learning algorithms  based on PyTorch respectively.

### Environment

```bash
conda create -n FL python=3.7

conda activate FL

conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
(modify pytorch and cuda version according to your device)

pip install -r requirements.txt

```

### Datasets

1. MINST
3. EMINST
4. Cifar10

### Usage

Run the code

```asp
python server.py -config_path ./configs/fedavg.json
```
