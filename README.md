# ICPF: In-Context Prompt-augmented Framework [AAAI2025]

> Official github repository for **In-context Prompt-augmented Micro-video Popularity Prediction**.

## 🛠️ Environment
* Python >= 3.10
* Pytorch >= 2.2.0
* CUDA Version >= 11.6
```bash
cd ICPF
conda env create -f environment.yml
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia(If torch is not installed)
```

## 🔥 Training and Testing ICPF

For example, to **train** ICPF with one 3090 GPU.
* First complete the [dataset preparation and retrieve section](retriever/Readme.md), and then configure the processed dataset (pkl file) path in the [config files](config).
* Then you can run

```bash 
python train.py --dataset_id="microlens" --dataset="MICROLENS"
```
* When the training is completed, the inference results will be obtained.
