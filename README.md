# Transfer Learning Implementation

## Create Environment

```bash
conda create --prefix ./envs python==3.7.9 -y
```

## Activate Environment

```bash
conda activate ./envs
```

## Install Requirements/Libs

```bash
pip install -r requiements.txt
```

## Train the Base model
```bash
python src/01_base_model_creation.py
```
## Train the New model using transfer learning
```bash
python src/02_transfer_learning_even_odd.py
```

