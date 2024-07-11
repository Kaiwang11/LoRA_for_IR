# LoRA_for_IR


## Usage
```
conda create -n LoRA_for_IR python=3.10
conda activate LoRA_for_IR
git clone git@github.com:Kaiwang11/LoRA_for_IR.git
cd LoRA_for_IR
```
pip install -e.
```
### Data Loading

To download the data, run the following command:

```bash
python data_load.py

```

```bash
python src/main.py \
-epoch 1\
-d nfcorpus\ #dataset name
-exp_name experiment name\ #any name of checkpoint.
-dora False\
-vera False\
```
For more help :
```
```bash
python src/main.py -h 
```

