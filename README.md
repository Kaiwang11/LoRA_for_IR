# LoRA_for_IR
##
This is a repo for finetune LLM by LoRA on Retrieval task.
We design two methods to re-weight the sentence embedding by element-wise multiplication with LoRA matrix.
The overview can be described by picture below:
<!-- ![](structure.png) -->
<img src="structure.png" alt="A beautiful landscape" width="300">


In this repo , we experiment the re-weighting  ability of LoRA module on sentence embedding generated by sentence-transformer. We provide two type of reweighting method : single and multiple. We also apply LoRA+ on our work.



- `-d : str` : The datasets include nfcorpus,fiqa,scifact,nq. default : nfcorpus
- `-p : bool` : Executing LoRA+ . Default False.
- `-exp_name : str` : Name of checkpoint directory.Default : lora_{exp_name}_epoch{}_{model name}_{dataset}. 
- `-exp_type: str ` :Trigger  our method single,multi or None which is pure LoRA . Default None .
- `-r : int ` : Rank of LoRA module. default 32.
- `-dora : bool` : Apply dora , can execute simultaneously with our method.default False.
- `-vera : bool`: Apply vera , can execute simultaneously with our method.default False.
- `model : bool`: Retriever model name . Default : distilbert-base-uncased
- `batch : int` : Batch size. Dafault 32.
- `-epoch : int ` : Epoch number. Default 10.
## Usage
```
conda create -n LoRA_for_IR python=3.10
conda activate LoRA_for_IR
git clone git@github.com:Kaiwang11/LoRA_for_IR.git
cd LoRA_for_IR
pip install -e.
```
### Data Loading

To download the data(fiqa,nq,nfcorpus,scifact), run the following command:
```
python data_load.py -d DATASET_NAME 
```

```bash
python src/main.py \
-epoch 1\
-d nfcorpus\ #dataset name
-exp_name experiment name\ #name of checkpoint.
-dora False\
-vera False\
```
For more help :
```bash
python src/main.py -h 
```

