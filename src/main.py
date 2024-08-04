from sentence_transformers import losses, models, SentenceTransformer
from beir import util, LoggingHandler
from beir.losses import BPRLoss,MarginMSELoss
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.train import TrainRetriever
import pathlib, os
import json
import logging
from time import time
import tracemalloc
import argparse
from peft import LoraConfig,LoftQConfig,get_peft_model,AdaLoraConfig,LoHaConfig,LoKrConfig,VeraConfig
import torch
from lora_plus import create_loraplus_optimizer

parser = argparse.ArgumentParser(description='Argument Parser for Dataset, Experiment Name, and Epoch Number')
parser.add_argument('-d', '--dataset', type=str, help='Name of the dataset',default='nfcorpus')
parser.add_argument('-exp_name', '--experiment_name', type=str, help='Name of the experiment',default='')
parser.add_argument('-exp_type', '--experiment_type', type=str, help='type of the experiment such as single,multi,None',default=None)
parser.add_argument('-epoch', '--epoch', type=int, help='Number of epochs',default=10)
parser.add_argument('-p','--plus',type=bool,help = 'Lora plus',default=False)
parser.add_argument('-r','--rank',type=int,help='rank',default=32)
parser.add_argument('-pool','--pool_mode',type=str,help='pooling_mode',default=None)

parser.add_argument('-model','--model_name',type=str,help='hugging face model name',default="distilbert-base-uncased")
parser.add_argument('-dora','--dora',type=bool,help='dora',default=False)
parser.add_argument('-vera','--vera',type=bool,help='vera',default=False)
parser.add_argument('-batch','--batch_size',type=int,help='batch zise',default=32)
parser.add_argument('-target_module','--target_module',type=str,help='target module',default='["q_lin","out_lin","k_lin","v_lin"]')
parser.add_argument('-lora_type','--lora_type',type=str,help='lora type : could be vera,dora,loha,lokr,adalora or none for lora',default=None)

args = parser.parse_args()
vera=False
dora=False
if args.lora_type=='vera':
    lora_config =VeraConfig(
                r=args.rank,
                target_modules=json.loads(args.target_module),
                # bias="all", 
                modules_to_save=["decode_head"],
            )
    vera=True
elif args.lora_type=='loha':
    lora_config=LoHaConfig(
        r=args.rank,
        lora_alpha=args.rank*2,
        target_modules=json.loads(args.target_module),
        modules_to_save=["decode_head"],
        rank_dropout=0.0,
        module_dropout=0.0,
        init_weights=True,
        use_effective_conv2d=True,
    )

elif args.lora_type=='lokr':
    lora_config=LoKrConfig(
        r=args.rank,
        lora_alpha=args.rank*2,
        target_modules=json.loads(args.target_module),
        modules_to_save=["decode_head"],
        rank_dropout=0.0,
        module_dropout=0.0,
        init_weights=True,
        use_effective_conv2d=True,
    )
elif args.lora_type=='adalora':

    lora_config = AdaLoraConfig(
         r=args.rank,
        lora_alpha=args.rank*2,
        target_modules=json.loads(args.target_module),
        lora_dropout=0.5,
    )
else:
    if args.lora_type=='dora':
        dora=True
    lora_config = LoraConfig(
         r=args.rank,
        lora_alpha=args.rank*2,
        target_modules=json.loads(args.target_module),
        lora_dropout=0.5,
        bias="all", #‘none’, ‘all’ or ‘lora_only'
        modules_to_save=["decode_head"],
        use_dora=dora
    )
start=time()
logging.basicConfig(format='%(asctime)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S',level=logging.INFO,handlers=[LoggingHandler()])

dataset = parser.parse_args().dataset
num_epochs = parser.parse_args().epoch

if dataset=='nq':
    data_path_train='./datasets/nq-train'
    data_path_test='./datasets/nq'
elif dataset=='scifact':
    data_path_train,data_path_test='./datasets/scifact','./datasets/scifact'
elif dataset=='fiqa':
    data_path_train,data_path_test='./datasets/fiqa','./datasets/fiqa'
elif dataset=='hotpotqa':
    data_path_train,data_path_test='./datasets/hotpotqa','./datasets/hotpotqa'
else :
    data_path_train,data_path_test='./datasets/nfcorpus','./datasets/nfcorpus'

corpus, queries, qrels = GenericDataLoader(data_path_train).load(split="train")
dev_corpus, dev_queries, dev_qrels = GenericDataLoader(data_path_test).load(split="test")

if args.experiment_type:
    pool_mode=args.experiment_type
else:
    pool_mode=args.pool_mode
model_name=args.model_name 
word_embedding_model = models.LoraTransformer(model_name,lora_config=lora_config, max_seq_length=350,experiment_type=args.experiment_type,lora_type=args.lora_type)
pooling_model = models.Pooling(pooling_mode=pool_mode,word_embedding_dimension=word_embedding_model.get_word_embedding_dimension())


model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
retriever = TrainRetriever(model=model, batch_size=args.batch_size)

#### Prepare training samples
train_samples = retriever.load_train(corpus, queries, qrels)
train_dataloader = retriever.prepare_train(train_samples, shuffle=True)


train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model)
#### Prepare dev evaluato
ir_evaluator = retriever.load_ir_evaluator(dev_corpus, dev_queries, dev_qrels)

model_save_path = os.path.join(f"checkpoints/lora_{args.experiment_name}_epoch{num_epochs}", "{}_{}".format(model_name, dataset))


os.makedirs(model_save_path, exist_ok=True)

#### Configure Train params
evaluation_steps = 100000
warmup_steps = int(len(train_samples) * num_epochs / retriever.batch_size * 0.1)
optimizer=None

if args.plus==True:
    optimizer_cls = torch.optim.AdamW
    optimizer_kwargs = {'lr': 2e-5, 'eps': 1e-6, 'betas': (0.9, 0.999), 'weight_decay': 0.0}
    loraplus_lr_ratio = 20.0
    optimizer = create_loraplus_optimizer(model, optimizer_cls, optimizer_kwargs, loraplus_lr_ratio)
retriever.fit(train_objectives=[(train_dataloader, train_loss)],
                evaluator=ir_evaluator,
                epochs=num_epochs,
                output_path=model_save_path,
                warmup_steps=warmup_steps,
                evaluation_steps=evaluation_steps,
                # steps_per_epoch=10000,
                use_amp=True,
             optimizer_custom=optimizer
             )
print("Training ends")
# print(lora_config)

