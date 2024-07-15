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
# from peft.tuners.dora import DoraConfig
# from peft.tuners.rosa import RosaConfig, RosaSchedulerT,
from peft import LoraConfig,LoftQConfig,get_peft_model,AdaLoraConfig,IA3Config,LoHaConfig,LoKrConfig
import torch
from lora_plus import create_loraplus_optimizer
# loftq_config = LoftQConfig(loftq_bits=4)
# if not  os.path.exists(f"./full_backward.txt"):
#     file_b=open(f"./full_backward.txt",'w')
# else:
#     file_b=open(f"./full_backward.txt",'a')
# if not  os.path.exists(f"./full_optimize.txt"):
#     file_o=open(f"./full_optimize.txt",'w')
# else:
#     file_o=open(f"./full_optimize.txt",'a')

parser = argparse.ArgumentParser(description='Argument Parser for Dataset, Experiment Name, and Epoch Number')
parser.add_argument('-d', '--dataset', type=str, help='Name of the dataset',default='nfcorpus')
parser.add_argument('-exp_name', '--experiment_name', type=str, help='Name of the experiment',default='')
parser.add_argument('-exp_type', '--experiment_type', type=str, help='type of the experiment such as single,multi,None',default='None')
parser.add_argument('-epoch', '--epoch', type=int, help='Number of epochs',default=10)
parser.add_argument('-p','--plus',type=bool,help = 'Lora plus',default=False)
parser.add_argument('-r','--rank',type=int,help='rank',default=32)
parser.add_argument('-pool','--pool_mode',type=str,help='pooling_mode',default=None)

parser.add_argument('-model','--model_name',type=str,help='hugging face model name',default="distilbert-base-uncased")
parser.add_argument('-dora','--dora',type=bool,help='dora',default=False)
parser.add_argument('-vera','--vera',type=bool,help='vera',default=False)
parser.add_argument('-batch','--batch_size',type=int,help='batch zise',default=32)
parser.add_argument('-target_module','--target_module',type=str,help='target module',default='["q_lin","out_lin","k_lin","v_lin"]')
# parser.add_argument('-pool','--pool_mode',type=str,help='pooling_mode',default=None)
args = parser.parse_args()
lora_type=None
if args.vera==True:

    lora_config =VeraConfig(
                r=args.rank,
                target_modules=json.loads(args.target_module),
                bias="all", 
                modules_to_save=["decode_head"],
            )
    lora_type='vera'
else:
    lora_config = LoraConfig(
         r=args.rank,
        lora_alpha=args.rank*2,
        target_modules=json.loads(args.target_module),
        lora_dropout=0.5,
        bias="all", #‘none’, ‘all’ or ‘lora_only'
        modules_to_save=["decode_head"],
        use_dora=args.dora
    )
    if args.dora:
        lora_type='dora'
# lora_config = AdaLoraConfig(
#     peft_type="ADALORA", target_r=16, init_r=16,lora_alpha=64, target_modules=["q_lin","v_lin","k_lin","out_lin"],
#     lora_dropout=0.01,
# )
start=time()
logging.basicConfig(format='%(asctime)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S',level=logging.INFO,handlers=[LoggingHandler()])

dataset = parser.parse_args().dataset
num_epochs = parser.parse_args().epoch

if dataset=='nq':
    data_path_train='./datasets/nq-train'
    data_path_test='./datasets/nq'
elif dataset=='test':
    data_path_train='./datasets/nq_for_exp_train/'
    data_path_test='./datasets/nq_for_exp_test/'
elif dataset=='scifact':
    data_path_train,data_path_test='./datasets/scifact','./datasets/scifact'

elif dataset=='fiqa':
    data_path_train,data_path_test='./datasets/fiqa','./datasets/fiqa'
elif dataset=='hotpotqa':
    data_path_train,data_path_test='./datasets/hotpotqa','./datasets/hotpotqa'
else :
    data_path_train='./datasets/nfcorpus'
    data_path_test='./datasets/nfcorpus'

corpus, queries, qrels = GenericDataLoader(data_path_train).load(split="train")
dev_corpus, dev_queries, dev_qrels = GenericDataLoader(data_path_test).load(split="test")

tracemalloc.start()
model_name=args.model_name #"hkunlp/instructor-xl"#distilbert-base-uncased"#"intfloat/e5-small"##"# "BAAI/bge-small-en"#v#
word_embedding_model = models.LoraTransformer(model_name,lora_config=lora_config, max_seq_length=350,experiment_type=args.experiment_type,lora_type=lora_type)
# word_embedding_model = models.PeftTransformer(model_name,peft_config=lora_config, max_seq_length=350)
pooling_model = models.Pooling(pooling_mode=args.pool_mode,word_embedding_dimension=word_embedding_model.get_word_embedding_dimension())


model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
# model = SentenceTransformer(modules=[word_embedding_model])
# snap = tracemalloc.take_snapshot()
retriever = TrainRetriever(model=model, batch_size=args.batch_size)
#### Prepare training samples
train_samples = retriever.load_train(corpus, queries, qrels)
train_dataloader = retriever.prepare_train(train_samples, shuffle=True)


train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model)
# train_loss = losses.CachedMultipleNegativesRankingLoss(model=retriever.model)
# train_loss = losses.MultipleNegativesSymmetricRankingLoss(model=retriever.model)


#### Prepare dev evaluato
ir_evaluator = retriever.load_ir_evaluator(dev_corpus, dev_queries, dev_qrels)

model_save_path = os.path.join(pathlib.Path(__file__).parent.absolute(), f"output/lora_{args.experiment_name}_epoch{num_epochs}", "{}-v1-{}".format(model_name, dataset))

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
snapshot1 = tracemalloc.take_snapshot()
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
end=time()
snapshot2 = tracemalloc.take_snapshot()
stats = snapshot2.statistics('filename')
current, peak = tracemalloc.get_traced_memory()
top_stats = snapshot2.compare_to(snapshot1, 'lineno')
# for stat in top_stats[:10]:  # Show top 10 memory consuming lines
#     print(stat)
# Output the total memory usage
print(f"Total memory usage: {current / 10**6} MB")
tracemalloc.stop()
print(f'Total Time : {end-start}')
print(lora_config)
