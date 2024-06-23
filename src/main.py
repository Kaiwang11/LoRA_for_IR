from sentence_transformers import losses, models, SentenceTransformer
from beir import util, LoggingHandler
from beir.losses import BPRLoss,MarginMSELoss
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.train import TrainRetriever
import pathlib, os

import logging
from time import time
import tracemalloc
import argparse
# from peft.tuners.dora import DoraConfig
from peft import LoraConfig,VeraConfig
import torch
from lora_plus import create_loraplus_optimizer
# loftq_config = LoftQConfig(loftq_bits=4)
parser = argparse.ArgumentParser(description='Argument Parser for Dataset, Experiment Name, and Epoch Number')
parser.add_argument('-d', '--dataset', type=str, help='Name of the dataset',default='nfcorpus')
parser.add_argument('-exp', '--experiment', type=str, help='Name of the experiment',default='')
parser.add_argument('-epoch', '--epoch', type=int, help='Number of epochs',default=10)
parser.add_argument('-p','--plus',type=bool,help = 'Lora plus',default=False)
parser.add_argument('-r','--rank',type=int,help='rank',default=32)
parser.add_argument('-pool','--pool_mode',type=str,help='pooling_mode',default=None)
parser.add_argument('-dora','--dora',type=bool,help='execute dora,default False',default=False )

parser.add_argument('-vera','--vera',type=bool,help='execute vera,default False',default=False )
args = parser.parse_args()
# locals().update(vars(args))6  

if args.vera==True:

    lora_config =VeraConfig(
        r=args.rank,
        target_modules=["q_lin","v_lin","k_lin","out_lin"],#['query','value','key'],#["q_lin","v_lin","k_lin","out_lin"],#,###,
        bias="all", #‘none’, ‘all’ or ‘lora_only'
        modules_to_save=["decode_head"],
    )
else:
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank*2,
        target_modules=["q_lin","v_lin","k_lin","out_lin"],#,
        lora_dropout=0.5,
        bias="all", #‘none’, ‘all’ or ‘lora_only'
        modules_to_save=["decode_head"],
        # loftq_config=loftq_config
        use_dora=args.dora
    )

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
else :
    data_path_train='./datasets/nfcorpus'
    data_path_test='./datasets/nfcorpus'

exp=parser.parse_args().experiment
corpus, queries, qrels = GenericDataLoader(data_path_train).load(split="train")
dev_corpus, dev_queries, dev_qrels = GenericDataLoader(data_path_test).load(split="test")

tracemalloc.start()
model_name ="distilbert-base-uncased"

word_embedding_model = models.LoraTransformer(model_name,lora_config=lora_config, max_seq_length=350,)
pooling_model = models.Pooling(pooling_mode=args.pool_mode,word_embedding_dimension=word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
# snap = tracemalloc.take_snapshot()
retriever = TrainRetriever(model=model, batch_size=32)
#### Prepare training samples
train_samples = retriever.load_train(corpus, queries, qrels)
train_dataloader = retriever.prepare_train(train_samples, shuffle=True)


train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model)
# train_loss = losses.CachedMultipleNegativesRankingLoss(model=retriever.model)
# train_loss = losses.MultipleNegativesSymmetricRankingLoss(model=retriever.model)


#### Prepare dev evaluato
ir_evaluator = retriever.load_ir_evaluator(dev_corpus, dev_queries, dev_qrels)

model_save_path = os.path.join(pathlib.Path(__file__).parent.absolute(), f"output/lora_{exp}_epoch{num_epochs}", "{}-v1-{}".format(model_name, dataset))

os.makedirs(model_save_path, exist_ok=True)

#### Configure Train params
evaluation_steps = 100000
warmup_steps = int(len(train_samples) * num_epochs / retriever.batch_size * 0.1)
optimizer=None

if parser.parse_args().plus==True:
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
end=time()
current, peak = tracemalloc.get_traced_memory()
# Output the total memory usage
print(f"Total memory usage: {current / 10**6} MB")
tracemalloc.stop()
print(f'Total Time : {end-start}')
print(lora_config)
