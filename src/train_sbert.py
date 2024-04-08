'''
This examples show how to train a basic Bi-Encoder for any BEIR dataset without any mined hard negatives or triplets.

The queries and passages are passed independently to the transformer network to produce fixed sized embeddings.
These embeddings can then be compared using cosine-similarity to find matching passages for a given query.

For training, we use MultipleNegativesRankingLoss. There, we pass pairs in the format:
(query, positive_passage). Other positive passages within a single batch becomes negatives given the pos passage.

We do not mine hard negatives or train triplets in this example.

Running this script:
python train_sbert.py
'''

from sentence_transformers import losses, models, SentenceTransformer
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.train import TrainRetriever
import pathlib, os
import logging
from time import time
import tracemalloc
#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )
start=time()

#### Download nfcorpus.zip dataset and unzip the dataset
dataset ="nq"


#url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
#out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
#data_path = util.download_and_unzip(url, out_dir)
#exit()
data_path_train='./datasets/nq-train'
data_path_test='./datasets/nq'
data_path="./datasets/nfcorpus"
#### Provide the data_path where nfcorpus has been downloaded and unzipped
corpus, queries, qrels = GenericDataLoader(data_path_train).load(split="train")
#### Please Note not all datasets contain a dev split, comment out the line if such the case
dev_corpus, dev_queries, dev_qrels = GenericDataLoader(data_path_test).load(split="test")
tracemalloc.start()
#### Provide any sentence-transformers or HF model
model_name ="distilbert-base-uncased" 
word_embedding_model = models.Transformer(model_name, max_seq_length=350)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

#### Or provide pretrained sentence-transformer model
# model = SentenceTransformer("msmarco-distilbert-base-v3")

retriever = TrainRetriever(model=model, batch_size=32)

#### Prepare training samples
train_samples = retriever.load_train(corpus, queries, qrels)
train_dataloader = retriever.prepare_train(train_samples, shuffle=True)

#### Training SBERT with cosine-product
# train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model)
#### training SBERT with dot-product
train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model, similarity_fct=util.dot_score)

#### Prepare dev evaluator
ir_evaluator = retriever.load_ir_evaluator(dev_corpus, dev_queries, dev_qrels)

#### If no dev set is present from above use dummy evaluator
# ir_evaluator = retriever.load_dummy_evaluator()

num_epochs = 5
#### Provide model save path
model_save_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "output", "{}-v1-{}-epcho{}".format(model_name, dataset,num_epochs))
os.makedirs(model_save_path, exist_ok=True)

#### Configure Train params
evaluation_steps = 100000
warmup_steps = int(len(train_samples) * num_epochs / retriever.batch_size * 0.1)

retriever.fit(train_objectives=[(train_dataloader, train_loss)], 
                evaluator=ir_evaluator, 
                epochs=num_epochs,
                output_path=model_save_path,
                warmup_steps=warmup_steps,
                evaluation_steps=evaluation_steps,
                # steps_per_epoch=10000,
                use_amp=True)
# logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
# ndcg, _map, recall, precision = retriever.evaluate(qrels, reranked_results, retriever.k_values)
end=time()

memory_stats = tracemalloc.get_traced_memory()
print("Total memory usage:", memory_stats[0] / (1024 * 1024), "MB")
tracemalloc.stop()

print(f'total time : {end-start}')
# print(f'epoch : {num_epochs} , batch_size(')
