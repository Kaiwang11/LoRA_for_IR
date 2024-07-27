from sentence_transformers import SentenceTransformer, models, losses
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.train import TrainRetriever
import pathlib, os, gzip
import logging
import json
import argparse

parser=argparse.ArgumentParser(description='Download datase . I experimnt nq,nfcorpus,fiqa,scifact .' )
parser.add_argument('-d','--dataset', type=str,help='dataset name',default='fiqa')
dataset = parser.parse_args().dataset
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "./datasets")
util.download_and_unzip(url, out_dir)



