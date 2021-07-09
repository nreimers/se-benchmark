from sentence_transformers import SentenceTransformer, LoggingHandler, models
from seb import Evaluation
import logging
import os
import seb
from transformers import AutoTokenizer, AutoModel
import sys
import types
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn

#Use TPU
import torch_xla.core.xla_model as xm
dev = xm.xla_device()

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout



tasks = [
    seb.tasks.BinaryClassification.SprintDuplicateQuestions(datasets_folder='datasets'),
    seb.tasks.BinaryClassification.TwitterSemEval2015(datasets_folder='datasets'),
    seb.tasks.BinaryClassification.TwitterURLCorpus(datasets_folder='datasets'),
    seb.tasks.STS.STSbenchmark(datasets_folder='datasets'),
    seb.tasks.STS.SICKR(datasets_folder='datasets'),
    seb.tasks.STS.BIOSSES(datasets_folder='datasets'),
    seb.tasks.Reranking.AskUbuntuDupQuestions(datasets_folder='datasets'),
    seb.tasks.Reranking.StackOverflowDupQuestions(datasets_folder='datasets'),
    seb.tasks.Reranking.SciDocs(datasets_folder='datasets'),
    seb.tasks.Retrieval.QuoraRetrieval(datasets_folder='datasets'),
    seb.tasks.Retrieval.CQADupStack(datasets_folder='datasets'),
    seb.tasks.Clustering.TwentyNewsgroupsClustering(datasets_folder='datasets'),
    seb.tasks.Clustering.StackExchangeClustering(datasets_folder='datasets'),
    seb.tasks.Clustering.RedditClustering(datasets_folder='datasets'),
]

logging.info("System to evaluate: {}".format(len(sys.argv[1:])))

#Use TPU
import torch_xla.core.xla_model as xm
dev = xm.xla_device()

class AutoModelForSentenceEmbedding(nn.Module):
    def __init__(self, model_name, device, normalize=True):
        super(AutoModelForSentenceEmbedding, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.normalize = normalize
        self.device = device
        self.max_seq_length = 128
        self.model.to(device)

    def forward(self, **kwargs):
        model_output = self.model(**kwargs)
        embeddings = self.mean_pooling(model_output, kwargs['attention_mask'])
        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings
    
    def encode(self, sentences, batch_size=32, convert_to_tensor=False, **kwargs):
        all_emb = []

        for start_idx in range(0, len(sentences), batch_size):
            emb = self._encode_batch(sentences[start_idx:start_idx + batch_size])
            all_emb.append(emb)

        all_emb = np.concatenate(all_emb, axis=0)

        if convert_to_tensor:
            all_emb = torch.tensor(all_emb)

        return all_emb

    def _encode_batch(self, sentences):
        encoded_input = self.tokenizer(sentences, padding='max_length', truncation='longest_first', return_tensors="pt", max_length=self.max_seq_length)

        with torch.no_grad():
            sentence_embeddings = self.forward(**encoded_input.to(self.device).to(self.device))

        return sentence_embeddings.cpu().numpy()

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def save_pretrained(self, path):
        self.model.save_pretrained(path)

model_name = sys.argv[1]
logging.info(model_name)

model = AutoModelForSentenceEmbedding(model_name, device=dev)
#model.load_state_dict(torch.load(os.path.join(sys.argv[1], "xm_save.bin")))
model.eval()

eval = Evaluation(datasets_folder='datasets')
eval.run_all(model, tasks=tasks, split='test', output_folder=os.path.join('results', model_name.strip("/").replace("/", "-")))

