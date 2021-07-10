from sentence_transformers import SentenceTransformer, LoggingHandler, models
from beir import util, LoggingHandler
from beir.retrieval import models as beir_models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

import logging
import pathlib, os
import random
import sys

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# TODO : Support multiple datasets
# Setup BEIR
dataset = "nfcorpus"

url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

# Loading Models
logging.info("System to evaluate: {}".format(len(sys.argv[1:])))

for model_name in sys.argv[1:]:
    logging.info(model_name)
    try:
        word = models.Transformer(model_name)
        pool = models.Pooling(word.get_word_embedding_dimension())
        norm = models.Normalize()

        model = SentenceTransformer(modules=[word, pool, norm])

        beir_model = beir_models.SentenceBERT()
        beir_model.q_model = model
        beir_model.doc_model = model

        # Testing Models
        model = DRES(beir_model, batch_size=16)
        retriever = EvaluateRetrieval(model, score_function="cos_sim")

        results = retriever.retrieve(corpus, queries)

        logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
        ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

        mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")
        recall_cap = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="recall_cap")
        hole = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="hole")

        # TODO : Parameterize
        top_k = 10

        query_id, ranking_scores = random.choice(list(results.items()))
        scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
        logging.info("Query : %s\n" % queries[query_id])

        for rank in range(top_k):
            doc_id = scores_sorted[rank][0]
            # Format: Rank x: ID [Title] Body
            logging.info(
                "Rank %d: %s [%s] - %s\n" % (rank + 1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text")))
    except Exception as e:
        print(e)