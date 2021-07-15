import argparse

from sentence_transformers import SentenceTransformer, models
from beir import util, LoggingHandler
from beir.retrieval import models as beir_models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

import logging
from logging import FileHandler
import pathlib, os
import random


def main(model_names, eval_datasets, sample_k=10, asymmetric_mode=False):
    print(eval_datasets)
    if eval_datasets is None:
        eval_datasets = ["nfcorpus"]

    # Setup BEIR
    for dataset in eval_datasets:
        logging.info("Evaluating BEIR dataset : " + dataset)

        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
        out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
        data_path = util.download_and_unzip(url, out_dir)

        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

        if asymmetric_mode:
            try:
                assert len(model_names) == 2
                q_model = model_names[0]
                a_model = model_names[1]

                q_word = models.Transformer(q_model)
                q_pool = models.Pooling(q_word.get_word_embedding_dimension())
                q_norm = models.Normalize()
                q_model = SentenceTransformer(modules=[q_word, q_pool, q_norm])

                a_word = models.Transformer(a_model)
                a_pool = models.Pooling(a_word.get_word_embedding_dimension())
                a_norm = models.Normalize()
                a_model = SentenceTransformer(modules=[a_word, a_pool, a_norm])

                beir_model = beir_models.SentenceBERT()
                beir_model.q_model = q_model
                beir_model.doc_model = a_model

                # Testing Models
                model = DRES(beir_model, batch_size=16)
                retriever = EvaluateRetrieval(model, score_function="cos_sim")

                results = retriever.retrieve(corpus, queries)

                logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
                ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

                mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")
                recall_cap = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="recall_cap")
                hole = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="hole")

                query_id, ranking_scores = random.choice(list(results.items()))
                scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
                logging.info("Query : %s\n" % queries[query_id])

                for rank in range(sample_k):
                    doc_id = scores_sorted[rank][0]
                    # Format: Rank x: ID [Title] Body
                    logging.info(
                        "Rank %d: %s [%s] - %s\n" % (
                            rank + 1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text")))
            except Exception as e:
                print(e)

        # Loading Models
        for model_name in model_names:
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

                query_id, ranking_scores = random.choice(list(results.items()))
                scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
                logging.info("Query : %s\n" % queries[query_id])

                for rank in range(sample_k):
                    doc_id = scores_sorted[rank][0]
                    # Format: Rank x: ID [Title] Body
                    logging.info(
                        "Rank %d: %s [%s] - %s\n" % (
                            rank + 1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text")))
            except Exception as e:
                print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--models", nargs="+", help="List of local model paths to run.", required=True)
    parser.add_argument("-t", "--tests", nargs="*", help="List of BEIR tests to run. "
                                                         "If not specified, nfcorpus is run.", required=False)
    parser.add_argument("-k", type=int, help="K samples to display.", default=10, required=False)
    parser.add_argument("--asymmetric_mode", action="store_true", default=False, help="If set: Different Question / Answer models")
    parser.add_argument("-o", "--output", help="Output logging file path.", required=False)
    args = parser.parse_args()

    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    # Log to file as well if argument is provided.
    if args.output is not None:
        logging.getLogger().addHandler(FileHandler(args.output))

    main(args.models, args.tests, args.k, args.asymmetric_mode)
