from sentence_transformers import SentenceTransformer, LoggingHandler
from seb import Evaluation
import logging
import os
import seb
import sys

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
    seb.tasks.STS.BIOSSES(datasets_folder='datasets'),
    seb.tasks.STS.SICKR(datasets_folder='datasets'),
    seb.tasks.STS.STSbenchmark(datasets_folder='datasets'),
    seb.tasks.Reranking.AskUbuntuDupQuestions(datasets_folder='datasets'),
    seb.tasks.Reranking.SciDocs(datasets_folder='datasets'),
    seb.tasks.Reranking.StackOverflowDupQuestions(datasets_folder='datasets'),
    seb.tasks.Retrieval.CQADupStack(datasets_folder='datasets'),
    seb.tasks.Retrieval.QuoraRetrieval(datasets_folder='datasets'),
    seb.tasks.Clustering.RedditClustering(datasets_folder='datasets'),
    seb.tasks.Clustering.StackExchangeClustering(datasets_folder='datasets'),
    seb.tasks.Clustering.TwentyNewsgroupsClustering(datasets_folder='datasets'),
]

logging.info("System to evaluate: {}".format(len(sys.argv[1:])))

for model_name in sys.argv[1:]:
    logging.info(model_name)
    try:
        model = SentenceTransformer(model_name)
        eval = Evaluation(datasets_folder='datasets')
        eval.run_all(model, tasks=tasks, split='test', output_folder=os.path.join('results', model_name.strip("/").replace("/", "-")))
    except Exception as e:
        print(e)
