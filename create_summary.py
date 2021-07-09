import glob
import json
import sys
import os
import numpy as np

tasks = ['20NewsgroupsClustering', 'AskUbuntuDupQuestions', 'BIOSSES', 'CQADupStack', 'QuoraRetrieval', 'RedditClustering', 'SICK-R', 'STSbenchmark', 'SciDocs', 'SprintDuplicateQuestions', 'StackExchangeClustering', 'StackOverflowDupQuestions', 'TwitterSemEval2015', 'TwitterURLCorpus']

def create_summary(folder):
    summary = {}

    for filename in glob.glob(os.path.join(folder, '*.json')):
        task_name = os.path.basename(filename).rsplit(".", maxsplit=1)[0]

        with open(filename, encoding='utf8') as fIn:
            data = json.load(fIn)

            if 'v_measure' in data:
                summary[task_name] = data['v_measure']
            elif 'map' in data:
                summary[task_name] = data['map']
            elif 'cosine_spearman' in data:
                summary[task_name] = max(data['cosine_spearman'], data['euclidean_spearman'], data['manhatten_spearman'])
            elif 'dot_score' in data:
                summary[task_name] = max(data['cos_sim']['map@k']['100'], data['dot_score']['map@k']['100'])
            elif 'max' in data and 'ap' in data['max']:
                summary[task_name] = data['max']['ap']
            elif 'map@100' in data: #CQADupStack:
                summary[task_name] = data['map@100']


    


    out_names = []
    out_data = []
    out_values = []
    for name in tasks:
        out_names.append(name)
        out_values.append(summary.get(name, -0.01)*100)
        out_data.append("{:.2f}".format(summary.get(name, -0.01)*100))

    data_out = []
    if len(out_data) == 14:
        data_out.append(os.path.basename(folder.rstrip("/"))
            .replace("home-ukp-reimers-sbert-sentence-transformers-examples-training-paraphrases-output-training_data_benchmark-", "")
            .replace("distilroberta-base-norm-", "")
            .replace("nreimers-MiniLM-L6-H384-uncased-norm-", "")
            .replace("loss_scale_parameter-", "")
            .replace("loss_scale_parameter_sym-", "")
            .replace("loss_sym-", "")
            .replace("tpu-models-", ""))
            
        data_out.append("{:.2f}".format(np.mean(out_values)))
        data_out.append("\t".join(out_data))
        
    return data_out
        
rows = []
for folder in sys.argv[1:]:
    data_out = create_summary(folder)
    if data_out is not None and len(data_out) > 0:
        rows.append(data_out)


print("\t".join(["Model", "Avg"]+tasks))
for row in sorted(rows, key=lambda x: x[1], reverse=True):
    print("\t".join(row))