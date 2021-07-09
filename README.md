# Sentence Embeddings Benchmark

This code can be used to benchmark sentence embedding models.

## Requirements
```
pip install sentence-transformers
```

## Run CPU & GPU
A model trained with the [se-pytorch-xla](https://github.com/nreimers/se-pytorch-xla) can be benchmarked like this:
```
python run_seb_pytorch_automodel.py /path/to/model
```

This loads the transformer model and adds mean pooling and a normalization layer on-top. If you trained with another setup, you have to update the code.

This code will run either on CPU or on GPU. Not on TPU.

When you have a full [sentence-transformers](https://www.sbert.net) model including pooling layer etc., you can run:
```
python run_seb_sbert.py /path/to/model
```

## Run TPU

When you are on a TPU and have pytorch_xla installed, you can run:
```
python run_seb_pytorch_xla.py /path/to/model
```

Note: Embeddings will just be computed on a single core.

## Summary
Results are written for the models to the results folder. To create a tsv overview, you can run:
```
python create_summary.py results/model1 results/model2 results/model3...
```
