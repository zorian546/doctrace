# docs-reranker-minilm

`cross-encoder/ms-marco-MiniLM-L-6-v2` fine-tuned on the DocTrace gold questions over the FastAPI docs, using hard negatives mined from the merged semantic and BM25 candidate pools.

- Training: 290 pairs, 72 held out for validation, 3 epochs (`scripts/tune_reranker.py`)
- Effect on the 65 gold questions with source passages: MRR 0.624 to 0.739, Hit Rate@5 0.769 to 0.877
- Loaded by `doctrace/search/fusion.py`, which falls back to the stock model if the weights are missing

The weight file is stored with Git LFS.
