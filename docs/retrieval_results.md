Retrieval evaluation on 20 synthetic questions (269 chunks, embeddings: `intfloat/multilingual-e5-small`)

| Retriever | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|
| BM25 (lexical) | 0.50 | 0.70 | 0.80 | 0.644 |
| **Dense (multilingual-e5)** | 0.65 | 0.85 | 0.95 | 0.763 |
| Hybrid (RRF) | 0.60 | 0.85 | 0.95 | 0.749 |
