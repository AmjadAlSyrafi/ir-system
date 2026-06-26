# IR System — Evaluation Report

## Metric Comparison

| Model     |    MAP |   P@10 |   nDCG@10 |   Recall |
|:----------|-------:|-------:|----------:|---------:|
| tfidf     | 0.7176 |  0.116 |    0.7765 |   0.7496 |
| bm25      | 0.7151 |  0.117 |    0.7748 |   0.7498 |
| embedding | 0.7348 |  0.118 |    0.7911 |   0.7472 |
| hybrid    | 0.7275 |  0.119 |    0.785  |   0.7474 |


## Analysis

- **Best MAP**: `embedding` (0.7348)
- **Best MAP**: `embedding` (0.7348)
- **Best P@10**: `hybrid` (0.1190)
- **Best nDCG@10**: `embedding` (0.7911)
- **Best Recall**: `bm25` (0.7498)
