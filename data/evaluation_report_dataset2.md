# IR System — Evaluation Report

## Metric Comparison

| Model     |    MAP |   P@10 |   nDCG@10 |   Recall |
|:----------|-------:|-------:|----------:|---------:|
| tfidf     | 0.2726 |  0.075 |    0.3645 |    0.375 |
| bm25      | 0.3137 |  0.084 |    0.4153 |    0.42  |
| embedding | 0.2119 |  0.06  |    0.2854 |    0.3   |
| hybrid    | 0.3088 |  0.085 |    0.4113 |    0.425 |


## Analysis

- **Best MAP**: `bm25` (0.3137)
- **Best MAP**: `bm25` (0.3137)
- **Best P@10**: `hybrid` (0.0850)
- **Best nDCG@10**: `bm25` (0.4153)
- **Best Recall**: `hybrid` (0.4250)
