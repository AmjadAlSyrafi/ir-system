"""
IR evaluation metrics: Precision@k, Recall@k, AP, MAP, nDCG@k.

All methods accept ``ranked_results`` as a list of dicts with at minimum
a ``doc_id`` key, as returned by the retrieval service models.
"""

import logging
import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class IREvaluator:
    """Computes standard IR evaluation metrics against a set of qrels.

    Args:
        qrels: Relevance judgments as ``{query_id: {doc_id: relevance_score}}``.
            Any score > 0 is considered relevant for binary metrics.
    """

    def __init__(self, qrels: Dict[str, Dict[str, int]]) -> None:
        self.qrels = qrels
        # Latest comparison DataFrame (populated by compare_models).
        self._last_comparison: Optional[pd.DataFrame] = None
        self._last_report_path: Optional[str] = None

    # ------------------------------------------------------------------
    # Individual metrics
    # ------------------------------------------------------------------

    def precision_at_k(
        self, query_id: str, ranked_results: List[Dict], k: int = 10
    ) -> float:
        """Fraction of the top-k retrieved documents that are relevant.

        Args:
            query_id: Query identifier used to look up qrels.
            ranked_results: Ranked list of ``{doc_id, ...}`` dicts.
            k: Cutoff rank.

        Returns:
            P@k in [0, 1].
        """
        relevant = self._relevant_set(query_id)
        top_k = [r["doc_id"] for r in ranked_results[:k]]
        if not top_k:
            return 0.0
        return len(set(top_k) & relevant) / k

    def recall_at_k(
        self, query_id: str, ranked_results: List[Dict], k: Optional[int] = None
    ) -> float:
        """Fraction of all relevant documents that appear in the top-k results.

        Args:
            query_id: Query identifier.
            ranked_results: Ranked result list.
            k: Cutoff rank.  If ``None``, all results are considered.

        Returns:
            Recall in [0, 1].  Returns 0.0 if there are no relevant docs.
        """
        relevant = self._relevant_set(query_id)
        if not relevant:
            return 0.0
        retrieved = [r["doc_id"] for r in (ranked_results[:k] if k else ranked_results)]
        return len(set(retrieved) & relevant) / len(relevant)

    def average_precision(self, query_id: str, ranked_results: List[Dict]) -> float:
        """Area under the precision-recall curve (AP).

        AP = sum(P@i * rel(i)) / |relevant|  for i = 1 .. |results|

        Args:
            query_id: Query identifier.
            ranked_results: Full ranked result list.

        Returns:
            AP in [0, 1].
        """
        relevant = self._relevant_set(query_id)
        if not relevant:
            return 0.0
        hits = 0
        cumulative_precision = 0.0
        for rank, item in enumerate(ranked_results, start=1):
            if item["doc_id"] in relevant:
                hits += 1
                cumulative_precision += hits / rank
        return cumulative_precision / len(relevant)

    def ndcg_at_k(
        self, query_id: str, ranked_results: List[Dict], k: int = 10
    ) -> float:
        """Normalised Discounted Cumulative Gain at rank k.

        Uses graded relevance scores from qrels (not binary).

        DCG@k  = sum(rel_i / log2(i+1))     i = 1..k
        IDCG@k = DCG of ideal (sorted) ranking
        nDCG@k = DCG@k / IDCG@k

        Args:
            query_id: Query identifier.
            ranked_results: Ranked result list.
            k: Cutoff rank.

        Returns:
            nDCG@k in [0, 1].
        """
        qrel_scores = self.qrels.get(query_id, {})
        if not qrel_scores:
            return 0.0

        top_k = ranked_results[:k]
        dcg = sum(
            qrel_scores.get(item["doc_id"], 0) / math.log2(rank + 2)
            for rank, item in enumerate(top_k)
        )

        ideal_gains = sorted(qrel_scores.values(), reverse=True)[:k]
        idcg = sum(
            gain / math.log2(rank + 2) for rank, gain in enumerate(ideal_gains)
        )

        return dcg / idcg if idcg > 0 else 0.0

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------

    def mean_average_precision(
        self, results_per_query: Dict[str, List[Dict]]
    ) -> float:
        """Mean Average Precision over a set of queries.

        Args:
            results_per_query: ``{query_id: ranked_results}``.

        Returns:
            MAP in [0, 1].
        """
        aps = [
            self.average_precision(qid, results)
            for qid, results in results_per_query.items()
        ]
        return float(np.mean(aps)) if aps else 0.0

    # ------------------------------------------------------------------
    # Full model evaluation
    # ------------------------------------------------------------------

    def evaluate_model(
        self, model_name: str, results_per_query: Dict[str, List[Dict]], k: int = 10
    ) -> Dict:
        """Compute all metrics for every query and return a summary.

        Args:
            model_name: Label for the model (used in the output dict).
            results_per_query: ``{query_id: ranked_results}``.
            k: Cutoff rank for P@k and nDCG@k.

        Returns:
            Dict with keys ``model_name``, ``MAP``, ``mean_recall``,
            ``mean_precision_at_k``, ``mean_ndcg_at_k``,
            ``per_query_results``.
        """
        per_query: Dict[str, Dict] = {}
        for query_id, results in results_per_query.items():
            per_query[query_id] = {
                "ap": self.average_precision(query_id, results),
                f"recall": self.recall_at_k(query_id, results),
                f"precision_at_{k}": self.precision_at_k(query_id, results, k),
                f"ndcg_at_{k}": self.ndcg_at_k(query_id, results, k),
            }

        map_score = float(np.mean([v["ap"] for v in per_query.values()])) if per_query else 0.0
        mean_recall = float(np.mean([v["recall"] for v in per_query.values()])) if per_query else 0.0
        mean_p = float(np.mean([v[f"precision_at_{k}"] for v in per_query.values()])) if per_query else 0.0
        mean_ndcg = float(np.mean([v[f"ndcg_at_{k}"] for v in per_query.values()])) if per_query else 0.0

        logger.info(
            "[%s] MAP=%.4f  Recall=%.4f  P@%d=%.4f  nDCG@%d=%.4f",
            model_name, map_score, mean_recall, k, mean_p, k, mean_ndcg,
        )

        return {
            "model_name": model_name,
            "MAP": round(map_score, 4),
            "mean_recall": round(mean_recall, 4),
            f"mean_precision_at_{k}": round(mean_p, 4),
            f"mean_ndcg_at_{k}": round(mean_ndcg, 4),
            "per_query_results": per_query,
        }

    # ------------------------------------------------------------------
    # Model comparison
    # ------------------------------------------------------------------

    def compare_models(
        self, model_results: Dict[str, Dict[str, List[Dict]]], k: int = 10
    ) -> pd.DataFrame:
        """Compare multiple models on all aggregate metrics.

        Args:
            model_results: ``{model_name: {query_id: ranked_results}}``.
            k: Cutoff rank.

        Returns:
            DataFrame with models as rows and metrics as columns.
        """
        rows = []
        for model_name, results_per_query in model_results.items():
            summary = self.evaluate_model(model_name, results_per_query, k)
            rows.append(
                {
                    "Model": model_name,
                    "MAP": summary["MAP"],
                    f"P@{k}": summary.get(f"mean_precision_at_{k}", 0.0),
                    f"nDCG@{k}": summary.get(f"mean_ndcg_at_{k}", 0.0),
                    "Recall": summary["mean_recall"],
                }
            )

        df = pd.DataFrame(rows).set_index("Model")
        self._last_comparison = df
        return df

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(
        self, comparison_df: pd.DataFrame, output_path: str
    ) -> None:
        """Write a Markdown evaluation report to disk.

        Args:
            comparison_df: DataFrame produced by :meth:`compare_models`.
            output_path: Destination file path (should end in ``.md``).
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        best_map_model = comparison_df["MAP"].idxmax()
        best_map = comparison_df["MAP"].max()

        lines = [
            "# IR System — Evaluation Report\n",
            "## Metric Comparison\n",
            comparison_df.to_markdown(),
            "\n\n## Analysis\n",
            f"- **Best MAP**: `{best_map_model}` ({best_map:.4f})",
        ]

        # Note which model leads on each metric.
        for col in comparison_df.columns:
            winner = comparison_df[col].idxmax()
            val = comparison_df[col].max()
            lines.append(f"- **Best {col}**: `{winner}` ({val:.4f})")

        report = "\n".join(lines) + "\n"
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(report)

        self._last_report_path = output_path
        logger.info("Evaluation report saved to '%s'.", output_path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _relevant_set(self, query_id: str) -> set:
        """Return the set of relevant doc_ids for *query_id* (score > 0)."""
        return {
            doc_id
            for doc_id, score in self.qrels.get(query_id, {}).items()
            if score > 0
        }