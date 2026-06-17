"""
Dataset loader for IR system using ir-datasets library.

Supports loading documents, queries, and relevance judgments (qrels)
from standard IR benchmark datasets.
"""

import json
import logging
import os
from typing import Dict, Generator, Iterator

import ir_datasets

logger = logging.getLogger(__name__)

DATASET_ALIASES: Dict[str, str] = {
    "dataset1": "msmarco-passage/dev/small",
    "dataset2": "beir/nq/train",
}


class DatasetLoader:
    """Loads IR datasets via ir-datasets and exposes documents, queries, and qrels."""

    def __init__(self) -> None:
        self._cache: Dict[str, ir_datasets.Dataset] = {}

    def _get_dataset(self, dataset_name: str) -> ir_datasets.Dataset:
        """Return a cached dataset handle, resolving aliases if needed.

        Args:
            dataset_name: An ir-datasets dataset ID or a known alias.

        Returns:
            The ir-datasets dataset object.

        Raises:
            ValueError: If the dataset ID is unknown or unavailable.
        """
        name = DATASET_ALIASES.get(dataset_name, dataset_name)
        if name not in self._cache:
            logger.info("Loading dataset: %s", name)
            try:
                dataset = ir_datasets.load(name)
            except KeyError as exc:
                raise ValueError(
                    f"Dataset '{name}' not found in ir-datasets. "
                    "Check the dataset ID at https://ir-datasets.com/"
                ) from exc
            except Exception as exc:
                raise ValueError(
                    f"Failed to load dataset '{name}': {exc}"
                ) from exc
            self._cache[name] = dataset
        return self._cache[name]

    def load_documents(
        self, dataset_name: str
    ) -> Generator[Dict[str, str], None, None]:
        """Yield documents from the dataset.

        Args:
            dataset_name: Dataset ID or alias.

        Yields:
            Dicts with keys ``doc_id`` and ``text``.

        Raises:
            ValueError: If the dataset has no document collection.
        """
        dataset = self._get_dataset(dataset_name)
        if not dataset.has_docs():
            raise ValueError(
                f"Dataset '{dataset_name}' does not have a document collection."
            )
        logger.info("Streaming documents from '%s'", dataset_name)
        for doc in dataset.docs_iter():
            yield {"doc_id": doc.doc_id, "text": doc.text}

    def load_queries(
        self, dataset_name: str
    ) -> Generator[Dict[str, str], None, None]:
        """Yield queries from the dataset.

        Args:
            dataset_name: Dataset ID or alias.

        Yields:
            Dicts with keys ``query_id`` and ``text``.

        Raises:
            ValueError: If the dataset has no queries.
        """
        dataset = self._get_dataset(dataset_name)
        if not dataset.has_queries():
            raise ValueError(
                f"Dataset '{dataset_name}' does not have queries."
            )
        logger.info("Streaming queries from '%s'", dataset_name)
        for query in dataset.queries_iter():
            yield {"query_id": query.query_id, "text": query.text}

    def load_qrels(self, dataset_name: str) -> Dict[str, Dict[str, int]]:
        """Load all relevance judgments into memory.

        Args:
            dataset_name: Dataset ID or alias.

        Returns:
            Nested dict ``{query_id: {doc_id: relevance_score}}``.

        Raises:
            ValueError: If the dataset has no qrels.
        """
        dataset = self._get_dataset(dataset_name)
        if not dataset.has_qrels():
            raise ValueError(
                f"Dataset '{dataset_name}' does not have qrels."
            )
        logger.info("Loading qrels from '%s'", dataset_name)
        qrels: Dict[str, Dict[str, int]] = {}
        for qrel in dataset.qrels_iter():
            qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = qrel.relevance
        return qrels

    def get_dataset_stats(self, dataset_name: str) -> Dict[str, int]:
        """Return document, query, and qrel counts for a dataset.

        Counts are computed by iterating the collections; for large datasets
        this may take time. Missing collections are reported as -1.

        Args:
            dataset_name: Dataset ID or alias.

        Returns:
            Dict with keys ``num_docs``, ``num_queries``, ``num_qrels``.
        """
        dataset = self._get_dataset(dataset_name)
        stats: Dict[str, int] = {}

        if dataset.has_docs():
            logger.info("Counting documents in '%s'…", dataset_name)
            try:
                stats["num_docs"] = dataset.docs_count() or sum(
                    1 for _ in dataset.docs_iter()
                )
            except Exception as exc:
                logger.warning("Could not count docs: %s", exc)
                stats["num_docs"] = -1
        else:
            stats["num_docs"] = -1

        if dataset.has_queries():
            logger.info("Counting queries in '%s'…", dataset_name)
            try:
                stats["num_queries"] = sum(1 for _ in dataset.queries_iter())
            except Exception as exc:
                logger.warning("Could not count queries: %s", exc)
                stats["num_queries"] = -1
        else:
            stats["num_queries"] = -1

        if dataset.has_qrels():
            logger.info("Counting qrels in '%s'…", dataset_name)
            try:
                stats["num_qrels"] = sum(1 for _ in dataset.qrels_iter())
            except Exception as exc:
                logger.warning("Could not count qrels: %s", exc)
                stats["num_qrels"] = -1
        else:
            stats["num_qrels"] = -1

        return stats

    def save_sample(
        self, dataset_name: str, n: int, output_path: str
    ) -> None:
        """Save the first *n* documents from a dataset to a JSON file.

        The output file contains a JSON array of ``{doc_id, text}`` objects.
        Parent directories are created automatically.

        Args:
            dataset_name: Dataset ID or alias.
            n: Number of documents to include in the sample.
            output_path: Destination file path (should end in ``.json``).

        Raises:
            ValueError: If the dataset has no documents.
            OSError: If the output file cannot be written.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        sample = []
        for i, doc in enumerate(self.load_documents(dataset_name)):
            if i >= n:
                break
            sample.append(doc)

        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(sample, fh, ensure_ascii=False, indent=2)

        logger.info(
            "Saved %d documents from '%s' to '%s'",
            len(sample),
            dataset_name,
            output_path,
        )


def main() -> None:
    """Load both benchmark datasets, print stats, and save document samples."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    loader = DatasetLoader()

    targets = [
        ("msmarco-passage", "data/datasets/dataset1/sample.json"),
        ("beir/nq/train", "data/datasets/dataset2/sample.json"),
    ]

    for dataset_name, sample_path in targets:
        print(f"\n{'=' * 60}")
        print(f"Dataset: {dataset_name}")
        print("=" * 60)

        try:
            stats = loader.get_dataset_stats(dataset_name)
            print(f"  Documents : {stats['num_docs']:>12,}")
            print(f"  Queries   : {stats['num_queries']:>12,}")
            print(f"  Qrels     : {stats['num_qrels']:>12,}")
        except ValueError as exc:
            logger.error("Could not retrieve stats for '%s': %s", dataset_name, exc)
            continue

        try:
            loader.save_sample(dataset_name, n=1000, output_path=sample_path)
            print(f"  Sample saved to: {sample_path}")
        except (ValueError, OSError) as exc:
            logger.error(
                "Could not save sample for '%s': %s", dataset_name, exc
            )


if __name__ == "__main__":
    main()
