"""
Text preprocessing pipeline for IR system.

Supports stemming (PorterStemmer) or lemmatization (WordNetLemmatizer),
stopword removal, and parallel batch processing via multiprocessing.
"""

import logging
import re
from multiprocessing import Pool, cpu_count
from typing import Dict, List

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("punkt_tab", quiet=True)

logger = logging.getLogger(__name__)

# Module-level instance used by multiprocessing workers (must be picklable).
_worker_preprocessor: "TextPreprocessor | None" = None


def _init_worker(language: str, do_stemming: bool, do_lemmatization: bool,
                 remove_stopwords: bool, min_token_length: int) -> None:
    global _worker_preprocessor
    _worker_preprocessor = TextPreprocessor(
        language=language,
        do_stemming=do_stemming,
        do_lemmatization=do_lemmatization,
        remove_stopwords=remove_stopwords,
        min_token_length=min_token_length,
    )


def _worker_preprocess(text: str) -> List[str]:
    assert _worker_preprocessor is not None
    return _worker_preprocessor.preprocess(text)


class TextPreprocessor:
    """Configurable text preprocessing pipeline.

    Args:
        language: NLTK stopword language (e.g. ``"english"``).
        do_stemming: Apply Porter stemming. Mutually exclusive with
            ``do_lemmatization``.
        do_lemmatization: Apply WordNet lemmatization. Mutually exclusive
            with ``do_stemming``. When both are ``True``, lemmatization wins.
        remove_stopwords: Strip NLTK stopwords for the chosen language.
        min_token_length: Discard tokens shorter than this many characters.
    """

    def __init__(
        self,
        language: str = "english",
        do_stemming: bool = True,
        do_lemmatization: bool = False,
        remove_stopwords: bool = True,
        min_token_length: int = 2,
    ) -> None:
        if do_lemmatization and do_stemming:
            logger.warning(
                "Both do_stemming and do_lemmatization are True; "
                "lemmatization takes precedence."
            )
            do_stemming = False

        self.language = language
        self.do_stemming = do_stemming
        self.do_lemmatization = do_lemmatization
        self.remove_stopwords = remove_stopwords
        self.min_token_length = min_token_length

        try:
            self._stop_words = set(stopwords.words(language))
        except OSError as exc:
            raise ValueError(
                f"NLTK stopwords not available for language '{language}'."
            ) from exc

        self._stemmer = PorterStemmer() if do_stemming else None
        self._lemmatizer = WordNetLemmatizer() if do_lemmatization else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """Preprocess a single text string into a list of clean tokens.

        Pipeline:
        1. Lowercase
        2. Strip characters that are not ASCII letters or whitespace
        3. Whitespace-split tokenization
        4. Stopword removal (optional)
        5. Stemming or lemmatization (optional, mutually exclusive)
        6. Minimum token length filter

        Args:
            text: Raw input text.

        Returns:
            List of preprocessed tokens.
        """
        if not isinstance(text, str):
            logger.warning("preprocess() received non-string input (%s); coercing.", type(text))
            text = str(text)

        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        tokens: List[str] = text.split()

        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self._stop_words]

        if self._lemmatizer is not None:
            tokens = [self._lemmatizer.lemmatize(t) for t in tokens]
        elif self._stemmer is not None:
            tokens = [self._stemmer.stem(t) for t in tokens]

        tokens = [t for t in tokens if len(t) >= self.min_token_length]
        return tokens

    def preprocess_batch(
        self, texts: List[str], n_jobs: int = -1
    ) -> List[List[str]]:
        """Preprocess a list of texts in parallel using multiprocessing.

        Args:
            texts: List of raw input strings.
            n_jobs: Number of worker processes. ``-1`` uses all available CPUs.

        Returns:
            List of token lists, one per input text.
        """
        if not texts:
            return []

        workers = cpu_count() if n_jobs == -1 else max(1, n_jobs)
        # For small batches, spawning processes costs more than it saves.
        if len(texts) < workers * 4:
            return [self.preprocess(t) for t in texts]

        logger.info(
            "Batch preprocessing %d texts with %d workers.", len(texts), workers
        )
        with Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(
                self.language,
                self.do_stemming,
                self.do_lemmatization,
                self.remove_stopwords,
                self.min_token_length,
            ),
        ) as pool:
            results = pool.map(_worker_preprocess, texts)
        return results

    def preprocess_to_string(self, text: str) -> str:
        """Preprocess text and return tokens joined into a single string.

        Args:
            text: Raw input text.

        Returns:
            Space-joined preprocessed tokens.
        """
        return " ".join(self.preprocess(text))

    def preprocess_document(self, doc: Dict[str, str]) -> Dict[str, object]:
        """Preprocess a document dict, adding ``tokens`` and ``processed_text``.

        Args:
            doc: Dict with at least ``doc_id`` and ``text`` keys.

        Returns:
            Original dict enriched with:
            - ``tokens``: list of preprocessed tokens
            - ``processed_text``: space-joined token string
        """
        tokens = self.preprocess(doc["text"])
        return {
            **doc,
            "tokens": tokens,
            "processed_text": " ".join(tokens),
        }

    def preprocess_query(self, query: str) -> Dict[str, object]:
        """Preprocess a query string.

        Args:
            query: Raw query text.

        Returns:
            Dict with keys:
            - ``original``: unchanged input
            - ``tokens``: list of preprocessed tokens
            - ``processed_text``: space-joined token string
        """
        tokens = self.preprocess(query)
        return {
            "original": query,
            "tokens": tokens,
            "processed_text": " ".join(tokens),
        }
