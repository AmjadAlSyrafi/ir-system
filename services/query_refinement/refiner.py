"""
Query refinement: spell correction, synonym expansion, and history-based boosting.
"""

import json
import logging
import os
from typing import Dict, List, Optional

import nltk
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker

for _r in ("wordnet", "stopwords", "averaged_perceptron_tagger",
           "averaged_perceptron_tagger_eng", "punkt", "punkt_tab", "omw-1.4"):
    nltk.download(_r, quiet=True)

logger = logging.getLogger(__name__)

_STOP_WORDS = set(stopwords.words("english"))
# WordNet POS tags that map to nouns and verbs (worth expanding)
_EXPAND_POS = {"NN", "NNS", "NNP", "NNPS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}


class QueryRefiner:
    """Applies spell correction, synonym expansion, and history boosting to queries.

    Args:
        use_spellcheck: Apply pyspellchecker spell correction.
        use_synonyms: Expand content words with WordNet synonyms.
        use_history: Boost query with terms from similar past queries.
        history_file: JSON file path for persisting per-user query history.
    """

    def __init__(
        self,
        use_spellcheck: bool = True,
        use_synonyms: bool = True,
        use_history: bool = True,
        history_file: str = "query_history.json",
    ) -> None:
        self.use_spellcheck = use_spellcheck
        self.use_synonyms = use_synonyms
        self.use_history = use_history
        self.history_file = history_file

        self._spell: Optional[SpellChecker] = SpellChecker() if use_spellcheck else None

    # ------------------------------------------------------------------
    # Public pipeline
    # ------------------------------------------------------------------

    def refine(self, query: str, user_id: str = "default") -> Dict:
        """Apply all enabled refinements in order and return a trace dict.

        Order: spell_correct → expand_with_synonyms → refine_with_history.

        Args:
            query: Raw query string from the user.
            user_id: Identifier used to scope query history.

        Returns:
            Dict with keys:
            - ``original_query``
            - ``corrected_query``
            - ``expanded_query``
            - ``final_query``
            - ``changes_made``: list of human-readable change descriptions
        """
        original = query
        changes: List[str] = []

        # Step 1 — spell correction
        corrected = self.spell_correct(query) if self.use_spellcheck else query
        if corrected != query:
            changes.append(f"Spell-corrected: '{query}' → '{corrected}'")
        query = corrected

        # Step 2 — synonym expansion
        expanded = self.expand_with_synonyms(query) if self.use_synonyms else query
        if expanded != query:
            extra = set(expanded.split()) - set(query.split())
            changes.append(f"Synonyms added: {sorted(extra)}")
        query = expanded

        # Step 3 — history boosting
        boosted = self.refine_with_history(query, user_id) if self.use_history else query
        if boosted != query:
            extra = set(boosted.split()) - set(query.split())
            changes.append(f"History boost terms added: {sorted(extra)}")
        query = boosted

        # Persist original query to history after refinement.
        if self.use_history:
            self.save_to_history(original, user_id)

        return {
            "original_query": original,
            "corrected_query": corrected,
            "expanded_query": expanded,
            "final_query": query,
            "changes_made": changes,
        }

    # ------------------------------------------------------------------
    # Spell correction
    # ------------------------------------------------------------------

    def spell_correct(self, query: str) -> str:
        """Correct obvious spelling mistakes using pyspellchecker.

        Unknown words that look like proper nouns (capitalised) or short
        abbreviations (≤2 chars) are left unchanged.

        Args:
            query: Raw query string.

        Returns:
            Query with misspelled words replaced by their most likely correction.
        """
        if self._spell is None:
            return query

        tokens = query.split()
        corrected_tokens = []
        for token in tokens:
            # Preserve capitalised words and very short tokens.
            if token[0].isupper() or len(token) <= 2:
                corrected_tokens.append(token)
                continue
            lower = token.lower()
            correction = self._spell.correction(lower)
            if correction and correction != lower:
                logger.debug("Spell: '%s' → '%s'", lower, correction)
                corrected_tokens.append(correction)
            else:
                corrected_tokens.append(token)
        return " ".join(corrected_tokens)

    # ------------------------------------------------------------------
    # Synonym expansion
    # ------------------------------------------------------------------

    def expand_with_synonyms(self, query: str, max_synonyms_per_term: int = 2) -> str:
        """Add top synonyms for content words (nouns and verbs) via WordNet.

        POS tagging is used so that only meaningful word classes are
        expanded; stopwords and non-content words are skipped entirely.

        Args:
            query: Query string (ideally already spell-corrected).
            max_synonyms_per_term: Maximum synonyms to add per token.

        Returns:
            Original query string with synonym terms appended.
        """
        try:
            tokens = word_tokenize(query)
            tagged = nltk.pos_tag(tokens)
        except Exception as exc:
            logger.warning("POS tagging failed (%s); skipping synonym expansion.", exc)
            return query

        extra_terms: List[str] = []
        for word, pos in tagged:
            lower = word.lower()
            if lower in _STOP_WORDS or pos not in _EXPAND_POS or len(lower) <= 2:
                continue

            # Map Penn Treebank tag to WordNet POS.
            wn_pos = wordnet.VERB if pos.startswith("V") else wordnet.NOUN
            synonyms: List[str] = []
            for synset in wordnet.synsets(lower, pos=wn_pos):
                for lemma in synset.lemmas():
                    syn = lemma.name().replace("_", " ").lower()
                    if syn != lower and syn not in _STOP_WORDS and syn not in synonyms:
                        synonyms.append(syn)
                if len(synonyms) >= max_synonyms_per_term:
                    break

            extra_terms.extend(synonyms[:max_synonyms_per_term])

        if not extra_terms:
            return query

        logger.debug("Synonym expansion added: %s", extra_terms)
        return query + " " + " ".join(extra_terms)

    # ------------------------------------------------------------------
    # History-based boosting
    # ------------------------------------------------------------------

    def refine_with_history(self, query: str, user_id: str = "default") -> str:
        """Boost query with terms from the most similar historical query.

        Similarity is measured by Jaccard overlap on lowercased tokens.
        If the best past query scores above 0.5, its unique terms are
        appended to the current query.

        Args:
            query: Current query (spell-corrected and/or expanded).
            user_id: History scope key.

        Returns:
            Boosted query string, or original if no good match is found.
        """
        history = self._load_history().get(user_id, [])
        if not history:
            return query

        query_tokens = set(query.lower().split()) - _STOP_WORDS
        best_score = 0.0
        best_past: Optional[str] = None

        for past in history[-50:]:  # Only look at the 50 most recent queries.
            past_tokens = set(past.lower().split()) - _STOP_WORDS
            union = query_tokens | past_tokens
            if not union:
                continue
            score = len(query_tokens & past_tokens) / len(union)
            if score > best_score:
                best_score = score
                best_past = past

        if best_score > 0.5 and best_past:
            past_tokens = set(best_past.lower().split()) - _STOP_WORDS
            boost_terms = past_tokens - query_tokens
            if boost_terms:
                logger.debug(
                    "History boost (score=%.2f) adding: %s", best_score, boost_terms
                )
                return query + " " + " ".join(sorted(boost_terms))

        return query

    def save_to_history(self, query: str, user_id: str = "default") -> None:
        """Append a query to the user's history file.

        Args:
            query: The original (unrefined) query to record.
            user_id: History scope key.
        """
        history = self._load_history()
        history.setdefault(user_id, []).append(query)
        self._save_history(history)

    # ------------------------------------------------------------------
    # History I/O
    # ------------------------------------------------------------------

    def _load_history(self) -> Dict[str, List[str]]:
        if not os.path.exists(self.history_file):
            return {}
        try:
            with open(self.history_file, encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load history file '%s': %s", self.history_file, exc)
            return {}

    def _save_history(self, history: Dict[str, List[str]]) -> None:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.history_file)), exist_ok=True)
            with open(self.history_file, "w", encoding="utf-8") as fh:
                json.dump(history, fh, ensure_ascii=False, indent=2)
        except OSError as exc:
            logger.warning("Could not save history file '%s': %s", self.history_file, exc)

    def get_history(self, user_id: str = "default") -> List[str]:
        """Return a user's query history list.

        Args:
            user_id: History scope key.

        Returns:
            List of past query strings, oldest first.
        """
        return self._load_history().get(user_id, [])

    def clear_history(self, user_id: str = "default") -> None:
        """Delete all history for a user.

        Args:
            user_id: History scope key.
        """
        history = self._load_history()
        history.pop(user_id, None)
        self._save_history(history)