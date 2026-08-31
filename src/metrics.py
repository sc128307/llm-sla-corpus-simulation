# ==========================================================
# src/metrics.py — Linguistic feature extraction (Biber 1988 framework)
#
# Promoted from notebooks/linguistic_lib.py (Phase 0, docs/ENGINEERING_PLAN.md).
# Changes vs. the old notebook module:
#   1. Lazy spaCy loading  — `import src.metrics` works even without spaCy;
#      the model is loaded on first use and can be overridden with the
#      SPACY_MODEL env var (default: en_core_web_sm).
#   2. Deterministic grammar-error metric — the value is ALWAYS produced by a
#      named engine (LanguageTool HTTP server, or the built-in rule-based
#      fallback) and never silently 0. The active engine + version are exposed
#      on the auditor so the paper's method section can record them exactly.
#   3. CEFR vocab resolves to the canonical data/processed/cefr_full_vocab.csv
#      by default (the old module silently returned AVD = 0 when missing).
#
# Backward-compatible surface (kept identical for notebooks/pipeline_new.ipynb):
#   DataPreprocessor(), LinguisticAuditor(vocab_dir=..., server_url=...),
#   DISCOURSE_MARKERS, NOMINALIZATION_SUFFIXES, VOCAB_FILENAME,
#   analyze_sample() output keys.
# ==========================================================
import os
import re
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("metrics")
logger.addHandler(logging.NullHandler())

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
VOCAB_FILENAME = "cefr_full_vocab.csv"
SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")

# Where the canonical CEFR lexicon lives (project-root relative).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VOCAB_PATH = PROJECT_ROOT / "data" / "processed" / VOCAB_FILENAME

# Biber (1988) & Halliday (1976) lists (unchanged from linguistic_lib.py)
DISCOURSE_MARKERS = {
    "and", "also", "moreover", "furthermore", "besides", "additionally",
    "in addition", "but", "however", "nevertheless", "nonetheless", "yet",
    "on the other hand", "although", "though", "because", "since",
    "therefore", "thus", "hence", "consequently", "as a result", "so",
    "when", "while", "then", "after", "before", "meanwhile",
    "subsequently", "first", "firstly", "second", "secondly", "finally",
    "lastly", "next", "specifically",
}

NOMINALIZATION_SUFFIXES = (
    "tion", "ment", "ness", "ity", "ance", "ence", "ism", "ship", "dom",
    "sis", "ure",
)

# ------------------------------------------------------------------
# Grammar-error engines
# ------------------------------------------------------------------
# Engine identifiers recorded for the method section.
# Priority (auto): roberta-cola (fine-tuned on CoLA, models/grammar_cola)
#   > languagetool-http-server (server_url) > rule-based-fallback.
# Theory: Lau et al. 2017 (acceptability is a probabilistic continuum);
# Warstadt et al. 2019 (CoLA). See docs/METHODOLOGY_JUSTIFICATION.md §5.1.
GRAMMAR_ENGINE_ROBERTA_COLA = "roberta-cola"
GRAMMAR_ENGINE_LANGUAGETOOL = "languagetool-http-server"
GRAMMAR_ENGINE_RULE_BASED = "rule-based-fallback"
GRAMMAR_ENGINE_AUTO = "auto"

# Fine-tuned acceptability model (trained by scripts/train_grammar_cola.py).
GRAMMAR_MODEL_DIR = PROJECT_ROOT / "models" / "grammar_cola"

# Version of the built-in rule set (fallback only).
RULE_BASED_ENGINE_VERSION = "1.0"

# Deterministic, high-precision rules targeting classic L2 learner errors.
# Each is a (regex, description) pair. This is a conservative proxy metric:
# it measures a *fixed subset* of observable error types, so values are
# comparable across corpora even though they undercount true errors.
RULE_BASED_ERROR_PATTERNS: List[Tuple[str, str]] = [
    # Doubled words: "the the", "is is"
    (r"\b(\w+) \1\b", "doubled word"),
    # Lowercase letter directly after sentence-final punctuation
    (r"(?<=[.!?]\s)[a-z]", "sentence-start capitalization"),
    # "a" before a vowel-initial word (basic a/an rule; ignores sound)
    (r"\ba ([aeiou])\w*\b", "article a/an"),
    # Missing space after comma or period: "hello,world" / "end.Sentence"
    (r"[a-z][,.][a-z]", "missing space after punctuation"),
    # Common uncountable/mass nouns wrongly pluralised
    (r"\b(peoples|informations|advices|furnitures|equipments|staffs)\b",
     "uncountable plural"),
    # Double comparatives
    (r"\bmore (better|worse|easier|harder|faster|slower|bigger|smaller|"
     r"higher|lower)\b", "double comparative"),
    # "be + base verb" (e.g. "I am agree") — classic L1-transfer pattern
    (r"\b(?:am|is|are|was|were) (agree|like|want|need|have|know|think|"
     r"believe|hope|love|hate)\b", "be + base verb"),
]


def _compile_rule_patterns() -> List[Tuple[re.Pattern, str]]:
    return [(re.compile(p, re.IGNORECASE), desc) for p, desc in
            RULE_BASED_ERROR_PATTERNS]


# ------------------------------------------------------------------
# spaCy (lazy)
# ------------------------------------------------------------------
_nlp = None
_nlp_model_loaded = None


def get_nlp():
    """Load (once) and return the spaCy pipeline. Safe to call repeatedly."""
    global _nlp, _nlp_model_loaded
    if _nlp is None:
        try:
            import spacy
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "spaCy is required for linguistic feature extraction. "
                "Install it with: pip install spacy && python -m spacy "
                "download en_core_web_sm"
            ) from e
        try:
            _nlp = spacy.load(SPACY_MODEL)
        except OSError as e:  # pragma: no cover
            raise RuntimeError(
                f"spaCy model '{SPACY_MODEL}' not found. Install it with: "
                f"python -m spacy download {SPACY_MODEL}"
            ) from e
        _nlp_model_loaded = SPACY_MODEL
        logger.info("Loaded spaCy model '%s'", SPACY_MODEL)
    return _nlp


# ------------------------------------------------------------------
# Grammar acceptability model (RoBERTa-CoLA, lazy)
# ------------------------------------------------------------------
_grammar_model = None


def _load_grammar_model():
    """Lazy-load the fine-tuned acceptability model
    (models/grammar_cola). Returns (model, tokenizer, device) or None."""
    global _grammar_model
    if _grammar_model is None:
        if not (GRAMMAR_MODEL_DIR / "config.json").exists():
            return None
        try:
            import torch
            from transformers import (AutoModelForSequenceClassification,
                                      AutoTokenizer)
            tokenizer = AutoTokenizer.from_pretrained(str(GRAMMAR_MODEL_DIR))
            model = AutoModelForSequenceClassification.from_pretrained(
                str(GRAMMAR_MODEL_DIR))
            model.eval()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            _grammar_model = (model, tokenizer, device)
            logger.info("Grammar engine: RoBERTa-CoLA loaded (%s)", device)
        except Exception as e:  # pragma: no cover
            logger.warning("Could not load RoBERTa-CoLA (%s); will fall "
                           "back to rule-based engine.", e)
            return None
    return _grammar_model


def _grammar_model_version():
    """Version string for the method section: base model + CoLA MCC."""
    meta_path = GRAMMAR_MODEL_DIR / "metadata.json"
    if meta_path.exists():
        try:
            import json
            m = json.loads(meta_path.read_text(encoding="utf-8"))
            return f"RoBERTa-CoLA (base={m.get('base_model')}, " \
                   f"MCC={m.get('cola_mcc')}, {m.get('date')})"
        except Exception:
            pass
    return "RoBERTa-CoLA (metadata missing)"


# ------------------------------------------------------------------
# Text preprocessing
# ------------------------------------------------------------------
class DataPreprocessor:
    """Text cleaning: BOM/line-ending normalisation, header removal,
    abbreviation expansion, trailing word-count stripping."""

    def __init__(self):
        self.abbreviations = {
            "u.s.a.": "USA",
            "u.s.": "USA",
            "u.k.": "UK",
            "e.g.": "for example",
            "i.e.": "that is",
            "etc.": "etcetera",
        }

    def clean_text(self, text: str, topic: str = "") -> str:
        if not isinstance(text, str) or not text.strip():
            return ""

        # 1. Basic cleaning
        text = text.replace("\ufeff", "").replace("\ufffe", "")
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # 2. Header removal (only when the first line is short and
        #    contains title/topic keywords)
        lines = text.split("\n")
        if lines:
            first_line = lines[0].strip().lower()
            topic_keywords = topic.lower().split() if topic else []
            is_header = False
            if len(first_line) < 100:
                is_header = (
                    first_line.startswith("title:")
                    or first_line.startswith("topic:")
                    or "title:" in first_line
                    or any(k in first_line for k in topic_keywords
                           if len(k) > 3)
                )
            if is_header:
                lines = lines[1:]

        text = " ".join([l.strip() for l in lines if l.strip()])
        text = re.sub(r"\s+", " ", text)

        # 3. Abbreviation expansion
        for abbr, full in self.abbreviations.items():
            text = re.sub(r"\b" + re.escape(abbr), full, text,
                          flags=re.IGNORECASE)

        # 4. Strip trailing word-count artifacts (Grok artifact)
        wc_pattern = r"\s*\((?:\d+\s+words?|word count[:\s]+\d+)\)\s*$"
        text = re.sub(wc_pattern, "", text, flags=re.IGNORECASE)

        return text.strip()


# ------------------------------------------------------------------
# Main feature extractor
# ------------------------------------------------------------------
class LinguisticAuditor:
    """Core analysis class: AVD/CEFR, MTLD, MDD, Biber features.

    Args:
        vocab_dir: directory containing ``cefr_full_vocab.csv``. Defaults to
            the canonical ``data/processed/`` directory.
        server_url: optional LanguageTool HTTP server (e.g.
            ``http://localhost:8010/v2/``). When provided, grammar errors are
            counted by the server; otherwise the built-in rule-based engine
            is used. The active engine is exposed as ``self.grammar_engine``
            and ``self.grammar_engine_version``.
        grammar_engine: one of ``auto`` (default), ``languagetool-http-server``
            or ``rule-based-fallback``. ``auto`` uses the server when
            ``server_url`` is set, otherwise the rule-based engine.
    """

    def __init__(
        self,
        vocab_dir: Path = None,
        server_url: Optional[str] = None,
        grammar_engine: str = GRAMMAR_ENGINE_AUTO,
    ):
        if vocab_dir is None:
            vocab_dir = DEFAULT_VOCAB_PATH.parent
        self.vocab_dir = Path(vocab_dir)
        self.server_url = server_url
        self.grammar_engine = grammar_engine

        # Resolve the effective grammar engine up front so it is stable
        # across every sample in a run (determinism).
        # Priority: roberta-cola (fine-tuned, theory-backed) > LanguageTool
        # server > rule-based fallback.
        if grammar_engine == GRAMMAR_ENGINE_AUTO:
            if (GRAMMAR_MODEL_DIR / "config.json").exists():
                self.grammar_engine = GRAMMAR_ENGINE_ROBERTA_COLA
            elif server_url:
                self.grammar_engine = GRAMMAR_ENGINE_LANGUAGETOOL
            else:
                self.grammar_engine = GRAMMAR_ENGINE_RULE_BASED
        if self.grammar_engine == GRAMMAR_ENGINE_ROBERTA_COLA:
            self.grammar_engine_version = _grammar_model_version()
        elif self.grammar_engine == GRAMMAR_ENGINE_RULE_BASED:
            self.grammar_engine_version = RULE_BASED_ENGINE_VERSION
        else:
            self.grammar_engine_version = (
                "LanguageTool server (version reported by /v2/info)")
        if self.grammar_engine == GRAMMAR_ENGINE_RULE_BASED:
            logger.warning(
                "Grammar engine: rule-based fallback v%s (no RoBERTa-CoLA "
                "model at %s, no LanguageTool server). Record this "
                "engine+version in the method section.",
                RULE_BASED_ENGINE_VERSION, GRAMMAR_MODEL_DIR)
        else:
            logger.info("Grammar engine: %s (%s)", self.grammar_engine,
                        self.grammar_engine_version)

        # 1. CEFR lexicon
        self.cefr_dict = self._load_cefr_dict(self.vocab_dir / VOCAB_FILENAME)

        # Pre-compile the rule-based patterns once.
        self._rule_patterns = _compile_rule_patterns()

    # -- CEFR -------------------------------------------------------
    def _load_cefr_dict(self, path: Path) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        if path.exists():
            try:
                # Local import keeps `import src.metrics` dependency-free;
                # pandas is only needed when an auditor is constructed.
                import pandas as pd
                df = pd.read_csv(path)
                mapping = pd.Series(
                    df.level.values, index=df.word.str.lower()
                ).to_dict()
                logger.info("Loaded %d CEFR words from %s",
                            len(mapping), path)
            except Exception as e:
                logger.error("Error loading CEFR vocab %s: %s", path, e)
        else:
            # Fail loudly instead of silently returning AVD = 0.
            raise FileNotFoundError(
                f"{VOCAB_FILENAME} not found in {path.parent}. Pass "
                f"vocab_dir= pointing at a directory that contains it "
                f"(canonical location: {DEFAULT_VOCAB_PATH})."
            )
        return mapping

    def _get_cefr_level(self, word_text: str, lemma: str) -> str:
        # Prefer the surface form; fall back to the lemma (run vs running).
        lvl = self.cefr_dict.get(word_text.lower(), "Off-List")
        if lvl == "Off-List":
            lvl = self.cefr_dict.get(lemma.lower(), "Off-List")
        return lvl

    # -- Lexical diversity (MTLD) ------------------------------------
    @staticmethod
    def _calculate_mtld(tokens: list, threshold: float = 0.72) -> float:
        """MTLD (McCarthy & Jarvis, 2010) — pure-Python implementation."""

        def mtld_calc(word_list):
            factor = 0
            ttr = 1.0
            unique_words = set()
            tokens_count = 0
            for word in word_list:
                tokens_count += 1
                unique_words.add(word)
                ttr = len(unique_words) / tokens_count
                if ttr <= threshold:
                    factor += 1
                    ttr = 1.0
                    unique_words = set()
                    tokens_count = 0
            if tokens_count > 0:
                factor += (1 - ttr) / (1 - threshold)
            return factor

        if not tokens:
            return 0.0
        fwd = mtld_calc(tokens)
        bwd = mtld_calc(tokens[::-1])
        if fwd == 0 or bwd == 0:
            return 0.0
        return len(tokens) / ((fwd + bwd) / 2)

    # -- Grammar ------------------------------------------------------
    def _check_grammar_languagetool(self, text: str) -> int:
        """Count errors via a LanguageTool HTTP server (/v2/check).

        Fails fast (RuntimeError) on connection/parse errors: a silent 0
        would corrupt the comparison across corpora.
        """
        if not self.server_url:
            raise RuntimeError("No LanguageTool server_url configured.")
        endpoint = (
            self.server_url
            if self.server_url.endswith("check")
            else self.server_url + "check"
        )
        try:
            import requests
            response = requests.post(
                endpoint, data={"text": text, "language": "en-US"},
                timeout=15,
            )
            response.raise_for_status()
            matches = response.json().get("matches", [])
            return len(matches)
        except Exception as e:
            raise RuntimeError(
                f"LanguageTool server request failed ({self.server_url}): {e}"
            ) from e

    def _check_grammar_rules(self, text: str) -> int:
        """Count rule-based errors. Deterministic, versioned (v1.0).
        Fallback engine only — see METHODOLOGY_JUSTIFICATION.md §5.1."""
        count = 0
        for pattern, _desc in self._rule_patterns:
            count += len(pattern.findall(text))
        return count

    def _check_grammar_acceptability(self, text: str):
        """Mean sentence-level P(acceptable) via RoBERTa-CoLA.

        Returns (mean_p_acceptable, n_sentences) or (None, 0) if the model
        is unavailable (caller falls back to rules). Lazy-loads the model.
        Theory: Lau et al. 2017 (acceptability is a probabilistic
        continuum), Warstadt et al. 2019 (CoLA).
        """
        model, tokenizer, device = _load_grammar_model() or (None,) * 3
        if model is None:
            return None, 0
        import torch
        nlp = get_nlp()
        sents = [s.text.strip() for s in nlp(text).sents if s.text.strip()]
        if not sents:
            return 1.0, 0
        enc = tokenizer(sents, truncation=True, max_length=128,
                        padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[:, 1]
        return float(probs.mean().cpu()), len(sents)

    def _check_grammar(self, text: str):
        """Grammar probe. Primary: mean P(acceptable) (RoBERTa-CoLA);
        fallback: rule-based count when the model is unavailable."""
        if self.grammar_engine == GRAMMAR_ENGINE_ROBERTA_COLA:
            acc, n = self._check_grammar_acceptability(text)
            if acc is not None:
                return acc
        elif self.grammar_engine == GRAMMAR_ENGINE_LANGUAGETOOL:
            return self._check_grammar_languagetool(text)
        return self._check_grammar_rules(text)

    # -- Per-sample analysis ------------------------------------------
    def analyze_sample(self, text: str) -> Optional[Dict]:
        """Main analysis entry point. Returns None for empty/short texts."""
        if not text or len(text) < 10:
            return None

        nlp = get_nlp()
        doc = nlp(text)
        words = [t for t in doc if not t.is_punct]
        word_count = len(words)
        sent_count = len(list(doc.sents))

        if word_count == 0:
            return None

        # 1. CEFR distribution (A1–C2)
        level_counts = Counter(
            {"A1": 0, "A2": 0, "B1": 0, "B2": 0, "C1": 0, "C2": 0,
             "Off-List": 0}
        )
        for t in words:
            if t.is_alpha and t.pos_ != "PROPN":
                lvl = self._get_cefr_level(t.text, t.lemma_)
                if lvl in level_counts:
                    level_counts[lvl] += 1
                else:
                    level_counts["Off-List"] += 1

        # 2. Biber features
        # Pronoun (1st/2nd person) — Biber involvement dimension
        pronoun_count = sum(
            1
            for t in words
            if t.pos_ == "PRON"
            and ("1" in t.morph.get("Person")
                 or "2" in t.morph.get("Person"))
        )
        # Nominalization
        nom_count = sum(
            1
            for t in words
            if t.pos_ == "NOUN"
            and t.text.lower().endswith(NOMINALIZATION_SUFFIXES)
        )
        # Discourse connectors
        conn_count = sum(
            1 for t in words if t.text.lower() in DISCOURSE_MARKERS
        )
        # Passive (auxpass count per sentence)
        passive_count = sum(1 for t in words if t.dep_ == "auxpass")
        passive_ratio = passive_count / sent_count if sent_count else 0

        # 3. Complexity & diversity
        tokens_text = [t.text.lower() for t in words]
        mtld = self._calculate_mtld(tokens_text)

        # 3b. Lexical distribution (Ellis 2012 "phrasal teddy bear"):
        #   Top20_Word_Share = share of all content tokens taken by the 20
        #     most frequent words (learners over-rely on high-frequency safe
        #     words; aligned LLMs avoid them -> lower share).
        #   Hapax_Ratio = share of words appearing exactly once (lexical
        #     range; learners show higher hapax due to restricted vocab).
        from collections import Counter as _Counter
        freq = _Counter(tokens_text)
        top20_share = (sum(v for _, v in freq.most_common(20)) /
                       len(tokens_text)) if tokens_text else 0.0
        hapax_ratio = (sum(1 for v in freq.values() if v == 1) /
                       len(tokens_text)) if tokens_text else 0.0

        # 3c. Lexical cohesion (Crossley & McNamara): mean adjacent-sentence
        #   content-word overlap (normalized). LLM outputs tend to be
        #   over-cohesive (smooth) vs learner texts.
        sents = list(doc.sents)
        lex_cohesion = 0.0
        if len(sents) > 1:
            sent_word_sets = [
                {t.text.lower() for t in s
                 if t.is_alpha and not t.is_stop}
                for s in sents
            ]
            overlap_sum = 0.0
            for a, b in zip(sent_word_sets, sent_word_sets[1:]):
                if a and b:
                    overlap_sum += len(a & b) / min(len(a), len(b))
            lex_cohesion = overlap_sum / (len(sents) - 1)

        # 3d. Referential density (Halliday & Hasan cohesion): share of
        #   content tokens that are pronominal references (3rd person he/she/
        #   it/they + possessives). Learner texts show different reference
        #   patterns vs smooth LLM prose.
        referential_density = 0.0
        if words:
            ref_pron = sum(
                1 for t in words
                if t.pos_ == "PRON"
                and t.text.lower() in
                {"he", "she", "it", "they", "them", "his", "her", "its",
                 "their", "this", "that", "these", "those"}
            )
            referential_density = ref_pron / len(words)

        # MDD & tree depth
        total_mdd = 0.0
        total_depth = 0

        def get_subtree_depth(node):
            if not list(node.children):
                return 1
            return 1 + max(
                (get_subtree_depth(child) for child in node.children),
                default=0,
            )

        for sent in doc.sents:
            dists = [
                abs(t.i - t.head.i)
                for t in sent if t.head != t and not t.is_punct
            ]
            if dists:
                total_mdd += sum(dists) / len(dists)
            roots = [t for t in sent if t.head == t]
            sent_depth = max(
                (get_subtree_depth(r) for r in roots), default=0
            )
            total_depth += sent_depth

        mdd = total_mdd / sent_count if sent_count else 0
        avg_depth = total_depth / sent_count if sent_count else 0

        # 4. Grammar (theory-backed engine, see METHODOLOGY_JUSTIFICATION.md)
        # Primary: mean sentence-level P(acceptable) from RoBERTa-CoLA.
        # Grammar_Error_Rate = 100 * (1 - acceptability), a continuous
        # "error-likelihood" (Lau et al. 2017). Rule-based fallback returns
        # a hit-count rate instead (engine recorded in self.grammar_engine).
        grammar_val = self._check_grammar(text)
        if self.grammar_engine == GRAMMAR_ENGINE_ROBERTA_COLA:
            grammar_acceptability = grammar_val
            grammar_rate = 100.0 * (1.0 - grammar_acceptability)
        else:
            grammar_acceptability = None  # rules engine: no probability
            grammar_rate = (grammar_val / word_count) * 100 if word_count \
                else 0.0

        # 5. Readability
        try:
            import textstat
            fkgl = textstat.flesch_kincaid_grade(text)
        except Exception:
            fkgl = 0.0

        # NOTE: output keys are kept identical to the old
        # notebooks/linguistic_lib.py so downstream notebooks keep working.
        return {
            "Word_Count": word_count,
            "Sentence_Length": word_count / sent_count if sent_count else 0,
            "AVD_Combined": (
                (level_counts["C1"] + level_counts["C2"]) / word_count
            ) * 100,
            "CEFR_A1": (level_counts["A1"] / word_count) * 100,
            "CEFR_A2": (level_counts["A2"] / word_count) * 100,
            "CEFR_B1": (level_counts["B1"] / word_count) * 100,
            "CEFR_B2": (level_counts["B2"] / word_count) * 100,
            "CEFR_C1": (level_counts["C1"] / word_count) * 100,
            "MTLD": mtld,
            "Top20_Word_Share": round(top20_share * 100, 3),
            "Hapax_Ratio": round(hapax_ratio * 100, 3),
            "Lexical_Cohesion": round(lex_cohesion, 4),
            "Referential_Density": round(referential_density * 100, 3),
            "MDD": mdd,
            "Nominalization_Rate": (nom_count / word_count) * 100,
            "Passive_Ratio": passive_ratio,
            "Discourse_Density": (conn_count / word_count) * 100,
            "Pronoun_Ratio": (pronoun_count / word_count) * 100,
            "FKGL": fkgl,
            # Grammar (RoBERTa-CoLA): mean sentence P(acceptable) in [0,1];
            # Grammar_Error_Rate = 100*(1-acc). Under the rule fallback,
            # Grammar_Acceptability is None and the rate is a hit-count rate.
            "Grammar_Acceptability": grammar_acceptability,
            "Grammar_Error_Rate": grammar_rate,
            "Tree_Depth": avg_depth,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("== src/metrics.py self-test ==")
    try:
        auditor = LinguisticAuditor()  # resolves canonical vocab, prints engine
        print(f"Grammar engine        : {auditor.grammar_engine}")
        print(f"Grammar engine version: {auditor.grammar_engine_version}")
        print(f"CEFR lexicon entries  : {len(auditor.cefr_dict)}")
    except Exception as e:
        print(f"LinguisticAuditor init skipped (deps missing?): {e}")
    print(f"Rule-based patterns   : {len(RULE_BASED_ERROR_PATTERNS)}")
    print("OK")
