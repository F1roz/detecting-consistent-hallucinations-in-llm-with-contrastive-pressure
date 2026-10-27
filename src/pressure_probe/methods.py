"""The detectors compared in the paper: column name, label, access type and cost."""

from __future__ import annotations

from dataclasses import dataclass

from .config import Config


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    access: str  # "text" needs only generated text; "logits" needs token probabilities
    scope: str  # "all" scored on every question; "matched" only on the matched set
    is_ours: bool = False


METHODS = (
    Method("logprob", "Token log probability", "logits", "all"),
    Method("ptrue", "P(True)", "logits", "all"),
    Method("verbal", "Verbalized confidence", "text", "all"),
    Method("semantic_entropy", "Semantic Entropy", "text", "all"),
    Method("selfcheck_nli", "SelfCheckGPT-NLI", "text", "all"),
    Method("cross_family", "Cross family disagreement", "text", "all"),
    Method("metaqa", "MetaQA", "text", "matched"),
    Method("cps", "CPS (ours)", "text", "matched", True),
    Method("cps_cross", "CPS + cross family (ours)", "text", "matched", True),
)

BASELINES = tuple(m for m in METHODS if not m.is_ours)


def calls_per_question(config: Config) -> dict[str, int]:
    """Generations each method needs per question at deployment, including the answer itself."""
    samples, pressure, mutations = config.n_samples, config.n_pressure, config.metaqa_mutations
    return {
        "logprob": 1,
        "ptrue": 2,
        "verbal": 2,
        "semantic_entropy": 1 + samples,
        "selfcheck_nli": 1 + samples,
        "cross_family": 2,
        "metaqa": 1 + 2 + 2 * mutations,
        "cps": 1 + 1 + 2 * pressure,
        "cps_cross": 1 + 1 + 2 * pressure + 1,
        "f_counter": 1 + 1 + pressure,
        "f_doubt": 1 + pressure,
    }
