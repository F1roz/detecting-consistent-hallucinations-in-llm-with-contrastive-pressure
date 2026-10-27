"""Factual sentences for \\RESTXT{key}.

Each sentence restates measured numbers or whether a criterion fixed before the
experiment was met. None of them interprets causes; interpretation is left to the
authors in \\REVIEW boxes.
"""

from __future__ import annotations

from typing import Callable

from ..config import Config
from .fmt import TODO, auroc_ci, ci, integer, is_missing, latex_escape, num, pct, pvalue

Sentence = Callable[[list[dict], dict], str]


def build_sentences(summary: dict | None, config: Config, environment: dict | None) -> dict[str, str]:
    if not summary:
        return {}
    models = [dict(summary["models"][s.key], key=s.key) for s in config.models[:2] if s.key in summary["models"]]
    if not models:
        return {}
    context = {"environment": environment or {}, "config": config}
    builders: dict[str, Sentence] = {
        "abstract.results": _abstract,
        "intro.findings": _intro,
        "setup.backend": _backend,
        "setup.judge": _judge,
        "results.split": _split,
        "results.rq1": _rq1,
        "results.rq2": _rq2,
        "results.rq3": _rq3,
        "results.rq4": _rq4,
        "results.datasets": _datasets,
        "ablation.k": _ablation_k,
        "ablation.random": _ablation_random,
        "ablation.strict": _ablation_strict,
        "ablation.alternative": _ablation_alternative,
        "ablation.combination": _ablation_combination,
        "discussion.facts": _discussion,
        "conclusion.results": _conclusion,
    }
    return {key: builder(models, context) for key, builder in builders.items()}


def _name(model: dict) -> str:
    return latex_escape(model["display_name"])


def _method(model: dict, key: str) -> dict:
    return model["methods"].get(key, {})


def _auc(model: dict, key: str) -> str:
    return num(_method(model, key).get("auroc_matched", {}).get("value"))


def _label(model: dict, key: str | None) -> str:
    return latex_escape(_method(model, key).get("label", "")) if key else TODO


def _verdict(model: dict, hypothesis: str) -> str | None:
    entry = model["hypotheses"].get(hypothesis)
    if not entry or "supported" not in entry:
        return None
    return "met" if entry["supported"] else "not met"


def _join(parts: list[str]) -> str:
    if len(parts) <= 1:
        return "".join(parts)
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def _abstract(models, ctx) -> str:
    parts = []
    for m in models:
        best = m.get("best_text_baseline")
        parts.append(f"on {_name(m)}, \\cps{{}} reached an AUROC of {_auc(m, 'cps')} on consistently wrong "
                     f"versus consistently right answers, against {_auc(m, best) if best else TODO} for the "
                     f"strongest text only baseline ({_label(m, best)})")
    text = "Measured " + "; ".join(parts) + "."
    se = [_auc(m, "semantic_entropy") for m in models]
    text += f" Semantic Entropy reached {_join(se)} on the same items."
    return text


def _intro(models, ctx) -> str:
    lines = []
    for hypothesis in ("H1", "H2", "H3", "H4"):
        met = [_name(m) for m in models if _verdict(m, hypothesis) == "met"]
        lines.append(f"{hypothesis} was met for {_join(met) if met else 'neither model'}")
    return "Of the four criteria fixed before the full run, " + "; ".join(lines) + " (Section~\\ref{sec:results})."


def _backend(models, ctx) -> str:
    env = ctx["environment"]
    parts = []
    for m in models:
        backend = m.get("backend", {})
        parts.append(f"{_name(m)} ran on {latex_escape(str(backend.get('backend', 'an unrecorded backend')))} "
                     f"with {latex_escape(str(backend.get('precision', 'unrecorded')))} weights")
    gpu = latex_escape(str(env.get("gpu", "an unrecorded GPU")))
    return f"In the reported run, {_join(parts)}, on {gpu}."


def _judge(models, ctx) -> str:
    parts = []
    for m in models:
        methods = m.get("judge_methods", {})
        total = sum(methods.values())
        if not total:
            continue
        pieces = [f"{integer(count)} by {latex_escape(_judge_label(name))}" for name, count in sorted(methods.items())]
        parts.append(f"for {_name(m)}, {_join(pieces)}")
    return ("Answers were labelled " + "; ".join(parts) + ".") if parts else TODO


def _judge_label(method: str) -> str:
    if method.startswith("llm:"):
        return method[4:]
    return {"alias": "alias match", "nli": "NLI", "refusal": "the refusal filter"}.get(method, method)


def _split(models, ctx) -> str:
    sentences = []
    for m in models:
        c = m["counts"]
        consistent = c["CW"] + c["CR"]
        share = c["CW"] / consistent if consistent else None
        truthful = c["per_dataset"].get("truthfulqa", {}).get("CW")
        sentences.append(
            f"{_name(m)} gave {integer(c['CW'])} consistently wrong, {integer(c['CR'])} consistently right and "
            f"{integer(c['INC'])} inconsistent answers out of {integer(c['total'])}, with {integer(c['EXCL'])} "
            f"refusals excluded. Consistent errors were {pct(share)} of consistent answers, and "
            f"{integer(truthful)} of them came from TruthfulQA. The matched set has {integer(m['matched']['n'])} items."
        )
    return " ".join(sentences)


def _rq1(models, ctx) -> str:
    sentences = []
    for m in models:
        se, sc = _method(m, "semantic_entropy"), _method(m, "selfcheck_nli")
        h1 = m["hypotheses"].get("H1", {})
        sentences.append(
            f"On {_name(m)}, Semantic Entropy reached {auroc_ci(se.get('auroc_matched'))} and SelfCheckGPT-NLI "
            f"{auroc_ci(sc.get('auroc_matched'))} on the matched set, against {num(se.get('auroc_all'))} and "
            f"{num(sc.get('auroc_all'))} on all answers. H1 was {_verdict(m, 'H1') or TODO}"
            f"{'' if h1.get('supported') else ' because at least one interval excludes 0.5'}."
        )
    return " ".join(sentences)


def _rq2(models, ctx) -> str:
    sentences = []
    for m in models:
        best = m.get("best_text_baseline")
        comparison = m["comparisons_vs_cps"].get(best, {}) if best else {}
        any_best = m.get("best_baseline")
        flips = m["flip_rates"]
        sentences.append(
            f"On {_name(m)}, \\cps{{}} reached {auroc_ci(_method(m, 'cps').get('auroc_matched'))}. The strongest "
            f"text only baseline, {_label(m, best)}, reached {_auc(m, best) if best else TODO}; the difference was "
            f"{num(comparison.get('difference', {}).get('value'))} {ci(comparison.get('difference'))} "
            f"(DeLong, Holm adjusted $p$ = {pvalue(comparison.get('p_holm'))})."
        )
        if any_best and any_best != best:
            sentences.append(f"The strongest baseline of any kind was {_label(m, any_best)} at {_auc(m, any_best)}.")
        cw, cr = flips.get("CW", {}).get("f_counter", {}), flips.get("CR", {}).get("f_counter", {})
        sentences.append(
            f"Under counterclaims, consistently wrong answers were abandoned in {pct(cw.get('value'))} of replies "
            f"and consistently right answers in {pct(cr.get('value'))} (Fig.~\\ref{{fig:flip}}). "
            f"H2 was {_verdict(m, 'H2') or TODO} for {_name(m)}."
        )
    return " ".join(sentences)


def _rq3(models, ctx) -> str:
    sentences = []
    for m in models:
        signal = m["ablations"].get("signal", {})
        diff = signal.get("cps_minus_f_counter")
        sentences.append(
            f"On {_name(m)}, the raw counterclaim flip rate $F_{{\\mathrm{{C}}}}$ reached {num(signal.get('f_counter'))} "
            f"and the doubt flip rate $F_{{\\mathrm{{D}}}}$ alone {num(signal.get('f_doubt'))}, against "
            f"{num(signal.get('cps'))} for \\cps{{}} (difference {num((diff or {}).get('value'))} {ci(diff)}). "
            f"H3 was {_verdict(m, 'H3') or TODO}."
        )
    return " ".join(sentences)


def _rq4(models, ctx) -> str:
    sentences = []
    for m in models:
        combo = m["ablations"].get("combination", {})
        sentences.append(
            f"On {_name(m)}, cross family disagreement alone reached {num(combo.get('cross_family'))} and its rank "
            f"mean with \\cps{{}} reached {num(combo.get('rank_mean'))}; H4 was {_verdict(m, 'H4') or TODO}."
        )
    first = models[0]
    sentences.append(
        f"The combination needs {_method(first, 'cps_cross').get('calls', TODO)} generations per question, against "
        f"{_method(first, 'semantic_entropy').get('calls', TODO)} for Semantic Entropy and "
        f"{_method(first, 'metaqa').get('calls', TODO)} for MetaQA (Fig.~\\ref{{fig:cost}})."
    )
    return " ".join(sentences)


def _datasets(models, ctx) -> str:
    sentences = []
    for m in models:
        scored = {d: e["cps"] for d, e in m["per_dataset"].items() if not is_missing(e.get("cps"))}
        if len(scored) < 2:
            continue
        high = max(scored, key=scored.get)
        low = min(scored, key=scored.get)
        sentences.append(f"For {_name(m)}, \\cps{{}} scored highest on {_dataset_name(high)} ({num(scored[high])}) "
                         f"and lowest on {_dataset_name(low)} ({num(scored[low])}).")
    sentences.append("Cells are left empty when a dataset has fewer than five items in either class.")
    return " ".join(sentences)


def _dataset_name(key: str) -> str:
    return {"truthfulqa": "TruthfulQA", "triviaqa": "TriviaQA", "sciq": "SciQ"}.get(key, key)


def _ablation_k(models, ctx) -> str:
    parts = []
    for k in ("1", "2", "3"):
        values = [num(m["ablations"]["pressure_count"].get(k, {}).get("cps")) for m in models]
        calls = models[0]["ablations"]["pressure_count"].get(k, {}).get("calls", TODO)
        parts.append(f"with {k} of each ({calls} generations), {_join(values)}")
    names = _join([_name(m) for m in models])
    return f"\\cps{{}} reached, on {names} respectively, " + "; ".join(parts) + "."


def _ablation_random(models, ctx) -> str:
    present = [m for m in models if m["ablations"].get("random_alternative")]
    if not present:
        return TODO
    parts = [f"{num(m['ablations']['random_alternative'].get('cps_random'))} on {_name(m)} (self generated "
             f"alternative on the same items: {num(m['ablations']['random_alternative'].get('cps_self'))})"
             for m in present]
    return "With a random alternative, \\cps{} reached " + "; ".join(parts) + "."


def _ablation_strict(models, ctx) -> str:
    sentences = []
    for m in models:
        strict = m["ablations"].get("strict", {})
        sentences.append(
            f"When all samples must agree, the matched set of {_name(m)} keeps {integer(strict.get('n_cw'))} CW "
            f"and {integer(strict.get('n_cr'))} CR items; \\cps{{}} reached {num(strict.get('cps'))} and the "
            f"strongest text only baseline {num(strict.get('baseline'))}.")
    return " ".join(sentences)


def _ablation_alternative(models, ctx) -> str:
    parts = [f"{integer(m['ablations'].get('alternative_equivalent_count'))} items on {_name(m)}" for m in models]
    after = [num(m["ablations"].get("excluding_equivalent_alternatives")) for m in models]
    return (f"The NLI model judged the proposed alternative equivalent to the original answer for {_join(parts)}. "
            f"Excluding these items gives \\cps{{}} an AUROC of {_join(after)}.")


def _ablation_combination(models, ctx) -> str:
    learned = [num(m["ablations"].get("combination", {}).get("logistic_cv")) for m in models]
    rank = [num(m["ablations"].get("combination", {}).get("rank_mean")) for m in models]
    return (f"The logistic regression reference reached {_join(learned)}, against {_join(rank)} for the rank mean "
            f"that needs no training.")


def _discussion(models, ctx) -> str:
    sentences = []
    for m in models:
        ranked = sorted(m["methods"].items(), key=lambda kv: -kv[1]["auroc_matched"]["value"])
        position = [k for k, _ in ranked].index("cps") + 1 if "cps" in m["methods"] else None
        sentences.append(f"On {_name(m)}, \\cps{{}} ranked {position if position else TODO} of {len(ranked)} "
                         f"methods by AUROC on the matched set.")
        cr = m["flip_rates"].get("CR", {}).get("f_doubt", {})
        sentences.append(f"Consistently right answers were abandoned under plain doubt in "
                         f"{pct(cr.get('value'))} of replies.")
    return " ".join(sentences)


def _conclusion(models, ctx) -> str:
    cps = [_auc(m, "cps") for m in models]
    se = [_auc(m, "semantic_entropy") for m in models]
    met = [m for m in models if _verdict(m, "H2") == "met"]
    verdict = ("was met for both models" if len(met) == len(models) and len(models) > 1
               else f"was met for {_join([_name(m) for m in met])}" if met else "was not met")
    names = _join([_name(m) for m in models])
    return (f"On {names}, \\cps{{}} reached AUROCs of {_join(cps)} on consistent answers, where Semantic Entropy "
            f"reached {_join(se)}. The criterion H2, fixed before the experiment, {verdict}.")
