# Independent Review of the Package (before experiments)

Written as a reviewer would read the placeholder paper and the code, 22 October 2026.
It judges the design, not results, because no results exist yet.

## Summary of the submission

The paper proposes the Contrastive Pressure Score, a text only and training free
detector for answers that a model gets wrong consistently. It challenges the answer
with counterclaims naming the model's own alternative and with plain doubt, and uses
the difference between the two flip rates. It is compared with seven baselines on a
matched set of consistently wrong and consistently right answers from two 7B to 8B
models on three datasets.

## Strengths

1. The target is narrow and well motivated: Tan et al. (2025) and Ganesh et al. (2026)
   show the failure, and no black box remedy exists.
2. The doubt control addresses the obvious objection (that flipping measures
   sycophancy) inside the score rather than in the discussion.
3. Evaluation is paired: same items, same samples, same NLI rule, DeLong with Holm and
   paired bootstrap differences. Success criteria are fixed before data exist.
4. The ablations each answer one specific question.
5. Reproducibility is unusually strong: pinned revisions, saved ids, recorded
   environment, and every number in the paper comes from one file.

## Weaknesses a reviewer will raise

| # | Point | Severity | What to do |
| --- | --- | --- | --- |
| 1 | Only two small open models. Reviewers will ask whether the effect holds for larger or API models. | major | If compute allows, add one API model (for example GPT-4o-mini) as a third target on the matched set only; otherwise state clearly in Limitations. |
| 2 | Few CW items per model can give wide intervals (TriviaQA and SciQ CW rates may be a few percent). | major | Size from the pilot; report CW counts in the first table; pool models only as a secondary analysis. |
| 3 | Correctness labels from an LLM judge. | moderate | Report Cohen's kappa on the 100 human labels; report results with alias match only as a check if kappa is low. |
| 4 | Challenge wording is fixed (three templates). Huang (2026) shows rankings depend on wording. | moderate | Keep the per template flip rates in `scores.csv`; one sentence in Limitations; consider a template ablation if pages allow. |
| 5 | NLI equivalence on short answers is noisy and drives both the split and the flip decision. | moderate | Prepend the question (done); manually check 50 flip decisions alongside the correctness labels. |
| 6 | 4-bit quantization in Colab may differ from full precision. | minor | Recorded in `environment.json`; state precision in Setup (sentence is generated). |
| 7 | The method costs 8 generations, more than log probability. | minor | The cost figure and the "use only when samples agree" deployment note answer this. |
| 8 | Novelty against MetaQA and counterfactual probing could be questioned. | moderate | Related Work already separates them on error type and the doubt control; keep MetaQA as a baseline on identical items. |
| 9 | Main track deadline (21 August 2026) has passed; S25 rules are unconfirmed. | blocking for submission | Confirm the S25 deadline, page limit and blind policy on the track page before final formatting. |
| 10 | Six bibliography entries still marked Check. | moderate | Resolve via `CITATION_VERIFICATION_REPORT.md` before submission. |

## Code review notes

- Every stage is resumable and idempotent (tested). Child processes free GPU memory
  between stages.
- Backends for vLLM and transformers 5.x are untested here. Run the smoke notebook
  before anything else; a failure there is a code fix, not a configuration change.
- The judge falls back to NLI when no API key is set; the paper sentence states which
  judge ran, so this cannot go unreported.
- No attribution strings, keys or personal paths in the repository (tested).

## Verdict on readiness

The design is sound for a workshop paper. It is **not ready to submit**: results,
human labels, author names, three interpretation paragraphs and the AI use statement
are missing by design, and the venue rules need confirmation. If the pilot fails the
go rule, the honest outcome is a negative result paper or a change of design, not a
rewrite of the criteria.
