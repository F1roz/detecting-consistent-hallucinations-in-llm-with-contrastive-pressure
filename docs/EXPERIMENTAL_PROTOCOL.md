# Experimental Protocol

Fixed before the full run. Any change after the pilot is recorded in the change log at
the end with its reason, so readers can tell planned analyses from later ones.

## 1. Problem statement

Detectors that sample a model several times treat disagreement as a sign of
hallucination. When a model gives the same wrong answer every time, these detectors
see agreement and miss the error. Tan et al. (EMNLP 2025) showed that such errors stay
level or grow with model size and that Semantic Entropy scores near chance on them.
Their remedy needs hidden states of two models and labelled training data.

**Question.** Using only generated text and no training, can the reaction of an
answer to a named counterclaim, corrected for its reaction to plain doubt, separate
consistently wrong answers from consistently right ones?

## 2. Research questions and hypotheses

| Id | Question | Criterion for support |
| --- | --- | --- |
| H1 | Do sampling detectors fail on CW versus CR? | 95% bootstrap intervals of Semantic Entropy and SelfCheckGPT-NLI AUROC both include 0.5 |
| H2 | Does CPS separate CW from CR? | CPS AUROC at least 0.65, above the strongest text only baseline, Holm adjusted DeLong p below 0.05 |
| H3 | Does the doubt control help? | CPS AUROC above the AUROC of the raw counterclaim flip rate F_C |
| H4 | Does cross family disagreement add to CPS? | Rank mean of CPS and X above both components |

Each criterion is evaluated per model and printed in `summary.json` under
`hypotheses`. The paper states which criteria were met; it does not reinterpret a
missed criterion as support.

**Conditions under which the main hypothesis is not supported:** CPS AUROC below
0.65, or not above the best text only baseline, or the Holm adjusted p value at or
above 0.05.

## 3. Variables

| Kind | Variables |
| --- | --- |
| Independent | detection method (9 levels); target model (2); dataset (3); number of challenges k (1, 2, 3); alternative source (self generated, random); consistency threshold (0.9, 1.0) |
| Dependent | AUROC on CW vs CR (primary); AUROC on all answers; AURAC; generations per question; wall clock seconds |
| Controlled | questions and sample ids; prompts; decoding settings; NLI model; judge; seeds; matched set; imputation rule |
| Possible confounders | judge errors; NLI errors on short answers; model specific sycophancy; quantization on small GPUs; dataset mix (controlled by matching within dataset) |

## 4. Design

Paired, within item design: every method scores the same items, and where a method
uses samples, the same samples. This is what makes the DeLong test and paired
bootstrap valid.

### 4.1 Data

| Dataset | Split | Size | Selection |
| --- | --- | --- | --- |
| TruthfulQA | generation / validation | 817 | all |
| TriviaQA | rc.nocontext / validation | 1,000 | seeded random sample |
| SciQ | test | 1,000 | all (the split has 1,000 items) |

The pilot uses 100, 50 and 50 items. Sample ids are saved in
`results/<run>/questions.jsonl`. Dataset revisions are pinned by commit hash in
`src/pressure_probe/data.py`.

### 4.2 Models

Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct (revisions pinned in
`configs/full.yaml`). On GPUs with less than 20 GB the code loads 4-bit weights (AWQ
through vLLM, NF4 through transformers). The precision used is recorded per run and
reported in the paper.

### 4.3 Generation

- System prompt: "Answer with a short phrase only. Do not explain."
- Greedy answer at temperature 0; 10 samples at temperature 0.5, top-p 1, top-k off,
  repetition penalty 1; at most 32 new tokens.
- Pressure replies are greedy, in a second turn after the model's own answer.

### 4.4 Labels

1. Refusal filter: excluded from all analyses.
2. Alias match after normalisation (whole words).
3. Otherwise GPT-4o-mini with correct and incorrect references; if no API key, NLI
   bidirectional entailment against each reference.
4. Two authors label 100 consistent answers (50 CW, 50 CR, shuffled); Cohen's kappa
   reported.

### 4.5 Split and matched set

- Agreement r(q): share of samples equivalent to the greedy answer (DeBERTa-large-MNLI,
  bidirectional entailment with the question prepended).
- CW: r at least 0.9 and wrong. CR: r at least 0.9 and right. INC: otherwise.
- Matched set: every CW item plus the same number of CR items sampled within each
  dataset with a fixed seed.

### 4.6 Methods

| Method | Access | Generations per question |
| --- | --- | --- |
| Token log probability | logits | 1 |
| P(True) | logits | 2 |
| Verbalized confidence | text | 2 |
| Semantic Entropy | text | 11 |
| SelfCheckGPT-NLI | text | 11 |
| Cross family disagreement X | text | 2 |
| MetaQA (5 synonym, 5 antonym mutations) | text | 13 |
| CPS (k = 3) | text | 8 |
| CPS + X, rank mean | text | 9 |

## 5. Statistical analysis

- Primary metric: AUROC on the matched set, CW positive.
- Uncertainty: stratified bootstrap, 2,000 resamples, 95% percentile intervals.
- Comparisons: CPS against each of the seven baselines with the paired DeLong test
  (fast midrank algorithm), Holm correction across the seven tests, and paired
  bootstrap intervals for AUROC differences as effect sizes.
- Secondary: AUROC and AURAC on all consistent and inconsistent answers, for context.
- Missing scores: replaced by the median of that method's scores (counted and stored).
- No threshold is tuned. The logistic regression is reported only as a reference
  ceiling with out of fold predictions (5 fold stratified cross validation).

## 6. Ablations

All reuse collected replies except the random alternative (k extra generations).

| Ablation | Question it answers |
| --- | --- |
| k = 1, 2, 3 | How many challenges are needed? |
| F_C alone, F_D alone | Does each part carry signal; does the subtraction help (H3)? |
| Random alternative | Does plausibility of the named alternative matter? |
| Threshold 1.0 | Do results hold for the strictest definition of consistency? |
| Equivalent alternatives removed | Do alternatives that repeat the answer distort CPS? |
| Rank mean vs logistic regression | How much does training add over the rule? |

## 7. Leakage and fairness controls

- The alternative b is generated by the model and never taken from references.
- No threshold, weight or hyperparameter is fitted on evaluation items; the only
  fitted model is the out of fold reference regression.
- All methods receive the same items; budgets differ by design and are reported.
- Results are reported for every method and model, whether favourable or not.

## 8. Change log

| Date | Change | Reason |
| --- | --- | --- |
| 2026-10-22 | Protocol fixed | before pilot |
