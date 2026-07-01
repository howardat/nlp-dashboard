# NLP & Analytics for Sales Conversations — Research Brief

**Scope:** Proven methods to extract insights and detect *bottlenecks* and *problem markers* (language / logic / pricing mistakes, client frustration, objections) from thousands of sales-agent conversations, fast.

**Grounded in this project:** ~1,170 real DoroMarine AI-sales-agent transcripts (children's marine supplement, Kazakhstan/CIS). Mixed Russian + Kazakh, misspelled, code-switched. Success = phone number captured into CRM. Existing stack: `LaBSE → UMAP → HDBSCAN → Ollama-label` clustering + a per-turn LLM signal labeler + SQLite + Streamlit. Audience for the demo: a non-technical founder.

> **The one structural idea.** Resist making a single LLM-judge prompt do everything. Split problem-marker detection into **three engines** with very different reliability:
> 1. **Deterministic** (pricing/arithmetic mistakes) — highest precision, *not* an LLM.
> 2. **In-language ML** (sentiment/frustration, objections, dialogue acts) — analyze RU/KZ directly.
> 3. **Translate-then-English-tooling** (agent hallucination, coherence) — the best models here are English-only.
>
> This split is the spine of everything below.

---

## 1. The data reality that constrains every method

Almost all published precision/recall numbers come from clean, English, often human-human data. Your text is **short, noisy, code-switched RU+KZ sales chat**. Treat every headline metric as an *upper bound* you won't match first pass — expect to lose 10–25 points moving to noisy/multilingual ([fine-tuned vs zero-shot, arXiv:2406.08660](https://arxiv.org/abs/2406.08660)).

Two consequences that recur throughout:
- **Code-switching is the failure mode of "analyze-in-language."** Multilingual LLMs are *not* good code-switchers and underperform much smaller fine-tuned models on mixed-language text ([EMNLP 2023](https://aclanthology.org/2023.emnlp-main.774/)). mBERT/XLM-R technically "include" Kazakh but allocate it minimal capacity ([Turkic CA survey, arXiv:2407.05006](https://arxiv.org/html/2407.05006)) — so fine-tune, don't trust zero-shot Kazakh.
- **Small local models (your Ollama gemma) are in the danger zone for quantitative labels.** The largest study to date (13M labels, 18 LLMs, 37 tasks) found standard practice yields wrong statistical conclusions in ~31% of hypotheses for SOTA models and **~50% for smaller models** — including wrong-sign and exaggerated-magnitude errors ([LLM Hacking, Baumann et al. 2025](https://arxiv.org/abs/2509.08825)). Your per-turn objection-% and sentiment numbers must be validated before they're presented as facts (see §6).

---

## 2. Detecting problem markers (turn- and conversation-level)

### Engine A — Deterministic: numeric / pricing / calculation errors
**This is your highest-precision, most impressive win, and it is *not* an LLM judge.** LLMs have a structural "validation gap": they can compute arithmetic but fail to *verify* it, with 10–30% error even on simple numeric comparison, and they don't notice when a tool returns a wrong number ([Validation Gap, arXiv:2502.11771](https://arxiv.org/pdf/2502.11771); [Tools Fail, arXiv:2406.19228](https://arxiv.org/pdf/2406.19228)).
**Method:** LLM (or regex) *extracts* quantities from agent turns → recompute price/discount/total **in code** against a small product price table → flag mismatches. Reliable now, given the price list.

### Engine B — In-language ML
**Objection detection & classification.** No canonical academic taxonomy exists; the standard practitioner set is **price / trust / competitor / timing / "need to think" / authority** — which is essentially what `labeler.py` already encodes. The reference production pipeline is [AWS Comprehend's objection classifier](https://aws.amazon.com/blogs/machine-learning/identify-objections-in-customer-conversations-using-amazon-comprehend-to-enhance-customer-experience-without-ml-expertise/) (supervised multi-label over labeled turns). Fine-tuned small models beat zero-shot LLMs by 10–25 points on fine-grained labels ([arXiv:2406.08660](https://arxiv.org/abs/2406.08660), [Chae & Davidson 2026](https://journals.sagepub.com/doi/10.1177/00491241251325243)).
- **Objection *presence* (binary): reliable now.**
- **Fine-grained 6-way *type*: degrades on bilingual noise** — expect macro-F1 ~0.6–0.7, not 0.9.
- **Recommendation:** keep the LLM few-shot labeler as a *bootstrapper*; label a few thousand turns, hand-correct a sample, distill into a fine-tuned XLM-R / multilingual-E5 classifier for the at-scale pass.

**Sentiment & frustration — trajectory, not level.** The valuable signal is the *shift*, not absolute polarity; negative sentiment reliably *precedes* escalation ([Supportbench](https://www.supportbench.com/sentiment-analysis-b2b-detecting-frustration-before-escalation/), [Crescendo](https://www.crescendo.ai/blog/customer-sentiment-analysis)). Your `sentiment_delta` is exactly the right primitive. Kazakh sentiment is tractable: fine-tuned XLM-R reached F1≈0.87 on KazSAnDRA ([arXiv:2403.19335](https://arxiv.org/html/2403.19335v1), [IJCT](https://ijctjournal.org/transformer-based-resource-kazakh-language/)).
- Prefer embedding/LLM scoring over lexicons (lexicons collapse on misspelling).
- Derive **conversation-level features**: negative-run-length, min sentiment, max-to-min drop. These are far more robust than any single-turn score.
- **Orthographic effort markers survive misspelling** because they're not lexical: repeated questions, ALL-CAPS, punctuation bursts (`?!`, `???`), repair requests ("что?", "не поняла"). Cheap and robust. **Reliable now.**

**Dialogue-act tagging.** Don't adopt a 200-tag scheme. Use a small 6–10 act inventory (question / request / complaint / commitment / rejection / info-provide / greeting-closing), framed via **MIDAS** (purpose-built for *human-machine* dialogue, F1≈0.79 — [arXiv:1908.10023](https://arxiv.org/abs/1908.10023v1)) and kept mappable to **ISO 24617-2** for credibility. Apply with LLM few-shot. **Reliable now** because the inventory is coarse.

### Engine C — Translate-then-English-tooling: agent-side failures
This is where the project *differentiates* — detecting when the **AI agent** made the mistake, not the client.

- **Hallucination / groundedness (most mature).** Build a small structured KB (prices, dosages, claims, shipping) and run **NLI-grounding** of each agent turn against it — **MiniCheck / AlignScore / Vectara HHEM** ([HHEM leaderboard](https://github.com/vectara/hallucination-leaderboard), [awesome-hallucination-detection](https://github.com/EdinburghNLP/awesome-hallucination-detection)). When there's no source doc, **SelfCheckGPT** self-consistency works (AUC-PR ~92.5 NLI variant — [arXiv:2303.08896](https://arxiv.org/abs/2303.08896)). **Reliable now for the closed-world (price/product/policy) subset; aspirational for open-domain claims.** The consistency models are English-centric → translate the agent turn first.
- **Dialogue breakdown — adopt a ready-made schema.** The **DBDC** challenge defines the agent utterance that makes a conversation un-continuable, labeled **NB / PB / B** (no / possible / breakdown) ([DBDC, ACL L16-1502](https://aclanthology.org/L16-1502/)). Adopt these labels directly; an LLM-as-judge prompted with the DBDC definitions is the pragmatic implementation. **Reliable now** as a coarse agent-failure flag.
- **Ignored-question / non-sequitur.** Detect a client question (from dialogue-act tagging) → check whether the *next* agent turn answers it (NLI/relevance). Plus: repair requests in the *next client* turn are a strong cheap signal the *previous agent* turn failed ([arXiv:2510.24628](https://arxiv.org/pdf/2510.24628)). **Partially reliable now**; full coherence-NLI is aspirational on noisy bilingual text.

### Marker reliability summary
| Marker | Best approach | Status |
|---|---|---|
| Pricing / arithmetic error | LLM extracts → **recompute in code** | ✅ Reliable, highest precision |
| Frustration / sentiment trajectory | XLM-R per-turn → conv-level features + orthographic signals | ✅ Reliable |
| Objection *presence* | LLM few-shot → distill to fine-tuned encoder | ✅ Reliable |
| Dialogue breakdown (agent fail, coarse) | DBDC NB/PB/B via LLM-judge | ✅ Reliable |
| Dialogue acts (coarse) | MIDAS-style small inventory, LLM few-shot | ✅ Reliable |
| Agent hallucination (price/product facts) | Structured KB + NLI-grounding (translate first) | 🟡 Needs KB |
| Ignored-question | Question-detect → next-turn-answers check | 🟡 Partial |
| Fine-grained objection *type* (6-way) | Fine-tuned XLM-R | 🟠 ~0.6–0.7 F1 |
| Open-domain hallucination (no KB) | SelfCheckGPT | 🟠 Costly/lossy on RU+KZ |
| Language-mistake detection in KZ | — | 🔴 Genuinely low-resource |

---

## 3. Detecting bottlenecks (where conversations die)

### 3.1 Stage funnel + drop-off survival curve — the dashboard spine
1. **Induce stages** (greeting → discovery → pitch → objection → close) by clustering utterance embeddings into intent states ([unsupervised intent induction, arXiv:2307.15410](https://arxiv.org/pdf/2307.15410)) — you have no stage labels and messy text, so let embeddings find the stages.
2. **Funnel:** count conversations reaching each stage; the largest single-stage drop is the headline bottleneck ([ChatNexus](https://articles.chatnexus.io/knowledge-base/conversation-flow-analysis-optimizing-chatbot-dial/)). Most founder-legible output you can produce.
3. **Survival / hazard analysis on drop-off turn.** Plot % of conversations still alive at turn *k* (survival) and instantaneous death risk per turn (hazard). This is the rigorous version of "where do chats die," and it **natively handles censoring** — abandoned chats that just stop — which a naive average-turns metric mishandles ([dialogue survival, arXiv:2510.02712](https://arxiv.org/pdf/2510.02712)). **Cox proportional-hazards** adds covariates (language, sentiment, stage) to ask *what raises the death risk*.

> Your schema already stores `drop_off_turn` — you're one survival curve away from this.

### 3.2 Markov transition model
Model the conversation as a Markov chain over stages with absorbing **closed** / **lost** states; the probability of reaching "lost" from each state quantifies which *transitions* bleed deals. A **removal-effect** computation (drop a state, watch conversion fall — the marketing-attribution trick) ranks the most load-bearing stage ([Markov attribution precedent](https://www.triplewhale.com/blog/markov-chain-attribution)). Upgrades the funnel from "which stage" to "which transition," and captures the loops real chats have.

### 3.3 Escalation / human-takeover prediction — you have free ground truth
Your `assistant type=manager` turns are labeled handoffs, and your `error` role is a free strong feature. The academic anchor is **Machine-Human Chatting Handoff (MHCH)**; the DAMI model + **Golden-Transfer-within-Tolerance** metric judges handoff timing with tolerance, not exact-match ([arXiv:2012.07610](https://arxiv.org/abs/2012.07610)).
**Predictive features** (literature consensus): client repetition (bot didn't resolve), negative-sentiment spike, out-of-scope/low-confidence (your `error` role), explicit "оператор"/human asks, multi-question complexity, turns without progression.
**Do it as logistic regression / gradient boosting** — the *coefficients/SHAP themselves answer* "what makes the bot need rescuing?" Prediction and explanation in one model.

### 3.4 Contrastive won-vs-lost — "what winning conversations say differently"
- **Fightin' Words (Monroe et al. 2008)** — log-odds-ratio with an informative Dirichlet prior. Purpose-built for *short, sparse* text: the prior stops rare misspelled RU/KZ n-grams from producing garbage spikes that break naive TF-IDF / chi-square ([Monroe PDF](https://languagelog.ldc.upenn.edu/myl/Monroe.pdf), [ConvoKit impl](https://convokit.cornell.edu/documentation/fightingwords.html)). Output: ranked, signed list of phrases distinguishing closed vs lost. **Strongly recommended — low effort, high narrative payoff.**
- **SHAP on a won/lost classifier** over engineered features (question rate, token-share, sentiment slope, turns, CRM-capture, error count, language mix) — shows which *behaviors* (not words) separate outcomes, with sign/magnitude.
- Back individual-metric claims with **Mann-Whitney U / chi-square + Benjamini-Hochberg** correction (you'll test many metrics — keep it honest).

### 3.5 Conversation-intelligence signals that transfer to async text
From Gong/Chorus research, the ones that survive the move from audio *calls* to text *chat*:
- **Question rate** (won calls ~15–16 Q vs lost ~20; open > count — [Gong](https://www.gong.io/blog/talk-to-listen-conversion-ratio)). Directly countable.
- **Next-step / commitment** — Gong's #1 deal-risk signal is a *stalled next step*. **Your CRM tool-call IS the committed next step** — presence and turn-position give you this almost for free.
- **Role token-share** (text analogue of talk-to-listen; lost deals show 64% rep-talk — [Gong](https://www.gong.io/resources/labs/talk-to-listen-conversation-ratio/)).
- **Client reply-gap** (final gap before abandonment; live-chat: 53% abandon if no response in 3 min — [Helpable](https://www.gethelpable.com/blog/live-chat-response-time-benchmarks)).
- Audio-specific signals (talk-time *seconds*, interruptions, monologue length) **do not transfer** — drop them.

### 3.6 Trend / change-point for the time-series views
- **STL decomposition** to separate trend from weekday seasonality on volume/conversion/abandonment.
- **CUSUM change-point detection** to flag *when* a metric durably shifted (e.g., conversion dropped after a prompt change) — far more actionable to a founder than transient anomaly noise ([CUSUM overview](https://towardsdatascience.com/probabilistic-cusum-for-change-point-detection-121f793ab3a1/)). Your series is short (~1,170) → use descriptively, not for real-time alerting.

---

## 4. Insight extraction at scale — upgrading the existing pipeline

### 4.1 Your clustering IS BERTopic minus the useful parts
`clustering.py` is, almost line-for-line, the first four stages of **BERTopic** (embed → UMAP → HDBSCAN), but stops before the parts that make it trustworthy. Adopting BERTopic (LaBSE/E5 backend + your Ollama model as the `representation_model`) gives you, with *less* code:
- **c-TF-IDF keywords** — deterministic, auditable topic descriptors shown *alongside* the LLM label, so a mislabeled cluster is no longer invisible ([algorithm docs](https://maartengr.github.io/BERTopic/algorithm/algorithm.html)).
- **`.reduce_outliers()`** — reassigns HDBSCAN's `-1` noise to nearest topics (directly attacks your biggest weakness).
- **Better LLM prompt** — BERTopic feeds the LLM c-TF-IDF keywords + representative docs, not your "first 12 raw messages."
- **Topic merging, guided/seeded topics** (seed known DoroMarine concerns: цена/дорого, доставка, состав/безопасность, эффект, конкурент), and **dynamic topics over time** (price-objection share month-over-month — a great demo artifact).

### 4.2 Swap the embedding model: LaBSE → multilingual-E5
LaBSE was trained narrowly on *translation pairs* — great at "is A a translation of B," which is **not your task**. You want intra-language intent grouping, which E5's clustering/retrieval training targets. On MTEB, LaBSE ~45.2 overall vs mE5-base 59.5 / mE5-large 61.5 ([mE5 report, arXiv:2402.05672](https://arxiv.org/html/2402.05672v1)). mE5-base is also *smaller/faster* (~1.1GB vs LaBSE ~1.9GB).
> ⚠️ **Critical gotcha:** mE5 *requires* input prefixes. Prefix every message with `"query: "` for clustering — skipping it silently degrades E5 ([model card](https://huggingface.co/intfloat/multilingual-e5-large)). Validate the swap on *your* data (RU/KZ code-switch is off-distribution for both); keep LaBSE as the validation baseline.

### 4.3 Fix HDBSCAN tuning (your noise problem)
- **`min_cluster_size=5` is ~2 orders of magnitude too low** for thousands of points → fragmentation + noise. Use ~1–2% of N (≈30–75 for 3–5k turns).
- **Set `min_samples` explicitly to ~½ of `min_cluster_size`** (it currently defaults to `min_cluster_size`, which is aggressive). *This is your single most effective lever on the noise fraction.*
- Use `cluster_selection_epsilon` to merge micro-clusters; try `eom` vs `leaf`.
- Tighten UMAP first: `n_neighbors=15`, `min_dist=0.0`.
- **Evaluate without labels** via **DBCV** / HDBSCAN's `relative_validity_` (density-aware; silhouette fails on non-globular clusters — [DBCV](https://en.wikipedia.org/wiki/Density-based_clustering_validation)). Grid-search params to maximize `relative_validity_` subject to a noise ceiling + human readability of the top-N clusters.

### 4.4 Harden the LLM labeler
- **Codebook + 2–3 gold examples per field** and an explicit "ambiguous → abstain/null" rule.
- **Schema-enforced decoding + self-healing retry** (prompt-only JSON fails 5–20% on complex schemas); let the model write a `rationale` field *before* constrained fields (forcing JSON can degrade reasoning — [structured outputs](https://tianpan.co/blog/2025-10-29-structured-outputs-llm-production)).
- **Use the confidences you already emit:** route low-confidence turns to an "uncertain" bucket, exclude from headline stats, or escalate to a stronger model.
- **Self-consistency** (sample 3–5×, majority-vote) for subjective fields; disagreement = low confidence.
- **Tiered models:** small local model for explicit fields (`faq_flag`, objection presence) at full scale; escalate only ambiguous turns.

### 4.5 Anomaly panel — "weird conversations to read" (nearly free)
You already run HDBSCAN, which exposes `outlier_scores_` (GLOSH). The smallest valid clusters + lowest-membership-probability points *are* your anomaly list — surface them as a review panel. Optionally add an **Isolation Forest** on conversation-level embeddings ([IsoForest](https://www.geeksforgeeks.org/machine-learning/anomaly-detection-using-isolation-forest/)).

### 4.6 Speed
- **Cache embeddings** keyed by message hash; only embed new turns. Your code re-embeds every run.
- **Drop the double UMAP fit** (you fit UMAP twice — 20D for clustering, separate 2D for viz). BERTopic reuses one reduction.
- Batch/GPU the encoder; mE5-base is faster than LaBSE.
- **The real bottleneck is LLM labeling, not embed/cluster** — cache label results by content hash; only relabel changed clusters.

---

## 5. KPIs & the dashboard

### Lead with 6–8 views (1–3 headline; 4–8 drill-down)
1. **Phone-capture (conversion) rate — North Star, top-left.** Benchmark band: lead-capture chatbots convert **15–35%** of conversations vs ~2% for static forms ([Amra & Elma](https://www.amraandelma.com/chatbot-lead-conversion-statistics/), [ZoomInfo](https://pipeline.zoominfo.com/marketing/chatbot-metrics)).
2. **Lost-conversation breakdown by failure cluster** (taxonomy-labeled), each as **pattern + N + %** — the centerpiece. (Lead with *patterns + frequency*, not anecdotes.)
3. **Drop-off by stage** (greeting → qualifying → price → ask-for-number).
4. **Objection-handling success rate** by type.
5. **Escalation/handoff rate paired with abandonment** — exposes "silent give-ups." ⚠️ *Low escalation + high abandonment = failure masquerading as success* ([Decagon](https://decagon.ai/glossary/what-is-chatbot-containment-rate)). Healthy handoff band: 15–30%.
6. **Fallback/confusion rate, segmented RU vs KZ** — surfaces where misspelled multilingual input breaks the agent.
7. **Turns-to-conversion** (won vs lost) — friction indicator.
8. **Sentiment trend (aggregate only)** — never a per-chat verdict (end-of-conversation sentiment correlates with CSAT only ~0.4 — [Thematic](https://getthematic.com/insights/contact-center-sentiment-analysis)).

Everything for a *first* dashboard is in the transcripts. Only true CSAT (needs a survey) and cross-session FCR (needs user-ID linkage) require outside data. **Don't fake per-chat CSAT.**

### Containment / escalation benchmarks (founder yardstick)
- Containment: conversational support **50–65%**; best-in-class AI **70–80%**; `<50%` signals an NLP bottleneck ([Netguru](https://www.netguru.com/blog/chatbot-kpis)).
- Cost: chatbot conversation ~$0.50 vs ~$6.00 human — the containment-savings story.
- *(Vendor stats skew optimistic — present as ranges, not promises.)*

### Presentation principles (evidence-backed)
Insight not data; **North Star top-left**; **prioritize failure clusters by $ recoverable** (lost chats × plausible win-rate × order value) so the founder fixes the *expensive* problem, not the loud one; narrative "finding → 3 real examples → one-line fix"; one governed definition of "conversion."

---

## 6. The product thesis made concrete: failure cluster → grounded prompt fix

A four-step loop the dashboard runs per cluster — this is the defensible mechanism behind the self-service promise:

1. **Cluster + label** lost conversations into a failure taxonomy. Templates: **Drift-Bench**'s four classes — *Flaw of Intention / Premise / Parameter / Expression* ([arXiv:2602.02455](https://arxiv.org/pdf/2602.02455)) — adapt cleanly to sales: misread-intent / wrong-assumption / wrong-detail / bad-phrasing, plus sales-specific (asked-for-number-too-early, ignored-objection, non-compliant health claim, ignored-Kazakh-message).
2. **Contrast won vs lost** within the same cluster to isolate the differentiating agent behavior ([Learning from Contrastive Prompts, arXiv:2409.15199](https://arxiv.org/pdf/2409.15199)).
3. **Generate the fix via failure-driven Automatic Prompt Optimization.** Feed the LLM the failing snippets + current system prompt; it writes a "textual-gradient" critique and emits a concrete prompt edit ([Pryzant APO](https://cameronrwolfe.substack.com/p/automatic-prompt-optimization), [stability-aware APO, arXiv:2601.22373](https://arxiv.org/pdf/2601.22373)). Productized in DSPy/MIPRO and [MLflow Prompt Optimization](https://mlflow.org/prompt-optimization).
4. **Ground + validate.** Every suggestion ships with 2–3 *cited* real examples and an estimated recoverable-conversion impact, and is **back-tested on held-out conversations** before being recommended. Ungrounded "make it friendlier" is noise; *"in these 14 lost chats the customer said 'дорого' and the agent just repeated the price — add this value-anchoring rebuttal; chats that already do this converted at X% vs Y%"* is shippable.

**The founder-facing card per cluster:** *Pattern (N chats, X% of losses, ~$Z recoverable) → 3 real examples → one specific prompt line to add → expected lift.*

> ⚠️ **LLM-as-judge has measurable biases** in the eval/scoring steps: position (up to ~30% reversal on order swap), verbosity (longer = scored higher), self-enhancement (10–25% own-output preference) ([bias study, arXiv:2606.19544](https://arxiv.org/html/2606.19544)). Mitigate: clear rubric, randomize/swap order, prefer binary labels, human spot-check. Frameworks: **RAGAS** (faithfulness/answer-relevancy, no ground-truth needed), **τ²-bench** for multi-turn customer-service agents (agents lose up to 25 task-success points moving from single-shot to interactive multi-turn — [Sierra](https://sierra.ai/uk/blog/benchmarking-agents-in-collaborative-real-world-scenarios)).

---

## 7. Validation protocol (the single most credibility-building thing to show)

1. **Gold set:** stratified-sample 150–300 client turns (oversample lost/abandoned + each objection type).
2. **Two human annotators**, same codebook. Measure **Cohen's κ between humans first** — if humans can't agree (κ < ~0.6) the construct is too subjective to report (likely true for `sentiment_delta`). NLP norm κ ≥ 0.67 ([IAA / Cohen's κ](https://medium.com/data-science/inter-annotator-agreement-2f46c6d37bf3)).
3. **Score the labeler vs gold:** per-field precision/recall/F1, κ(LLM, human-consensus), calibration (do confidence buckets match accuracy?). Show these *next to* every headline metric.
4. **Guard against LLM-hacking:** check conclusions are stable across ≥2–3 prompt paraphrases and ≥2 models; if sign/significance flips, label the metric "indicative, not measured." Regression-correct aggregates against the gold set rather than reporting raw LLM counts ([Baumann et al. 2025](https://arxiv.org/abs/2509.08825)).
5. **Validate clustering separately:** report DBCV / `relative_validity_` + noise fraction; have a human confirm top-N cluster labels match members (c-TF-IDF keywords make this fast).
6. **Lock a regression set** — freeze it; re-run on every embedding/param/prompt change so "improvements" are measured, not vibes.

---

## 8. Prioritized roadmap (impact ÷ effort)

| # | Action | Impact | Effort |
|---|---|---|---|
| 1 | Cache embeddings + drop the double UMAP fit | High | Low |
| 2 | Fix HDBSCAN params (`min_cluster_size`↑30–75, set `min_samples`≈½) | High | Low |
| 3 | Swap LaBSE → multilingual-e5-base **with `"query: "` prefix** | High | Low |
| 4 | **Pricing/arithmetic error detector** (extract → recompute in code) | High | Low–Med |
| 5 | **Stage funnel + drop-off survival curve** (uses existing `drop_off_turn`) | High | Med |
| 6 | **Fightin' Words** closed-vs-lost (ConvoKit) | High | Low |
| 7 | Adopt **BERTopic** (E5 + Ollama representation; c-TF-IDF + `.reduce_outliers()`) | High | Med |
| 8 | **Validation harness** (§7); treat labeler stats as provisional until validated | High | Med |
| 9 | **Escalation classifier** on `manager` label → SHAP for "why the bot needs rescuing" | Med–High | Med |
| 10 | Sentiment-trajectory + orthographic frustration features | Med–High | Med |
| 11 | DBDC NB/PB/B agent-failure flag (LLM-judge) | Med | Med |
| 12 | Agent-hallucination grounding vs a small product-fact KB | Med | Med |
| 13 | Anomaly panel from HDBSCAN `outlier_scores_` | Med | Low |
| 14 | Failure-cluster → grounded prompt-fix loop (the product thesis) | High | Med–High |

**Deliberately deprioritized:** audio-specific CI metrics (talk-time seconds, interruptions), fine-grained 6-way objection typing as a *headline* number, real-time CUSUM alerting (series too short), uplift modeling (no clean intervention yet), open-domain hallucination without a KB.

## 9. Honest caveats to put in front of the founder
- **abandoned ≠ lost** — survival/censoring framing keeps that distinction honest.
- **Manager-takeover outcomes are confounded** — operators may cherry-pick easy saves, so "human-closed" rates are correlational, not causal, until an A/B/uplift design exists.
- **Sentiment ≠ CSAT** at per-chat level (r≈0.4) — trend only.
- **Low escalation can be a *bad* sign** if abandonment is high.
- **Small-model quantitative labels are ~50%-unreliable** until validated against gold (§7) — that validation step is itself the most trust-building artifact in the demo.
- **Always back-test generated prompt fixes** on held-out conversations.

---

### Key sources by theme
- **Bottleneck/funnel/survival:** [intent induction 2307.15410](https://arxiv.org/pdf/2307.15410) · [dialogue survival 2510.02712](https://arxiv.org/pdf/2510.02712) · [MHCH/DAMI 2012.07610](https://arxiv.org/abs/2012.07610) · [Fightin' Words](https://languagelog.ldc.upenn.edu/myl/Monroe.pdf) · [Gong talk-to-listen](https://www.gong.io/resources/labs/talk-to-listen-conversation-ratio/)
- **Problem markers:** [fine-tuned>zero-shot 2406.08660](https://arxiv.org/abs/2406.08660) · [AWS objection pipeline](https://aws.amazon.com/blogs/machine-learning/identify-objections-in-customer-conversations-using-amazon-comprehend-to-enhance-customer-experience-without-ml-expertise/) · [KazSAnDRA 2403.19335](https://arxiv.org/html/2403.19335v1) · [MIDAS 1908.10023](https://arxiv.org/abs/1908.10023v1) · [DBDC L16-1502](https://aclanthology.org/L16-1502/) · [SelfCheckGPT 2303.08896](https://arxiv.org/abs/2303.08896) · [Vectara HHEM](https://github.com/vectara/hallucination-leaderboard) · [Validation Gap 2502.11771](https://arxiv.org/pdf/2502.11771)
- **Topic modeling/extraction:** [BERTopic](https://maartengr.github.io/BERTopic/algorithm/algorithm.html) · [mE5 report 2402.05672](https://arxiv.org/html/2402.05672v1) · [HDBSCAN params](https://hdbscan.readthedocs.io/en/latest/parameter_selection.html) · [DBCV](https://en.wikipedia.org/wiki/Density-based_clustering_validation) · [LLM Hacking 2509.08825](https://arxiv.org/abs/2509.08825)
- **KPIs/eval/prompt-fix:** [Netguru chatbot KPIs](https://www.netguru.com/blog/chatbot-kpis) · [Decagon containment](https://decagon.ai/glossary/what-is-chatbot-containment-rate) · [chatbot lead-conversion stats](https://www.amraandelma.com/chatbot-lead-conversion-statistics/) · [RAGAS/frameworks](https://atlan.com/know/llm-evaluation-frameworks-compared/) · [τ²-bench](https://sierra.ai/uk/blog/benchmarking-agents-in-collaborative-real-world-scenarios) · [APO/textual gradients](https://cameronrwolfe.substack.com/p/automatic-prompt-optimization) · [Contrastive prompts 2409.15199](https://arxiv.org/pdf/2409.15199) · [LLM-judge bias 2606.19544](https://arxiv.org/html/2606.19544)
