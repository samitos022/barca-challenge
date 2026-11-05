# Temporal possession v1 — experimental

## Status

Real-match inference completed on 2026-11-05; all eight synthetic tests pass and the
generated submission validates (199,319 rows). The first run exposed missing `N/A`
team attribution in Smart Tagging; these tags now provide no team evidence, while
unknown club names still fail explicitly. No parameters were changed.

Public-window pipeline check: spellF1 **0.3556**, macroF1 **0.5537**, accuracy **0.7037**.
The best shipped baseline by spellF1, smooth125, scores 0.4242 on that same window.
The method predicts 10 possessions versus 17 reference possessions and detects only
6.54% of dead-ball frames. These are five-minute diagnostics, not full-match results
or evidence of SOTA performance; do not tune parameters against them.
Outputs: `evaluation/results/team_temporal.csv`, its JSON sidecar, and the scorer's
ignored `evaluation/scoreboard.csv` and `evaluation/reports/`.
Defaults are explicit engineering hypotheses, not trained or tuned on the public window.

## Run

From the repository root in the Python environment described in INIZIA_QUI.md:

```bash
python starter/temporal_possession.py
python evaluation/validate_submission.py evaluation/results/team_temporal.csv
python evaluation/run_evaluation.py
```

Run the last two commands only after generation succeeds. The scorer uses the public
five minutes for pipeline verification, not parameter selection. Private full-match
scores remain unavailable. There is no training or ground-truth access in generation.

Required inputs under `data/tactical/` (or `--data-dir PATH` / `MTAH_DATA_DIR`):

- `BAR-ATH_tactical_atd.xml`
- `BAR-ATH_tactical_atd.txt`
- `BAR-ATH_tactical_smart_tagging.xml`

Video and homography are not needed for v1. Metadata determines track teams as in the
provided baseline. Smart Tagging team names must match metadata after case/accent/spacing
normalization. If they differ, provide their exact XML spellings with `--home-tag` and
`--away-tag`; never infer the mapping from public answers.

The output includes the CSV and a JSON sidecar with parameters, aggregate diagnostics
and the CSV SHA-256. Both generated files remain excluded from Git. With identical inputs,
code and numerical environment this method is deterministic; clean-environment reproduction
still needs verification with the actual bundle. Do not publish restricted data or extracts.

## Method

1. Stream raw tactical tracking, validating exactly one consecutive 1-based ID per video
   frame. Convert x/y to metres. Do not use ground-truth counter offsets for ATD alignment.
2. Measure each team's minimum distance to the ball and median observed player speed.
   Missing ball measurements supply no proximity or dead-ball evidence. Reject the endpoints
   of ball jumps exceeding 45 m/s; do not derive speed across a missing observation.
3. Turn distance into a smooth close-control score (2 m scale, weight 2). This is a proxy
   for control, not a claim of physical contact. Add bounded Smart Tagging support (0.45)
   for offensive phases and the opponent of defensive phases. Duplicate tags do not amplify
   the score; contradictory tags can support both sides. Unknown codes are recorded.
4. Add dead-ball support for an observed ball beyond a 0.5 m pitch margin (weight 3).
   A ball stationary below 0.5 m/s for at least 1 s adds dead-ball support only within a
   set-piece tag (weight 2.45). Low median player speed adds 0.25 only in that context.
   Neither a set-piece tag nor stillness alone forces D.
5. Divide per-frame evidence by fps and use Viterbi to maximize accumulated evidence minus
   1.5 score-seconds per state transition. Transitions are penalized, not forbidden; there
   is no hard minimum possession duration. Future observations can affect earlier decisions.
6. Decode halves independently using the organizers' published video blocks. Emit X outside
   those blocks and for a wholly unsupported half. Within a supported half the model chooses
   H/A/D; it has no calibrated uncertainty-based abstention yet. Exact ties use deterministic
   state order H, A, D, so unsupported stretches can inherit a state without strong evidence.

## Scope and limitations

This implements a first version of signal fusion and temporal decoding, not the full research
proposal. It does not reconstruct missing ball trajectories, use player-ball relative velocity,
train a neural model, infer identity-stable player tracks, or implement semi-Markov durations.
No claim of improved scoring is justified until evaluated appropriately.

Dead-ball detection is particularly provisional: a stationary ball during a broad set-piece
phase may already be in play; fouls away from boundaries may be missed. Erroneous positions
outside the field can suggest false stoppages. Persistent near-ball proximity does not establish
controlled possession; aerial balls and deflections remain ambiguous. Long missing stretches
can be assigned too confidently. Transition penalties can merge genuine short possessions.
Tag boundaries are not trusted as exact recovery/restart times, and missing tags are neutral.

Inspect data structure and debug implementation without hand-labelling this match or adding
frame-specific rules. Select/tune weights only using permitted external games, not the public
five minutes. Synthetic tests check software behaviour, not football accuracy.

## Verification

```bash
python -m unittest discover -s tests -v
```

Tests include exhaustive comparison of Viterbi to the true optimum on a tiny sequence,
occlusion and spurious switches, genuine changes and dead-ball intervals, half isolation,
defensive tag direction, exclusive interval endpoints, correlated duplicate tags, absence of
ball evidence, set-piece context, and tracking alignment/coordinate conversion.

## Research inspiration

The implementation is original and deliberately simpler than these methods:

- Vidal-Codina et al. (2022), control geometry and motion:
  https://link.springer.com/article/10.1007/s12283-022-00381-6
- TCN possession estimation (2023), temporal context:
  https://www.sciencedirect.com/science/article/pii/S0957417423002816
- PathCRF (2026), structured sequence decoding:
  https://arxiv.org/abs/2602.12080

These papers' results are not directly comparable to the competition's spellF1.

## Primo metodo del team

La versione sperimentale con tracking, Smart Tagging e Viterbi è in
`starter/temporal_possession.py`. I test sintetici funzionano senza
il pacchetto dati; generare e valutare le nuove predizioni richiede il pacchetto.
