# Error-Aware Knowledge Tracing

This repository investigates whether misconception annotations improve Bayesian
Knowledge Tracing (BKT) in tutor–student mathematics dialogues. It reproduces a
published MathDial BKT baseline, models five misconception families as latent
two-state chains, and tests whether their causal pre-turn states improve
correctness prediction through emission integration, transition integration, or
late fusion.

The main result is negative but informative: the current misconception
annotations are strongly associated with correctness on the turn where they are
observed, but the state inferred from earlier turns adds little reliable
predictive information beyond BKT. None of the tested integrations produces a
robust held-out improvement over the baseline.

## Research question

The project asks:

> Does a causally available latent misconception state improve turn-level
> correctness prediction beyond a standard per-KC BKT model?

The causal qualification is important. A misconception annotation attached to a
student response cannot be used to predict that same response. The models may
use the solution row and annotations from earlier turns, predict the current
turn, and only then update their states from the observed turn.

## Dataset and preprocessing

The repository contains the MathDial train and test CSV files in `data/`.
Preprocessing is implemented in `scripts/load_data.py` and reproduces the
paper-level filtering used by the baseline experiment:

1. Apply the dialogue typicality threshold.
2. Remove dialogues with failed correctness and KC annotation.
3. Apply the final-turn self-correctness override.
4. Remove real turns without KCs or correctness labels.
5. Retain dialogues with at least two tagged real turns.
6. Preserve the solution/turn-0 row for misconception-state initialization.

After filtering:

| Split | Dialogues | Tagged real turns | KC pseudo-observations |
|---|---:|---:|---:|
| Train | 2,050 | 10,448 | 23,953 |
| Test | 515 | 2,500 | 5,788 |

The paper-aligned test protocol scores 1,985 turns after excluding the first
retained tagged real turn in each test dialogue. That first turn still updates
the BKT state.

The five modeled misconception families are:

- comprehension
- relevance
- principles
- wrong operation
- steps

Their annotation symbols are `P` (present), `A` (evidence of absence), and `N`
(no informative observation).

## Experimental design

### BKT baseline

The baseline fits one four-parameter BKT model per KC using expectation
maximization. A multi-KC turn is exploded into one observation per KC and the
resulting KC probabilities are averaged back to the turn level. KCs unseen
during training receive a probability of `0.5`.

The NumPy implementation reproduces the published MathDial baseline closely:

| Metric | Published | Reproduced |
|---|---:|---:|
| Accuracy | 60.71% | 60.65% |
| AUC | 64.19% | 64.28% |
| F1 | 56.71% | 55.60% |

Accuracy and AUC agree within 0.1 percentage points. The remaining F1 difference
is attributed to a numerical guard in the referenced pyBKT implementation that
moves near-threshold predictions without materially changing their ranking.

### Misconception chains

Each misconception family is represented by a two-state hidden Markov chain:

- inactive: the misconception is not currently active;
- active: the misconception is currently active.

The solution row initializes the chain. Predictions are filtered pre-turn
states conditioned only on strictly earlier observations. The selected
"chain-of-record" configuration uses a low initial probability and onset rate
with strongly separated emissions, making a `P` observation act as a trigger.

Both five-family chains and broader conceptual/procedural grouped chains are
evaluated.

### Integration experiments

The project tests three integration locations:

1. **Emission integration** — misconception state changes the probability of a
   correct response conditional on BKT mastery.
2. **Transition integration** — misconception state suppresses the BKT learning
   transition.
3. **Late fusion** — frozen BKT predictions are combined algebraically with
   frozen misconception forecasts.

Pooling variants include maximum, mean, noisy-OR, and top-two noisy-OR. Grouped
models instead provide separate conceptual and procedural channels.

## Results

No integration produces a reliable improvement over the baseline. The highest
observed AUCs are only a few thousandths above it and were selected from grids
inspected on the test set, so they are descriptive rather than confirmatory.

| Experiment | Best observed test AUC | Interpretation |
|---|---:|---|
| BKT baseline | 0.6428 | Baseline of record |
| Pooled emission integration | 0.6440 | Tiny test-selected change; not reliable |
| Grouped emission integration | 0.6417 | Below the baseline |
| Pooled transition integration | 0.6436 | No detectable learning-rate contribution |
| Grouped transition integration | 0.6430 | Free parameters improve nothing |
| Late fusion, linear max pool | 0.6441 | AUC +0.0013, but F1 −0.0110 |

For the strongest late-fusion row, a dialogue-level bootstrap estimated an AUC
change of approximately `+0.0013` with a 95% interval from `−0.0006` to
`+0.0033`. The interval contains zero.

## Why the misconception integrations did not work

Several related limitations explain the consistent result across model
families.

### 1. The annotations are contemporaneous, not reliably predictive

On scored test turns, 87.5% of turns with at least one current `P` annotation are
incorrect, compared with 15.7% of turns without one. This is strong diagnostic
information, but using the current annotation to predict the current response
would be target leakage.

The causal forecast from earlier annotations correlates with current error at
only about `0.11` and with the error remaining after BKT at about `0.05`. The
annotation is informative once the response has happened; its history does not
reliably forecast the next response.

### 2. The inferred states saturate

The chain-of-record produces fitted misconception dwell times of roughly 11–89
turns, considerably longer than most dialogues. Once activated, a chain
therefore remains close to active for much of the dialogue. On scored test
turns, max-pooled misconception evidence has a median near `0.885`; noisy-OR is
even more saturated, with a median near `0.918`.

The resulting signal behaves more like "a misconception occurred earlier in
this dialogue" than a sensitive turn-level state.

### 3. Most signal is redundant between dialogues

High misconception-state dialogues are generally more difficult, but BKT
already captures much of that difficulty through correctness and KC history.
After removing each dialogue's mean, the association between misconception
state and BKT residual error reverses rather than strengthening. The chains do
not identify which later turns BKT will get wrong.

### 4. Pooling has no opportunity or relevance map

The data does not explicitly say whether a turn offered an opportunity to make
each misconception. Consequently, an absent or neutral annotation can mean:

- the student successfully avoided an applicable misconception;
- the misconception was irrelevant to the turn;
- the response did not expose enough evidence;
- or the annotation was simply uninformative.

Pooling all five families treats them as if they were equally relevant to every
turn. Mean pooling dilutes a potentially relevant family, while noisy-OR
compounds floor probabilities and max pooling is dominated by the most
persistent family. Grouping families further discards already-sparse detail.

### 5. Integration parameters compete with ordinary BKT parameters

In emission models, misconception coefficients compete with per-KC slip and
guess. In transition models, suppression coefficients compete with the learned
transition rate. Different parameter combinations can explain similar outcome
sequences, making the added parameters difficult to identify. Several fitted
coefficients are sensitive to initialization or collapse toward zero.

### 6. Some fusion rules mechanically lower predictions

The baseline already predicts fewer correct classifications than the observed
correct rate at the `0.5` threshold. Product and gated overlays lower these
probabilities further. With saturated misconception forecasts, product fusion
can reduce a typical prediction by almost an order of magnitude, causing F1 to
collapse even when ranking changes little.

### 7. Turn correctness is inherited by every KC

MathDial supplies one correctness label for a multi-KC turn. Exploding the turn
copies that label to every listed KC, even when only one KC caused the failure.
This contaminates several KC histories at once and allows ordinary BKT slip,
guess, prior, and learning parameters to absorb incorrectly attributed errors.
The dataset cannot determine which KC application succeeded or failed.

## Future work

The main future-work requirement is a different annotation design rather than
another fusion equation over the existing labels.

### Opportunity-aware annotations

For every turn and misconception family, annotate an opportunity variable:

$$
O_{f,t}=1
$$

when the turn genuinely affords an opportunity to express misconception family
`f`. Conditional on that opportunity, annotate the outcome as present or
absent. Preserve a separate not-applicable/not-assessable value when the chain
should not update.

The opportunity set should be determined from the problem, required KCs, and
possible solution routes before observing correctness. If opportunity depends
on an unobserved route, the model should marginalize over possible routes rather
than use the realized response to predict itself.

### Individually supervised inner chains

A future dataset should provide:

- KC-specific correctness for every applicable KC opportunity;
- misconception present/absent labels for every applicable misconception
  opportunity;
- explicit not-applicable values;
- and, where possible, links between misconception families, KCs, and solution
  routes.

This enables two clean inner-chain systems:

```text
KC-specific outcomes          -> KC mastery BKT chains
misconception opportunities   -> misconception-avoidance BKT chains
question correctness          -> global outer slip and guess only
```

If `absent` is coded as successful avoidance and `present` as failed avoidance,
a misconception chain estimates mastery of avoiding that misconception. It
updates only on applicable opportunities.

### Controlled outer-model comparison

Let the required KC chains pool to `x_KC` and the applicable misconception
avoidance chains pool to `x_M`. A conjunctive error-aware model can use:

$$
x = x_{KC}x_M
$$

and connect latent success to question correctness through global slip and
guess:

$$
P(\text{correct}) = x(1-s_0) + (1-x)g_0.
$$

The decisive ablation would compare:

1. the current inherited-label BKT;
2. individually annotated KC chains pooled through the global outer link;
3. the same KC model plus independently supervised misconception-avoidance
   chains.

The second-minus-first comparison measures improved KC attribution. The
third-minus-second comparison isolates the incremental value of misconception
latents. All models should use identical dialogue- or participant-level folds,
predict-before-update evaluation, training-only model selection, and paired
confidence intervals. Shuffled misconception labels should be included as a
negative control.

The misconception trajectories may also be more valuable as diagnostic outputs
than as predictors. That use should be evaluated separately from pre-response
correctness prediction.

## Repository structure

```text
.
├── data/
│   ├── mathdial_train.csv
│   └── mathdial_test.csv
├── scripts/
│   ├── load_data.py
│   ├── bkt_model.py
│   ├── chain.py
│   ├── misconception_chains.py
│   ├── misconception_chains_grouped.py
│   ├── pooling.py
│   ├── emission_integration_pooled.py
│   ├── emission_integration_grouped.py
│   ├── transition_integration_pooled.py
│   ├── transition_integration_grouped.py
│   └── overlay.py
├── 01_eda_mathdial.ipynb
├── 02_bkt_baseline.ipynb
├── 03_misconception_chains.ipynb
├── 04_grouped_chains.ipynb
├── 05_emission_integration_pooled_misconception_chains.ipynb
├── 06_emission_integration_unpooled_grouped_chains.ipynb
├── 07_transition_integration_pooled.ipynb
├── 08_transition_integration_grouped.ipynb
├── 09_overlay_late_fusion.ipynb
└── requirements.txt
```

## Notebook guide

| Notebook | Purpose |
|---|---|
| `01` | Dataset audit, paper-filter replication, sparsity and annotation EDA |
| `02` | Reproduce and freeze the paper-aligned BKT baseline |
| `03` | Fit and compare five-family misconception-chain configurations |
| `04` | Evaluate conceptual/procedural grouped chains |
| `05` | Add pooled misconception state to BKT emissions |
| `06` | Add grouped conceptual/procedural states to BKT emissions |
| `07` | Add pooled misconception state to BKT learning transitions |
| `08` | Add grouped states to BKT learning transitions |
| `09` | Test frozen post-hoc and pseudo-KC late-fusion rules |

Run the notebooks in numerical order when reproducing the complete analysis.

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Usage

Start JupyterLab from the repository root:

```bash
jupyter lab
```

To load the paper-filtered data in Python:

```python
from scripts.load_data import load_paper_filtered_data

train_df = load_paper_filtered_data("data/mathdial_train.csv")
test_df = load_paper_filtered_data("data/mathdial_test.csv")
```

To train and evaluate the baseline:

```python
from scripts.bkt_model import BKTModel

model = BKTModel(train_df, test_df, seed=221)
metrics = model.run()
print(metrics)
```

## Reproducibility and limitations

- The baseline uses seed `221` and one pyBKT-style EM initialization per KC.
- Reported downstream grids are exploratory because variants were compared on
  the supplied test split.
- Some custom joint-EM validity rows do not exactly equal the baseline because
  their initialization and optimization engines differ. Comparisons against
  those rows are therefore descriptive.
- The late-fusion scalar is fitted from in-sample training predictions;
  dialogue-level out-of-fold stacking is a stricter future design.
- The repository currently uses notebook-based checks rather than a standalone
  automated test suite.
- Data use and redistribution remain subject to the terms of the upstream
  MathDial dataset.

## Citation

The baseline experiment follows the MathDial BKT evaluation reported in:

> Scarlatos, Baker and Lan. *Exploring Knowledge Tracing in Tutor-Student
> Dialogues using LLMs*. LAK 2025.

When using this repository, cite both the upstream paper/dataset and this
implementation as appropriate.

## License

No software license is currently included in this repository. Add an explicit
license before distributing or reusing the code outside its present research
context.
