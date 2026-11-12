# Horse-Race Prediction and Betting Strategies

Conditional logit and neural models for Hong Kong horse racing, with probability diagnostics and retrospective betting simulations.

This project studies how horse characteristics, recent performance, race conditions, and market odds can be combined to estimate winning probabilities. It compares **conditional logit**, **deep logit**, and **two-tower logit** models, then uses ranking probabilities to explore win, place, quinella, and quinella-place betting.

The analysis is implemented in [model.ipynb](model.ipynb). It was developed using historical Hong Kong Jockey Club data.


## Repository files

| File | Role in the repository | Read directly by `model.ipynb`? |
| --- | --- | --- |
| [model.ipynb](model.ipynb) | Feature engineering, model fitting, probability evaluation, and betting experiments | Main notebook |
| [combined.csv](combined.csv) | Prepared race–horse table used for modeling | Yes |
| [place_odds.csv](place_odds.csv) | Place odds used as predictors and in place-bet evaluation | Yes |
| [q_qp_odds.csv](q_qp_odds.csv) | Quinella and quinella-place odds for horse pairs | Yes |
| [horse_info_new.csv](horse_info_new.csv) | Supporting dataset | No |
| [race_horse.csv](race_horse.csv) | Supporting dataset | No |
| [race_result.csv](race_result.csv) | Supporting dataset | No |
| [sectional_time_clean.csv](sectional_time_clean.csv) | Supporting dataset | No |

## Models

All three models assign a utility to each runner and normalize utilities within the same race:

$$
p_{ij}=\frac{\exp(\eta_{ij})}{\sum_{h=1}^{H_i}\exp(\eta_{ih})},
$$

where $H_i$ is the field size and $p_{ij}$ is the estimated winning probability of horse $j$ in race $i$. Padding and masks exclude unused runner positions from the model's softmax normalization.

| Model | Utility specification | Implementation |
| --- | --- | --- |
| Conditional logit | Linear combination of predictors | Both TensorFlow/Keras and SciPy implementations; both support L1/L2 penalties |
| Deep logit | A shared multilayer perceptron produces one scalar utility per runner | TensorFlow/Keras, with Keras Tuner random search |
| Two-tower logit | Separate horse-side and race-side representations are combined by an inner product | TensorFlow/Keras, with Keras Tuner random search |

The downstream variable `model_clogit` is the **SciPy implementation**, whose current fitting cell sets both penalties to zero. The notebook also includes a separate regularized TensorFlow conditional-logit fit. These are distinct experimental paths.

## Data requirements

### `combined.csv`

The modeling table needs these fields:

- **Identifiers:** `date`, `race_no`, `horse_no`, `horse_code`, `horse_url`.
- **Results and odds:** `place`, `time`, `win_odds`.
- **Horse and rider information:** `actual_weight`, `declare_weight`, `draw`, `rating`, `gear`, `jockey`, `sex`, `country_of_origin`, `colour`.
- **Race information:** `venue_full`, `distance`, `going`.

The notebook creates `race_id` by grouping `date` and `race_no`. It uses `horse_code` for histories across races and `horse_no` for positions in race-level arrays. Finishing positions include string-coded results and special statuses that the cleaning cells interpret.

### Odds files

| File | Columns used |
| --- | --- |
| `place_odds.csv` | `date`, `race_no`, `horse_no`, `place_odds` |
| `q_qp_odds.csv` | `date`, `race_no`, `horse_no1`, `horse_no2`, `q_odds`, `qp_odds` |

Place odds are merged with the modeling table on `date`, `race_no`, and `horse_no`. Pair odds are matched to races using `date` and `race_no`, then pivoted by the pair `(horse_no1, horse_no2)`. Probability arrays, outcomes, and odds tables must retain the same race and runner ordering.

## Features and experimental settings

Feature engineering includes log odds, previous-race implied probabilities, weight changes, recent horse and jockey placing rates, gear changes, previous special conditions, time since earlier starts, and measures of suitability to the current distance and surface.

The initial conditional-logit specification defines **26 horse predictors**, **five race variables**, and **130 horse-by-race interactions**. Its full model input contains the 26 horse predictors and 130 interactions. The tuning section temporarily uses a smaller feature set; see the execution-state note below.

The neural pipeline is constructed separately. It uses **16 horse predictors**, **four race variables**, and **16 within-race predictor means**, giving 36 inputs to deep logit. The two-tower model separates these into 32 horse-side inputs and four race-side inputs. This notebook does not automatically transfer an elastic-net-selected feature set into the neural models.

### Configured date split

| Partition | Date condition |
| --- | --- |
| Training | `2015-01-01 <= date < 2023-06-01` |
| Test | `2023-06-01 <= date < 2024-01-01` |

### Hardcoded final fits

| Implementation | Architecture | L1 / L2 coefficients | Fitting settings |
| --- | --- | --- | --- |
| TensorFlow conditional logit | Scalar linear utility, no bias | `0.05` / `0.0001` | Adam, learning rate `0.001`, 300 epochs; early-stopping callback is not passed to this final fit |
| SciPy conditional logit | Scalar linear utility | `0` / `0` | `scipy.optimize.minimize`, `depth=1` |
| Deep logit | Four hidden layers, 256 units each, ReLU; dropout defaults to zero | `0.001` / `0.01` | Adam, learning rate `0.0005`, up to 300 epochs; early-stopping patience 10 |
| Two-tower logit | Zero hidden layers; 128-dimensional projection in each tower | `0.1` / `0.0001` | Adam, learning rate `0.0005`, up to 300 epochs; early-stopping patience 30 |

## Evaluation and betting strategies

### Probability evaluation

The notebook computes winner negative log-likelihood, Keras accuracy and cross-entropy, and a binned goodness-of-fit diagnostic. A market baseline is constructed by normalizing inverse win odds within each race.

The binned diagnostic sums `(observed - expected)^2 / expected`, where observed counts are the numbers of winners in probability bins and expected counts are the sums of predicted winning probabilities. The bin boundaries are:

```text
0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 1
```

The code does not attach a degrees-of-freedom adjustment or a formal significance test to this statistic.

### Betting pools and allocation rules

Harville-style ordering probabilities are used to derive top-three place probabilities and the probabilities of successful quinella and quinella-place tickets.

| Pool | Event modeled |
| --- | --- |
| Win | A selected horse finishes first |
| Place | A selected horse finishes in the first three |
| Quinella | A selected pair finishes first and second in either order |
| Quinella place | Both horses in a selected pair finish in the first three |

The implemented allocation rules include:

- **Fixed stakes:** 1% of the initial bankroll per qualifying ticket by default.
- **Proportional stakes:** 1% of the current bankroll per qualifying ticket by default.
- **Kelly-type allocation:** stake fractions based on predicted probabilities and odds; the notebook contains two implementations under the same function name.

A ticket qualifies when `predicted_probability * odds - 1 > threshold`. The payoff calculation treats odds as **gross payout multipliers**, including the returned stake on a successful ticket.

`performance()` starts from a bankroll of **100** by default and plots the bankroll path. `compare_threshold()` reports **terminal bankroll return**, calculated as `final_bankroll / initial_bankroll - 1`. The current notebook evaluates pools separately and does not contain the mixed-pool allocation procedure described in the project report.


## Generated outputs

Successful execution of the relevant cells writes:

| Output | Contents |
| --- | --- |
| `model_deeplogit.keras` | Saved deep-logit model |
| `model_ttlogit.keras` | Saved two-tower model |
| `ConditionalLogitModel_tuner/` | Conditional-logit tuning state |
| `DeepLogit_tuner/` | Deep-logit tuning state |
| `TwoTowerLogit_tuner/` | Two-tower tuning state |

Tuner projects use `overwrite=True`. Rerunning those cells replaces existing tuning state for the same project. Plots and diagnostics remain inline; the notebook does not explicitly export prediction CSVs, figure files, or a conditional-logit model. The fitted scalers and preprocessing state are not saved, and loading the custom Keras models requires their class definitions and an appropriate deserialization setup.

## Authors

- WONG Yin San
- KAN Chun Yu Matthew
- LEE Ching Man Jenny
- LAU Sze Miu

## References

- Bolton, R. N., and Chapman, R. G. (1986). *Searching for Positive Returns at the Track: A Multinomial Logit Model for Handicapping Horse Races*. Management Science, 32(8), 1040–1060. [DOI](https://doi.org/10.1287/mnsc.32.8.1040).
- Harville, D. A. (1973). *Assigning Probabilities to the Outcomes of Multi-Entry Competitions*. Journal of the American Statistical Association, 68(342), 312–316. [DOI](https://doi.org/10.1080/01621459.1973.10482425).
- Kelly, J. L., Jr. (1956). *A New Interpretation of Information Rate*. Bell System Technical Journal, 35(4), 917–926. [DOI](https://doi.org/10.1002/j.1538-7305.1956.tb03809.x).
- Zou, H., and Hastie, T. (2005). *Regularization and Variable Selection via the Elastic Net*. Journal of the Royal Statistical Society: Series B, 67(2), 301–320. [DOI](https://doi.org/10.1111/j.1467-9868.2005.00503.x).
