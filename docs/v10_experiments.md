# Paper 4 v10 experiment protocol

The five extensions use published normalized protein intensities and known-truth simulation. The sampling unit is the paired donor within a cohort. Each comparison is on a log₂ scale throughout, except where natural-log isometric coordinates are identified in the tables.

## External tau-tangle application

The protocol locks MAPT (P10636), a neuronal cytoskeletal panel (MAP2, TUBB3, NEFL, NEFM), and a synaptic panel (SNAP25, SYP, STX1A, SYT1) before the extension computes effects. Panel choices were informed by annotation and coverage in the publication. Fit paired tangle-to-adjacent-neuropil contrasts in advanced AD (the primary application), intermediate AD, and primary age-related tauopathy (PART) separately. An effect relative to a panel subtracts that panel's mean paired change from the target's paired change. Report the regional contrast and the two relative contrasts on the same complete donor set with simultaneous Bonferroni 95% bounds. The original APOE and CTSD/TPP1 panels are also attempted. Coverage tables report every panel, including unavailable comparisons. This is a new lesion and study with a paired LCM–DIA design. Cross-study donor identity and inference about cell-level or subcellular localization are outside the measurements. [Thierry et al.](https://doi.org/10.1007/s00401-026-03064-9)

## Signal-to-span ratio

For the collapsed target and named reference classes, let `m` be the smallest reference-relative mean estimate and `s` the maximum minus minimum of those estimates. The diagnostic is `m/s` when `s > 1e-12` log₂. A positive call also requires a positive simultaneous lower bound for the regional contrast. At each margin (0 and +0.2 log₂), a true positive has a positive population regional change and all population relative contrasts above the specified margin. The original comparator requires corresponding simultaneous lower bounds for the regional and all relative contrasts.

The replay uses 360 v9 designs and their original simulation random streams. The v10 calibration and holdout use 276 designs each. Calibration combines shrinkage Gaussian draws with a forward ordering of reference effects. Holdout combines independent donor resampling with reversed effects. Both use the observed donor variation from the same two v9 protein families. Missingness is either absent, MCAR, or abundance-dependent MNAR. The latter learns its intercept separately under each design. Truth patterns, counts, and seeds are recorded in the protocol and scenario tables.

Select the least restrictive strict SSR threshold with a pooled estimable calibration false positive rate at most 5%. Compare that threshold with SSR > 1 and the original simultaneous interval workflow in each split. Report performance among estimable repeats and with all requested repeats as the denominator. Individual scenario Wilson intervals reflect binomial Monte Carlo error. A gate failure or undefined SSR is ranked below every finite score for the full ROC and contributes to the final tied step. The plot zooms into the false positive rates near the tested cutoffs. The measured data table includes the diagnostic ratio and simulated cutoffs, with the external study explicitly flagged as outside the simulated study design. [Morris et al.](https://doi.org/10.1002/sim.8086)

## Bayesian reference averaging

Fit one multivariate normal model to the paired regional and panel-relative donor scores. The prior is conjugate normal–inverse-Wishart with zero mean, mean precision `0.01`, degrees of freedom `d + 2` for contrast dimension `d`, and expected prior covariance `s²I` for `s` in `0.25, 0.5, 1.0` log₂. Under the common likelihood, the choice of reported reference has no likelihood effect and receives its specified prior weight in the posterior mixture. Equal choice probabilities and primary-reference probabilities of 0.5 and 0.75 are evaluated where distinct. Report probability above the margin for the selected-reference mixture and the separate posterior probability of exceeding the margin for every named reference while the regional effect is positive. For the joint event, factor out the exact probability of the rarest marginal and then draw 100,000 posterior samples conditional on that marginal event to estimate the remaining joint probability. The conditioning event, hit count, Monte Carlo standard error, and Monte Carlo interval accompany the estimate. [Hoeting et al.](https://doi.org/10.1214/ss/1009212519)

## GLS reference weights

Fit equal weights, unconstrained covariance-inverse weights proportional to `Σ⁻¹1` when the covariance is well conditioned, and nonnegative minimum-variance weights given by a 25% shrinkage toward the covariance diagonal. The last fit estimates weights on all other donors and applies them to the held-out donor, with weights summing to one. All methods use the same available panel and donor scope for each reported comparison. Outputs include donor and mean weights, estimate shifts, held-out prediction errors, and the precision-related effective number of standardized reference proteins. Weight changes alter the comparator. Prediction errors summarize out-of-fit precision on the observed donor sets.

## CoDA equivalence

For `a` targets and `b` reference proteins, the paired difference in target/reference log geometric means equals the original contrast. The same log ratio survives closure and follows from the corresponding CLR contrast. The ILR balance multiplies it by `sqrt(ab/(a+b))` in the same log base. The numeric check recomputes each coordinate from the two original arms and verifies donor estimates, rescaled simultaneous bounds, p-values, and rescaled margin decisions. Panel-specific estimates retain their donor scope. [Egozcue et al.](https://doi.org/10.1023/A:1023818214614)

## Inputs and outputs

The three published plaque input workbooks are selected by `paper4/data_manifest.json`. The exact fourth workbook, its source URL and SHA-256 appear in `paper4/v10/protocol.json`. All downloads use that provenance. The external mapping table identifies the canonical MAPT row separately from the P10636-2 entry. The matched donor rows in the external and CoDA tables identify donors using only codes from the published workbooks.

| Result | Use |
| --- | --- |
| `external_effects.csv`, `external_donor_scores.csv` | Estimates, simultaneous intervals, and Fig. 1 source data |
| `external_family_coverage.csv`, `external_available_panels.csv` | Shared and panel-specific complete donor counts or lack of coverage |
| `ssr_scenarios.csv`, `ssr_by_scenario.csv` | Scenario design, population truths, denominators, and method calls |
| `ssr_cutoffs.csv`, `ssr_performance.csv`, `ssr_roc.csv` | Fitted thresholds, split-level operating performance, and Fig. 2 source data |
| `ssr_observed.csv` | Diagnostic ratios and exploratory decisions in the measured cohort data |
| `bayes_model_average.csv`, `bayes_joint.csv`, `bayes_marginal.csv` | Reference-weight assumptions and posterior probabilities |
| `gls_estimates.csv`, `gls_weights.csv`, `gls_donor_scores.csv` | Method-level effects, fitted weights, and cross-fitting checks |
| `coda_equivalence.csv`, `coda_donors.csv` | Summary and donor-level agreement across ratio coordinates |
| `bayes_model_average.csv`, `bayes_joint.csv`, `gls_estimates.csv` | Fig. 3 source data |

The simulated repeat archive is `ssr_repeats.csv.gz`. Its reference columns follow `reference_order` in the scenario table. Repeats without an estimable contrast keep the donor count and availability flags. Every saved result is included regardless of its direction, coverage, or precision. `verification.json` records the workbook fingerprints, test outcomes, figure properties, and output fingerprints. The `--check` command checks the result bundle against those fingerprints.
