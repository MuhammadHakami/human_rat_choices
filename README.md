# Same States, Different Choices

**Human decisions improve same-goal maze transfer, and shape forgetting and vision–language–action geometry.**
WACV 2027 Workshop, Vision4Neuro · **[project page with live demos](https://muhammadhakami.github.io/human_rat_choices/)** · [paper](paper/wacv2027/vision4neuro_camera.pdf) · [supplement](paper/wacv2027/vision4neuro_camera_supp.pdf) · Lean proofs: [core](proofs/Proofs), [Mathlib](proofs_ml/ProofsML)

People (in VR) and rats (in a real arena) solved the **same** maze configurations ([de Cothi et al. 2022](https://doi.org/10.5281/zenodo.6734981)).
We train identical visual agents on their recorded choices. At the **exact states** they visited, we also train agents on random **legal** moves.
What remains is the effect of the *choice*, not of the coverage.

| Recorded people vs rats, same maze and start | Agents trained on each choice set, same start, a room none of them has seen |
|:---:|:---:|
| ![animals](assets/animals_config16_start9.gif) | ![agents](assets/agents_seed1000_room0A_familiar_trial9.gif) |
| `python play.py animals` | `python play.py agents` |

![story](paper/wacv2027/figs/fig_camera_story.png)

| Question | Result (30 paired seeds, 16 fresh pre-registered rooms, 95% CI) |
|---|---|
| **Out-of-distribution transfer** | human choices − legal moves **+9.9 pp** success [6.9, 12.9], 8.2 pp fewer revisits. Bound: frozen agents, familiar goal (moved goal +2.0 pp) |
| **Rats** | rat choices − legal moves **+0.9 pp** [−2.7, 4.4]. Species and apparatus are confounded |
| **Forgetting** | a cue-free latent-cause gate keeps **43%** of the old goal (4.7% without the gate) |
| **Bio-plausible learning** | a mushroom-body circuit (three-factor rule + awake replay) learns the new goal **+12.4 pp better than PPO** [3.8, 21.1] while keeping as much of the old goal (TOST p = .02) |
| **Vision–language–action geometry** | one view predicts the move (75% vs 47%) but not the route. Unpaired alignment matches networks trained on the same choices (FOSCTTM 0.06), not views to actions (chance) |

<p>
<img src="paper/wacv2027/figs/fig_camera_kpi.png" width="100%"><br>
<img src="paper/wacv2027/figs/fig_view_route.png" width="49%"> <img src="paper/wacv2027/figs/fig_camera_showcase.png" width="49%">
</p>

## Project page: the agents, live in the browser

[![project page](assets/site_preview.png)](https://muhammadhakami.github.io/human_rat_choices/)

The [project page](https://muhammadhakami.github.io/human_rat_choices/) runs the paper's agents in your browser: the same weights, renderer, place-belief filter and brain-like circuit, ported to JavaScript (`site/js/sim.js`). You can:

- **Play:** put any agents in any of the 16 unseen rooms, move the goal, or edit the walls.
- **Inside the agent:** watch its input, backbone, Kenyon-cell neck, heads and output at each step.
- **Learning lab:** watch the circuit learn a moved goal with its local three-factor rule and forget the old one, with or without the latent-cause gate.
- **Humans vs rats:** compare them in every maze.
- **Results:** every result chart, with every seed.

Build and check the page locally:

```bash
python tools/build_site_data.py            # site/data from the repo's code and records (add --cohort DIR for seeds 1001-1004)
node site/test/parity.mjs                  # JS vs Python: views, belief, logits ~1e-6; circuit synapses after 64 learning steps ~3e-6
node site/test/learned.mjs                 # learned PPO and mushroom-body agents vs the paper's recorded probes (30 agents)
node site/test/frozen.mjs REF.json         # frozen human/rat/legal/PPO agents, 16 rooms x 30 trials per seed, vs the records
bash tools/build_site.sh && python -m http.server -d _site 8000   # then open http://localhost:8000
```

`.github/workflows/pages.yml` re-runs the parity test and publishes `site/` (set Pages → Source: GitHub Actions). Browser runs use their own random numbers, so single runs differ from the paper's. Over 15 runs (3 priors × 5 seeds), the circuit's four probe scores match the paper's records within noise (all paired 95% CIs include 0).

## Quick start

```bash
bash setup.sh && source .venv/bin/activate   # downloads the raw maze data from Zenodo (81 MB, md5-checked)
python play.py animals        # people vs rats  -> outputs/animals_*.gif
python play.py agents         # four agents, same start -> outputs/agents_*.gif (GPU or CPU, ~15 s)
```

| Try | Command |
|---|---|
| another maze or start (25 configurations × 10 starts) | `play.py animals --config 6 --start 4` |
| the goal moved to a new place | `play.py agents --goal moved` |
| the second room of a pair | `play.py agents --room 3 --unseen` |
| pick the agents | `play.py agents --models human_bc rat_bc rat_sham cold_ppo` |
| another of the 30 seeds | `bash setup.sh --models`, then `play.py agents --seed 1017` |
| an interactive window instead of a GIF | add `--show` |

| Model (`--models`) | Trained on |
|---|---|
| `human_bc` / `rat_bc` | the recorded human / rat choice at each visited state, then PPO |
| `human_legal_uniform` / `rat_legal_uniform` | the same states, a random legal move as the label |
| `rat_sham` | the rat states with their choices shuffled |
| `cold_ppo` | no demonstrator: PPO from scratch |

## Reproduce the paper

**Level 1: every number and figure from the shipped records (CPU, minutes).** The numbers are macros written by a script, never typed by hand.

```bash
python scripts/v4n_camera_numbers_claude.py                     # -> paper/wacv2027/camera_numbers.tex
for f in story kpi showcase learning diag; do python scripts/figs/fig_camera_${f}_claude.py; done
python scripts/figs/fig_view_route_claude.py
bash scripts/build_v4n_camera.sh                                # -> both PDFs (needs pdflatex + bibtex)
(cd proofs && lake build)                                       # Lean 4: identities, trace bound, cost arithmetic
(cd proofs_ml && lake exe cache get && lake build)              # Lean 4 + Mathlib: softmax policy gradient, trace closed form
```

**Level 2: re-run the experiments (GPU, `bash setup.sh --models`).** Every study wrote `PROTOCOL.json` before any outcome. Each script runs `check` (a self-test), the study itself, then `analyse`.

| Claim | Script (`scripts/`) | Commands |
|---|---|---|
| priors: BC + PPO, 6 arms × 30 seeds | `prior_seeds30_claude.py` | `prereg`, `queue W N` |
| frozen transfer, fresh rooms | `fresh_rooms_confirm_claude.py` · `analyse_seeds30_claude.py` | `run main W N`, `analyse` |
| moved goal, greedy, loops, belief | `v4n_camera_ready_claude_20261031.py` | `probe COND W N`, `bcacc`, `analyse` |
| learning the moved goal (PPO, local rules) | `v4n_moved_goal_learning_claude_20261031.py` | `LEARNER W N`, `analyse` |
| forgetting, generalisation, spiking | `v4n_kpi_claude_20261031.py` | `kpi W N`, `snn W N` |
| context gate (G0 / G1 / G2) | `v4n_context_gating_claude_20261103.py` | `run W N`, `analyse` |
| brain circuit, stages 1–4 (Optuna NSGA-II; stage 4 lateral horn rejected) | `v4n_brainlike{,2,3,4}_claude_*.py` | `search W TRIALS`, `best`, `test W N`, `analyse` |
| Rosetta geometry and retrieval (`setup.sh --rosetta`) | `v4n_rosetta_matched_claude_20261031.py` · `v4n_rosetta_brains_claude_20261103.py` | `moments`, `features`, `rosetta`, `analyse` |

Finished records are never recomputed: a script skips any output that already exists. To check a result, move the shipped records aside, re-run, and compare them.

```bash
mv data/audit/v4n_brainlike_20261103/test /tmp/shipped && CUDA_VISIBLE_DEVICES=0 python scripts/v4n_brainlike_claude_20261103.py test 0 1
python tools/compare_records.py /tmp/shipped data/audit/v4n_brainlike_20261103/test      # outcomes must be identical
```

On the same GPU model, a re-run reproduces the records exactly (checked on an RTX 3090). The stage-2/3 commands assert the registered device index (`CUDA_VISIBLE_DEVICES=1`). The searches are post-NAS: the architecture, budget and rooms are fixed, and only the named mechanism is searched, on dev rooms. Each winner is tested once.

## What is in the repository

| Path | Contents |
|---|---|
| `play.py` | the animal and agent viewer |
| `scripts/` | every experiment, analysis and figure, byte-identical to the registered code (their hashes are in each `PROTOCOL.json`) |
| `arc_fly/` | grid-maze utilities shared by the scripts (the name is historical) |
| `data/audit/v4n_*`, `data/wacv/seeds30/` | protocols, selections, per-seed records, and the six seed-1000 checkpoints |
| `paper/wacv2027/` | LaTeX sources and PDFs. Supplement-only figures ship rendered |
| `proofs/`, `proofs_ml/` | Lean 4 projects (core; Mathlib) |
| `site/`, `tools/build_site*.{py,sh}` | the project page, its data exporter and its JS-vs-Python parity test |
| `docs/` | registered protocols and addenda |

Release assets (`bash setup.sh --models --records`): all 180 checkpoints (6 priors × 30 seeds), plus the large per-episode and feature records.

## Data and licences

- **Code:** Apache-2.0 (`LICENSE`).
- **Raw behaviour:** de Cothi et al., *Current Biology* 2022 (Zenodo 10.5281/zenodo.6734981, licence "other-open"). It is downloaded by `setup.sh` and is **not redistributed** here; `scripts/import_tartarus.py` converts it. The matched animal moments used by Rosetta are regenerated from it.
- **Optional, fetched from upstream:** the [unpaired-rosetta](https://github.com/dominik-schnaus/unpaired-rosetta) code at commit `fdfad84`, unmodified; mouse-cortex predictions from the [MICrONS foundation model](https://doi.org/10.1038/s41586-025-08829-y) (supplement only).

```bibtex
@inproceedings{hakami2027samestates,
  title     = {Same States, Different Choices: Human Decisions Improve Same-Goal Maze Transfer and Shape Forgetting and Vision--Language--Action Geometry},
  author    = {Hakami, Muhammad Y.},
  booktitle = {WACV Workshops (Vision4Neuro)}, year = {2027}}
```
