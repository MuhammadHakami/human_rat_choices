# Vision4Neuro camera-ready: every reviewer question, its answer, and the evidence (2026-10-31)

Paper: `paper/wacv2027/vision4neuro_camera.tex` (+ `camera_numbers.tex`, `camera_claims.tex`); supplement
`vision4neuro_camera_supp.tex`. Submitted files unchanged. Source: the decision and four reviews pasted by the author
(the OpenReview API returned a bot challenge). Earlier version of this map: `_v1.md`.

## The story every answer links to
Real humans rarely step back onto a visited cell (3.8% of moves; rats 14.4%). At the SAME recorded states, agents that
imitate human choices inherit this (fewer revisits, fewer loops) and that is what transfers to unseen rooms; legal
labels at the same states (coverage only) do not. We then test how far it carries: moved goals, continued learning by
PPO (AI) and by local, backpropagation-free rules (bio-plausible), forgetting, generalisation, spiking, and a
brain-like visual front end. Figure: `fig_camera_story` (page 1).

## Questions (numbered) → answer → evidence
| # | Who | Question | Answer | Evidence |
|---|---|---|---|---|
| Q1 | Chairs, FtUu, 6g1M, GQMP | Is the transfer only frozen, same-goal? Add a moved-goal test | Claim scoped to frozen, familiar goal AND tested with 3 registered moved goals per fresh room | Table 1, story c, showcase d |
| Q2 | GQMP | Is the "decision content" just a remembered goal location? | Moved goals + old-goal visits | diag c, showcase d |
| Q3 | FtUu, 6g1M | Does the human prior help when the agent keeps learning a new goal? | 30-seed moved-goal learning with PPO and local rules + searched configs | story d, Fig. learning |
| Q4 | UYhB | Specific to PPO? | Same priors with PPO vs TD-readout, TD-DFA, ED routing (Sakana), NAS+Optuna configs | Figs. learning, kpi |
| Q5 | FtUu | What map does the pose filter use? Does layout leak? | Open-arena template atlas; no wall map, goal or pose | Sec. 2 |
| Q6 | FtUu | Belief accuracy in generated rooms? | Top-1 100% clean | diag d |
| Q7 | FtUu | Dimming: does the filter break? | Yes: top-1 ~45%; all arms 13–21%; augmented camera-only agents recover | diag d, Fig. ladder |
| Q8 | FtUu | Rat null from unpredictable labels? | Rat choices learnable (56% vs 32% legal) but half the predictable information of human choices; claim limited to these inputs | diag e |
| Q9 | FtUu | Sampled or greedy? | Sampled (registered); greedy lowers every arm (Lean 4 proof: a repeated state of a deterministic memoryless system never reaches the goal) | Sec. 2, supp, `proofs/Proofs/Loops.lean` |
| Q10 | FtUu | Loop/revisit rates; is the gain just less looping? | ~95% of failures are loops; human choices −8.2 pp revisits, −8.7 pp trapped; 1.7 pp remains at equal trapping. Content = the humans' habit of not going back | story a–b, diag a–b |
| Q11 | FtUu | Why does pixel noise help every arm? | Noise breaks loops (trapped 55→46% human, 64→58% legal) | Sec. 3.2 |
| Q12 | FtUu | Simple teachers recover part of the gap | Human choices still beat shortest-path, entropy-matched optimal and goal-bias teachers | Fig. controls |
| Q13 | UYhB | Rat vs human agents compared directly? | Human − rat 13.7 pp; with legal labels 4.7 pp → ⅓ coverage, ⅔ choices | Table 1 |
| Q14 | UYhB | Arbitrary comparisons / Holm | One table of all main contrasts with CIs; Holm only within registered families | Table 1 |
| Q15 | UYhB | Take-home? "Better data"? | Cleaner data (shortest path, optimal) loses to human choices; how humans move is what helps | story, Fig. controls |
| Q16 | UYhB, GQMP | Neural results out of nowhere | Asked inside the story: does human-choice content survive a predicted-mouse-cortex front end? 10 seeds: same direction, unresolved; 30-seed extension registered, deferred | Sec. 3.4, supp |
| Q17 | GQMP | Weak absolute performance, small FF policy | Stated; animal reference kept; memorylessness explains loops | limitations |
| Q18 | GQMP | ±10 pp margin equals the effect | Equivalence language dropped; CIs reported | Table 1 |
| Q19 | GQMP | p-values overstate; seeds ≠ animals | CIs; seed caveat kept | Sec. 2 |
| Q20 | GQMP | Post-hoc follow-ups; prereg unverifiable | Each marked; camera-ready studies hash-bound (`data/audit/v4n_*_20261031`) | Sec. 2, supp |
| Q21 | 6g1M | E3 aliasing / joint reference | E3 in supplement with this limitation | supp |
| Q22 | 6g1M, GQMP | Unequal text/image information; Rosetta | Supplement | supp |
| Q23 | GQMP | Thin visual/neural content | Showcase (views + belief), occupancy maps, real-demonstrator story, cortex front end | Figs. story, showcase, paths |
| Q24 | GQMP, UYhB | Dense, mixed units, fragmented | ~1,900 words; pp only; one story | throughout |
| Q25 | Chairs | Retitle | "Same States, Different Choices: Human Decisions Improve Same-Goal Maze Transfer and Shape Forgetting and Vision–Language–Action Geometry" (author 2026-11-05: keep OOD, forgetting and VLA geometry, each bounded) | title |
| Q26 | Chairs | Move replay, cortex, alignment | Done | supp Sec. 1 |
| Q27 | author | Forgetting, OOD, continual, spiking claimed not shown | Measured on the same agents after moved-goal learning; spiking by IF conversion, op counts | Fig. kpi |
| Q28 | author (6g1M) | Do caption words leak the action? | Lexical control (AMENDMENT3): captions without any action word; text->action and composition unchanged (<=1 pp), closed loop still chance; text-action CKA halves, so linkage rests on prediction, not CKA | supp matched section |
| Q29 | author (UYhB) | Is local learning real or just more experience? | Shuffled-reward TD-readout (registered P3): 23-39% -> 9-17%, all p<1e-4 | supp decision-time section |
| Q30 | author (Q27) | Spiking efficiency beyond fixed T; link to crow/primate accumulate-to-bound (Schmidbauer & Nieder 2026) | P1 supported: human choices decide 0.48 steps sooner [0.23, 0.73], p=.0004; at theta=.9 2.6 vs 3.9 vs 5.3 steps, 37% fewer ops than legal. P2 (success) unresolved +3.7 [-1.3, 8.7] | Fig. kpi d (open markers), supp |

## Strengths the reviewers praised, kept
| Strength (reviewer) | Where |
|---|---|
| Legal-uniform state-matched control (all) | Contribution 1, Fig. pipeline, discussion |
| Fresh-room confirmation (6g1M, GQMP) | Table 1, Fig. controls |
| Camera-only refits (6g1M, GQMP) | Table 1, Fig. controls |
| Shortest-path/entropy-matched/behavioural teachers (6g1M, FtUu, GQMP) | Fig. controls |
| Seeds ≠ biological replication (6g1M, GQMP) | Sec. 2 |
| Species × apparatus confound acknowledged (6g1M, GQMP) | Sec. 3.1, limitations |
| Untrained representation controls, closed-loop tests (6g1M) | Supplement + pointer |
| Paired seeds, candid limitations (GQMP) | Sec. 2, limitations |

## Additions on author request
Lean 4 proof (Q9); bio-plausible learners incl. Sakana ED routing + NAS/Optuna (Q3–Q4); four claims (Q27); v1 local
learners failed dev calibration, replaced before test (AMENDMENT3). Deferred: 30-seed cortex extension (registered,
not run).

## Author block
Affiliation: Independent Researcher (author, 2026-10-31).

## Rosetta reproduction check (2026-10-31)
Upstream `dominik-schnaus/unpaired-rosetta` @ fdfad84 (no newer commits); pinned `pylibmgm==1.1.2` installed; their
tests `test_method.py`, `test_geometry.py`, `test_evaluation.py`: 46 passed (`data/audit/v4n_rosetta_matched_20261031/upstream_tests.log`).
Their protocol is applied unmodified to matched image/text/action data (stage E of `scripts/v4n_rosetta_matched_claude_20261031.py`).

## Rosetta authors' protocol on the matched data (stage E, finished 2026-10-31T23:37Z)
Upstream code unmodified (CPU-only solver). Geometry: CKA image-text 0.32, text-action 0.28-0.41, image-action 0.14-0.23 vs <=0.001 shuffled; mutual kNN small (0.10; 0.016-0.026 vs 0.007-0.022 shuffled).
Unpaired Wasserstein-Procrustes FOSCTTM (0.5 chance): image-text 0.57, image-action 0.50-0.53, text-action 0.34-0.42 (SD up to 0.17); 16 pairs: 0.30 / 0.39-0.43 / 0.26-0.32.
Reading: shared average geometry, but without pairs it does not recover which view matches which action. Agrees with the closed-loop result. Supplement updated (24 pages).

## Citation fix (2026-11-01)
The link the author supplied (10.1126/science.aem0341) is the Perspective by Colombo & Poutrain-Sorel, "The cognitive power of the avian brain", Science 394:385. The research article is Schmidbauer & Nieder, Science 394(6822):445-449, doi 10.1126/science.aed6459 (Crossref). Bib corrected; zhou2026embodied checked against arXiv metadata.

## Second review pass (2026-11-01)
- Q21 (6g1M aliasing / joint-training reference) now answered with evidence: `scripts/v4n_joint_reference_claude_20261101.py`, `data/audit/v4n_joint_reference_20261101` (registered 07:02Z; AMENDMENT1 bookkeeping fix before outcomes). Aliasing 15.2% of cells. J1 supported: joint retains old goal +60.0 pp PPO [54.0,65.9], +30.0 TD-readout [24.0,35.9]. J2 NOT supported (predicted no gain): summed success +45.0 [35.0,55.0] / +15.2 [8.6,21.8]. Forgetting = sequential interference. Main: kpi figure diamonds + one sentence; supp Sec. joint.
- GQMP: equivalence band removed from controls figure; Q&A items added for cortex resolution confound and E3 GPU confound.
- Abstract now states learning (PPO + local rules: prior never speeds learning) and spiking (advantage kept, bound reached sooner).
- Main Sec. 3.4 rewritten as "Is the content in the view or in the choice?": cortex + Rosetta linked to the loop/not-going-back conclusion; kept short per the Chairs.
- Rosetta stage E rerun on action-word-free captions (AMENDMENT5) in progress.

## Rosetta stage E on action-word-free captions (2026-11-01T10:01Z, AMENDMENT5)
- Determinism: 20/20 image-action jobs identical to the original stage E.
- Text-action CKA 0.16-0.31 (orig 0.28-0.41); unpaired FOSCTTM 0.47-0.55 (orig 0.34-0.42; <0.5 in 10/20 vs 18/20) -> earlier sub-chance matching was lexical overlap.
- Image-text CKA 0.23 (0.32), FOSCTTM0 0.42 (0.57); 16 pairs: 0.32 / text-action 0.34-0.42. mkNN unchanged.
- Paper: supplement matched section paragraph added; main Sec. 3.4 FOSCTTM now 0.47-0.55 (views or action-free captions). Main 9 pp (References on p8), supp 24 pp, 0 errors/undefined.

## Bio-plausible learners: stability, gating, brain-constrained circuit, spiking (2026-11-03, results at ~00:20 UTC)
- Stability search (`scripts/v4n_local_stability_claude_20261102.py`, `data/audit/v4n_local_stability_20261102/analysis.json`): cs_dfa vs PPO new +17.1 [9.3,24.9], OOD new +18.0 [12.6,23.5], old +2.5 [-2.7,7.8]; vs s2_dfa old +23.4, new -22.0. Primary prior AUC contrasts all Holm p>=.17 (estimates negative) — disclosed.
- Context gating (`v4n_context_gating_20261103/analysis.json`): old goal G0 4.7%, G1 57.5% (+52.8, Holm 1e-33), G2 inferred 42.7% (+38.0, Holm 3e-29); new goal unchanged (G2-G0 -0.04 [-0.36,0.29]).
- Brain-constrained circuit (`v4n_brainlike_20261103/analysis.json`, descriptive): ungated old 55% (vs PPO +31.0 [24.5,37.4]), new 15% (-45.5); inferred gate lowers both.
- Learned-network spiking (`v4n_snn_learned_20261103/summary.json`): conversion lossless; s2 expansion 0.8-3.5M synops/decision vs 24-100k PPO.
- Paper: KPI figure redrawn (a old/trained, b new/unseen, c keep-vs-learn, d spiking + learned crosses); KPIresults rewritten; supplement Sec. bio + tradeoff figure; 9 refs added (2026: Neymotin RS preprint, Dong & He Front Neurosci, Zhang arXiv 2609.31630). Neuron 2026 dendritic gating not cited (could not verify). Main: refs start p8.
- Rosetta brains: schedule-only deviation — geometry scores computed first in one pass (`geom`), second WP pool walks the list in reverse (`jobs_rev`).
- Rosetta brains geometry (`brains/geometry.json`, 810 jobs, fig `fig_rosetta_brains`, script `scripts/figs/fig_rosetta_brains_claude.py`): PCA144 vs full cortex CKA 1.00, all cortex CKAs identical (UYhB answered; mkNN 0.36). Hidden–cortex CKA human_bc 0.49 vs rat/legal 0.40, scratch 0.44. Net–net vs human_bc: other seed 0.82–0.88, rat 0.58–0.71, legal 0.50–0.70, scratch 0.47–0.57, PPO-learned 0.65–0.80, s2 0.77–0.88. Text–cortex 0.51 vs image–cortex 0.19. WP retrieval jobs still running (two pools).

## Chair-boundary pass + latent-cause search (2026-11-04T21:15Z)
- Old-goal retention, human vs legal (paired, 30 seeds, post hoc): PPO +13.0 pp [1.1, 24.9]; TD-readout +13.9 [5.5, 22.2]; consolidation (cs_dfa) +10.0 [4.0, 16.0]. Claim kept with CIs, marked post hoc.
- Main text: gate reported as a CI (not Holm p=1e-29, GQMP Q19); Sec. 3.4 rewritten to the title's story (a view holds the move, not the route) with new Fig. view_route (scripts/figs/fig_view_route_claude.py); mushroom-body detail left to the supplement.
- Float overrun found (Figs. 11-12 had floated past References; 11 pages with a FloatBarrier). Pipeline and occupancy-path figures moved to supplement Sec. movedfigs; all main figures now precede the Discussion; Discussion and References start on p8.
- Registered (PROTOCOL.json before any outcome) scripts/v4n_latent_search_claude_20261105.py: NSGA-II over the G2 latent-cause rule only (est, emin, drop, bound, dwell, ref), dev rooms 3 seeds x 3 priors, 160 trials, trial 0 = registered rule (dev 100 / 50). Test once on 30 x 3 fresh rooms; Holm over 4.

### Latent-cause search result (2026-11-05T00:23Z)
- 160 NSGA-II trials (dev). Best = trial 109 (dev 100 / 54.4 vs registered 100 / 50.0).
- Single test (FR.S30 x 3 priors, fresh rooms): G2s-G2 old goal -2.7 pp [-6.6, 1.1], Holm p=0.19; new goal -0.2 pp. Not adopted.
- Paper keeps \GTold=43. Supplement paragraph added ("Is the inference gap a tuning artefact?"). Files: data/audit/v4n_latent_search_20261105/{BEST,analysis}.json.

### Brain-constrained circuit stage 2 (2026-11-05T03:25Z)
- Registered search (scripts/v4n_brainlike2_claude_20261105.py; awake replay E x T + optional adaptive dopamine gain), 96 trials; stage-1 winners reproduced exactly as trials 0-1.
- Test (30 seeds x 3 priors, fresh rooms): ungated new 60.1 / old 24.6 (stage 1: 15.3 / 55.1). vs PPO: new -0.6 [-10.3, 9.0] TOST p=0.028; old +0.5 [-6.4, 7.4] TOST p=0.004 -> equivalent within +/-10 pp. Gated: old +13.3 [6.9, 19.7] vs PPO, new -7.3 [-16.9, 2.2].
- Unseen room new goal trails PPO (-7.9 [-13.8, -2.1]). Replay costs retention vs stage 1 (-30.5).
- Paper: Fig. kpi now shows stage 2 (both gate settings); one sentence in \KPIresults; supplement paragraph "Stage 2". Files data/audit/v4n_brainlike2_20261105/{BEST,analysis,analysis_ppo}.json.

### Rosetta retrieval jobs complete (2026-11-05T04:35Z)
- 882/882 jobs; summary data/audit/v4n_rosetta_matched_20261031/brains/foscttm_summary.json.
- Unpaired modality retrieval at chance (0.43-0.67); 16 pairs -> 0.12-0.47. Bridge composed no better than direct.
- Unpaired network-network: human_bc seeds 0.06; vs legal 0.31; vs rat 0.59; vs scratch 0.50; prior vs PPO-learned 0.20, vs local 0.32.
- Supplement sec:rbrains "still running" replaced; one clause added to main Sec. view-route. Main 9 pp (refs start p8), supp 32 pp.

### Brain-constrained circuit stage 3 + Lean trace bound (2026-11-05T08:43Z)
- Lean: proofs/Proofs/Trace.lean (trace_bounded, core Lean, lake build ok). Search samples rho = eta_v/(1-gamma lambda).
- Registered search (scripts/v4n_brainlike3_claude_20261105.py), 120 trials, workers on GPU 1 and GPU 0. Old-goal replay (O>0) rejected on dev.
- Test (GPU 1 only, 30 seeds x 3 priors): ungated new 73.2 / old 27.9. vs PPO new +12.4 [3.8, 21.1] p=.006; old +3.7 [-2.2, 9.7] TOST p=.019; unseen new -7.4 [-13.4, -1.5]. vs stage 2 new +13.1 [4.5, 21.6]. Gated = stage-2 rule, reproduced exactly.
- Paper: abstract/\KPIresults "learns the new goal better than PPO (+12.4 pp)"; Fig. kpi brain circuit = stage 3; supplement stage-3 paragraph and Lean item (x); README updated. Main 9 pp (refs p8), supp 33 pp.

### Brain-constrained circuit stage 4: lateral horn rejected (2026-11-06T03:54Z)
- Registered before outcomes (scripts/v4n_brainlike4_claude_20261106.py, PROTOCOL 2026-11-06 ~01:10Z). Dense lateral-horn copy of h (gain w) + unseen-room new goal as 3rd dev objective; O = 0. check: w = 0 reproduces stage-3 trial 97 exactly.
- Search: 120 trials, both workers GPU 1. Both dev winners have w = 0, so the mechanism was rejected. Gated winner = stage-3 rule; its test reproduces stage 3 exactly (90/90).
- Test (GPU 1, 30 seeds x 3 priors, 0/180 diverged), ungated (trial 90): new 75.4 / old 22.7 / unseen new 22.9 / unseen familiar 13.6. vs PPO new +14.6 [5.9, 23.4] p=.001; old -1.4 [-7.6, 4.7] TOST p=.004; unseen new -5.0 [-10.9, 0.9] TOST p=.047; unseen familiar -7.2 [-11.5, -2.9]. vs stage 3: unseen familiar -4.9 [-8.7, -1.1]; new-goal probes equivalent.
- Paper: supplement stage-4 paragraph only. Main text keeps stage 3, since a fourth tested selection is not a new headline. Main 9 pp unchanged, supp 33 pp.

### Brain-constrained circuit stage 5: Lean-motivated place drive rejected (2026-11-06T21:14Z)
- Lean 4 (proofs_ml/ProofsML/Transfer.lean, standard axioms): transfer_identity (readout at B moves by eta*delta*g*<z_A,z_B>), self_interference, drive_diff, drive_inner. Motivation: Kenyon-code cross-room overlap .71 vs PPO .95; place belief room-invariant (cos 1.00).
- Registered before outcomes (scripts/v4n_brainlike5_claude_20261106.py, PROTOCOL 2026-11-06T18:07:35Z). Place belief -> each Kenyon cell via cp random synapses, gain beta in {0,.5,1,2,4}, cp in {3,7,15}. ADDENDUM1: probe-shape crash on every beta>0 trial at first launch, fixed before any beta>0 outcome; relaunch 18:14Z.
- Search: 120 trials, GPU 1. Ungated dev winner beta = 0 (trial 25), gated winner beta = 1 (trial 57).
- Test (GPU 1, 30 seeds x 3 priors, 0/180 diverged). Ungated: new 68.9 / old 25.3 / unseen new 23.1; vs PPO new +8.1 [-0.6, 16.7], unseen new -4.7 [-10.8, 1.4] TOST p=.044; vs stage 3 unseen new +2.7 [-2.0, 7.5]. Gated (beta 1): new 66.9 / old 36.4 / unseen new 11.1; vs PPO old +12.3 [6.1, 18.4], unseen new -16.7 [-22.3, -11.2].
- Paper: supplement stage-5 paragraph (registered null). Main text keeps stage 3; main 9 pp, supp 33 pp.
