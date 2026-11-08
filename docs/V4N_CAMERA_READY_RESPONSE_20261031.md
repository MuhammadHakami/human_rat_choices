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
| Q16 | UYhB, GQMP | Neural results out of nowhere | Asked inside the story: does human-choice content survive a predicted-mouse-cortex front end? 10 seeds: same direction, unresolved; 30-seed extension registered, deferred | Sec. 3.4, supp; recorded-brain follow-ups (rat CA1/PFC/MEC, human hippocampal/MTL iEEG on the same replayed states): supp "same networks in the animals' own mazes" |
| Q17 | GQMP | Weak absolute performance, small FF policy | Stated; animal reference kept; memorylessness explains loops | limitations |
| Q18 | GQMP | ±10 pp margin equals the effect | Equivalence language dropped; CIs reported | Table 1 |
| Q19 | GQMP | p-values overstate; seeds ≠ animals | CIs; seed caveat kept | Sec. 2 |
| Q20 | GQMP | Post-hoc follow-ups; prereg unverifiable | Each marked; camera-ready studies hash-bound (`data/audit/v4n_*_20261031`) | Sec. 2, supp |
| Q21 | 6g1M | E3 aliasing / joint reference | E3 in supplement with this limitation | supp |
| Q22 | 6g1M, GQMP | Unequal text/image information; Rosetta | Supplement; Rosetta modalities (image, caption, action, hidden) also compared with recorded brains on the same states | supp; project page "Recorded brains on the same states" |
| Q23 | GQMP | Thin visual/neural content | Showcase (views + belief), occupancy maps, real-demonstrator story, cortex front end; recorded rat and human brain geometry in the animals' own mazes | Figs. story, showcase, paths; supp |
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

### Brain-constrained circuit stage 6: context-gated goal-vector pathway (2026-11-07T00:35Z)
- Lean 4 (Transfer.lean): concat_inner (joined-code overlap = <z_A,z_B> + g^2 <b_A,b_B>), copies_orthogonal (per-context copies of the place belief are orthogonal). Research basis: reports/Bio plausible OOD navigation transfer.md (central-complex vector memory, ranked method 1).
- Registered before outcomes (scripts/v4n_brainlike6_claude_20261107.py, PROTOCOL see PROTOCOL.json; smoke run g=.5 disclosed). Readout sees [z; g*b], g in {0,.1,.25,.5,1,2}, plus stage-3 space. Pilot: g=0 trials reproduce stage-3 dev values exactly.
- Search: 120 trials, GPU 1, no errors. Dev winners: ungated g=1 (trial 92), gated g=2 (trial 119).
- Test (GPU 1, 30 seeds x 3 priors, 0/180 diverged). Gated: new 83.7 / old 57.7 / unseen new 7.5 / unseen familiar 25.3; vs PPO new +22.9 [15.0, 30.8], old +33.6 [26.1, 41.0], unseen new -20.3 [-25.7, -14.9]; vs gated stage 3 new +30.3, old +20.3, unseen new -5.1 [-9.0, -1.3]. Ungated: new 80.9 / old 17.9 / unseen new 12.6; vs PPO new +20.1 [10.2, 30.0], unseen new -15.2 [-20.5, -9.8].
- Reading: the place pathway gives continual learning in the trained room (learn new, keep old) but hurts transfer to unseen rooms; PPO remains best there. Supplement stage-6 paragraph added; main text unchanged (supp 34 pp).

### Brain-constrained circuit stage 7: hippocampal goal map transfers (2026-11-07T12:57Z)
- Diagnosis: own place belief peaks on the true cell at 100% of steps in all fresh and dev rooms; stages 1-6 store goals as view->action habits. Supplied-memory planner ceilings on the fresh rooms (no learner) disclosed in the protocol.
- Lean 4 (proofs_ml/ProofsML/Wavefront.lean, standard axioms): wave_ascent, greedy_reaches_source.
- Registered 2026-11-07T11:04:13Z (scripts/v4n_brainlike7_claude_20261107.py). Mechanism: one-shot dopamine-gated Hebbian goal write on place cells (max rule), wavefront value, bump short-term depression, inhibition of return (per episode), drive in steps x kappa added to the stage-6 logits. Pre-registration fixes on dev smoke runs: max write (additive saturated), step-scaled drive (ratio drive outvoted). ADDENDUM1: transient checkpoint read failure, worker 1 restarted, trial 43 failed.
- Check: kappa=0 reproduces stage-6 dev records exactly. Search 79 trials; winners ungated kappa=50 (trial 13), gated kappa=10 (trial 66).
- Test (GPU 1, 30 seeds x 3 priors, 0/270 diverged). Ungated: new 92.2 / old 100.0 / unseen new 83.0 / unseen old 72.7; vs PPO +31.4 [22.9, 40.0], +75.9 [70.8, 81.0], +55.2 [47.4, 63.0], +51.9 [43.7, 60.2]. Gated: 92.0 / 94.7 / 71.5 / 70.3. Ablation (no STP, no IOR): 63.6 / 20.4 / 8.4 / 17.4 (unseen new -19.4 vs PPO).
- Caveat: the within-episode mechanisms are a memory PPO lacks; the ablation shows they are necessary. Supplement stage-7 paragraph; main text unchanged pending author decision.
- Scope (descriptive, added 2026-11-07T14:55Z after the test): transfer is to the same arena-frame goal place with new walls. Fresh pairs: familiar goal identical; moved goal identical in 2/8 pairs, one cell away in 6/8. Unseen-room new goal, same cell vs one cell away: goal map 72.7 / 86.2, PPO 31.9 / 26.6, stage 3 22.5 / 19.7, ablation 16.2 / 6.1. A goal far from any remembered place is outside the benchmark.
- Browser port (human_rat_choices, uncommitted): bl7 agents step-exact vs Python (100 episodes, logits <= 2e-5); browser vs recorded probes 0.0 pp trained, -0.9 ± 1.5 pp never-seen.
- Leakage audit (descriptive, 2026-11-07, site port, 15 agents, never-seen rooms): belief input (distal cue rows) identical across layouts (max diff 0); goal moved >=4 cells from every remembered goal: goal map 3.3%, stage 3 3.8%, PPO 4.2%; memory re-centred on that far goal: 99.6%; no memory: 10.4%. Search rooms and test rooms share no layouts; excluding dev seeds 1000-1002 from the test: unseen new 84.3 (all 83.0). Old goal "100%" in the trained room = the map holds both goals and inhibition of return checks the other after an empty visit (two candidate places, 45 moves).

### Shared mazes and brain geometry in the animals' own mazes (2026-11-08, descriptive, registered; paper claims unchanged)
- Tartarus layouts 6-25 (no network saw them; imitation used 1-5), own native starts, same frozen/learned networks, 30 seeds (scripts/v4n_shared_mazes_claude_20261107.py; ADDENDUM1 = animal tracking fix before agent metrics). Familiar / moved success: humans 97.4 (recorded), rats 85.0; frozen human_bc 56.7 / 2.9, rat_bc 55.4 / 3.5; ppo 10.4 / 21.4; bl3 16.3 / 16.7; bl7 66.5 / 69.9 (human_bc start). Path / shortest: humans 1.26, rats 1.67, human_bc 1.32, rat_bc 1.39, bl7 1.84. Occupancy BC to humans: human_bc 0.73, bl7 0.73, ppo 0.61 (human-rat 0.94).
- Brain geometry (scripts/v4n_brain_mazes_claude_20261107.py): rat W-track (DANDI 000978; 8 rats, CA1 + PFC units) and human towns (Zenodo 8058006; 11 patients, hippocampal iEEG), rebuilt in our arena; recorded trials replayed through each network (same states). Split-half ceilings: rat 0.89-0.91, human 0.55. Space RDM alone: rat 0.29, human 0.28. Raw model rho 0.02-0.23; partial (space, heading, pixels) <= 0.07 in rats (best CA1: rat_bc 0.071 ± 0.048), up to 0.117 ± 0.052 in humans (human_bc; bl3 0.107, bl7 0.105, rat_bc 0.034). Post hoc paired: human data human_bc - rat_bc +0.084 ± 0.048 (9/11 patients); rat data in the opposite direction, -0.015 to -0.027, CIs include 0. Next-move probability on the same states: rats bl7 .295, human_bc .289, rat_bc .288, ppo .214 (uniform .25); humans bl7 .335, human_bc .323, ppo .308.
- Open arenas (scripts/v4n_brain_arenas_claude_20261108.py, registered 2026-11-07T22:25Z; ADDENDUM1 = drop rat sessions with a constant neural RDM, after the first output was seen): human YC1 iEEG (OpenNeuro ds005522, 40 patients with MTL/hippocampal bipolar pairs) and rat open field (DANDI 000405, 5 rats, 115 sessions). Space RDM: rat 0.26, human 0.22; ceilings rat 0.36 (n=5), human 0.04. Partial RSA: rats human_bc 0.049, rat_bc 0.048, bl3 0.054, bl7 0.053, ppo 0.018; humans 0.00-0.04, no family difference. Next move: rats bl7 .312, rat_bc .286, human_bc .284, ppo .280 (corrected 2026-11-08: the earlier values .284/.282/.279 were from the pre-addendum run); humans bl7 .323, bl3 .306, human_bc .299, ppo .296.
- Rosetta modalities vs recorded brains (scripts/v4n_brain_rosetta_claude_20261108.py, registered 2026-11-08T06:52:19Z before this output; no new runs, reads the two collections; ADDENDUM1 = docstring condition count 13-36, not 19-56). Spearman rho between condition RDMs of image (DINOv2), caption (MPNet), action, hidden, brain; beyond space+heading. Towns: every hidden layer 0.04-0.12 (human_bc 0.121) vs view 0.011, caption -0.001. CA1: view 0.079 vs hidden 0.055-0.115. PFC: caption 0.095, exceeded only by PPO-learned 0.148. Arenas: view/caption 0.03-0.09 vs networks 0.02-0.05. Action never above hidden by >0.03 (-0.04 to 0.07). Caption-view 0.33-0.59; hidden shares more with action than view except bl7. Site: new interactive figure (setting x network x raw/beyond space). Supp paragraph added (35 pp).

### Reviewer/chair audit and number sync (2026-11-08)
- camera_numbers.tex regenerated from the records: byte-identical. Abstract numbers = OpenReview metadata (16/16).
- OpenReview metadata abstract was stale (stage-2 "matches PPO"); now matches the paper ("learns the new goal better than PPO (+12 pp)").
- Main hand-typed KPI numbers spot-checked vs data/audit/v4n_brainlike3_20261105/analysis.json (+12.4 [3.75, 21.14], TOST .019). Stage-7, shared-maze, brain-maze and arena supplement numbers checked against their analysis.json; one stale value set fixed in this file (arena rat next-move).
- Builds: main 9 pp (Discussion and References start p8; p9 references only), supp 35 pp, 0 undefined.
- Coverage: every Q1-Q30 row has an answer and a location; Q16/Q22/Q23 now also point to the recorded-brain evidence. Main-text claims unchanged; all additions are supplement + project page.

### Brain-constrained circuit stage 8: loops diagnosed and removed (2026-11-08T10:41Z)
- Author question: "why sometimes in new scenarios our biologically plausible system keeps repeating? ... root cause and resolving for biologically plausible way end to end"; "needs to be brain based throughout".
- Diagnosis (descriptive, page port, seeds 1000-1004 x 3 priors, trained + never-seen rooms, both goals; `data/audit/v4n_brainlike8_20261108/diagnosis/`, `human_rat_choices/site/test/loops.mjs`): memoryless agents (frozen, PPO, stage 3) loop by pushing into a wall their place-driven habit is confident about (habit-optimal 2-25% at trapped steps, p .74-.95, belief correct 100%, 8-14 of the last 16 moves are bumps (7.5 for the stage-3 circuit in its trained room with the familiar goal; up to 14.2); Lean trapped_forever). Stage-7 goal map: never-seen familiar 62% in these agents, 90% of failures trapped, and the DRIVE asks for the bump (chosen link already depressed 99%, D[a]=0, p_drive .78-1.00): a depressed link is valued as staying and the blurred field makes the agent's cell a local maximum of v next to a wall; stage 7's inhibition fired only above 70% of the field maximum (searched thr). Rarer: a field peak at a mis-estimated unreachable cell.
- Probe-time test (stage-7 weights): thr = 0 (dopamine dip at every visited place in proportion to its expectation) never-seen familiar 61.8 -> 100.0, never-seen moved 81.8 -> 84.4, trained unchanged, trapped 0%, bumps 14.7 -> 2.5. Collision-worth-zero 91.1; zero-field-behind-bump harmful (29.6); oracle visible walls 69.6 (not perception).
- Brain basis: dopamine pause at omitted reward scales with the violated expectation (Schultz, Dayan & Montague 1997; Hollerman & Schultz 1998); K(c)[c] = 1 so a visit removes the expectation exactly; inhibition of return on places (Posner & Cohen 1984). Stage-7 paragraph now cites O'Keefe & Nadel 1978, Hollup 2001, Dupret 2010, Ponulak & Hopfield 2013, Pfeiffer & Foster 2013, Bittner 2017, Tsodyks & Markram 1997 for each mechanism; the page glossary has one card per goal-map mechanism.
- Lean: `proofs_ml/ProofsML/Visited.lean` (visited_not_source, visited_has_better_neighbour), lake build 8661 jobs OK, no sorry (ADDENDUM1, added after registration).
- Registered 2026-11-08T10:12:41Z (`scripts/v4n_brainlike8_claude_20261108.py`, sha eb2c6d11...; CHECK: thr=.7 reproduces the stage-7 record cold_ppo seed 1000 exactly). Test GPU 1, 4 workers, 10:13-10:41Z, 180 runs, 0 diverged, all RTX 3090.
- Results (paired with stage 7, 30 seeds x 3 priors): ungated new 96.7 / old 100 / never-seen new 88.8 (+5.7 [0.4, 11.1]) / never-seen old 100 (+27.3 [21.4, 33.3]; excluding the 5 diagnosis seeds +25.7 [19.3, 32.2]). P1 met, P3 met, P2 exceeded in the favourable direction (+5.7 vs +-5). Gated: never-seen old +6.5 [3.1, 10.0], never-seen new -6.6 [-12.7, -0.4]; predictions not met; reported as a limitation.
- Disclosures: rule chosen after probe-time outcomes on seeds 1000-1004 (two test seeds); dry run of the registered analysis on 40 partial records per arm seen before completion (ADDENDUM2); nothing changed after registration.
- Paper: supplement stage-8 paragraph; main text unchanged. Page: "goal map+IOR" agent (stage 8) playable, in Keep-vs-learn / Forgetting, Play note on why memoryless agents repeat, glossary cards for the goal-map mechanisms, two new Lean cards.

### Thoroughness check on the stage-8 "100%" (2026-11-08, descriptive, after the test)
- Author: "how can the model reach 100% on unseen env even though the target goal are unseen ... make sure we have no leakage and really unseen cases".
- The goal is NOT unseen: the familiar goal is (3,3) in every room; the never-seen room's moved goal is 0-1 cells from the trained room's (registered mirror rule around target (6,6); pairs 5 and 7 identical, the other six one cell away). The layout is unseen: B differs from A in 30-32/100 cells (18 walls each, disjoint); every fresh room is >= 18 cells from every dev/prior-training room (wacv_native_prior_confirmation manifest) and >= 15 cells from every Tartarus layout 1-25; 16 distinct layouts.
- No leakage routes found: learning room always stage A (`M.room_of`), heldout probe hard-coded "B"; `probe_now` never calls the goal write; weights frozen in probes (JS replay with the saved weights reproduces every recorded probe count: 0.0 pp); belief starts uniform (-log 100) and reads only the 64 distal-cue features through an open-arena atlas; cue rows identical across rooms (max |A-B| = 0), floor rows differ (adjacent walls visible). No goal, map, room id or position in the 252 inputs.
- Far-goal controls (stage 8, 15 agents, never-seen rooms): goal >= 4 cells from every remembered place 5.3% +- 5.2 (stage 3 3.8, PPO 4.2); memory re-centred on it 100.0%; no memory 8.0%; moved goal as tested 89.1%. Stage 7 for comparison: 3.3 / 99.6 / 10.4 / 82.4.
- Not "optimal lowest steps": never-seen familiar 1.67x shortest (median 1.42; 17% exactly shortest; 3.0 bumps/episode); never-seen moved 4.10x (median 2.67; 5% exactly shortest; 4.7 bumps); trained moved 1.14x (71% exactly shortest). Longest shortest path 14 moves vs 45 budget: 100% = every trial finishes within ~3x the longest shortest path.
- Reading: stage 8 shows that a remembered place plus bump-repaired map plus inhibition at visited places finds the place through unknown walls within the budget; it says nothing about a goal the agent has never been rewarded at (5%, chance). The paper's scope paragraph (stage 7) already states this; the supplement stage-8 paragraph now repeats it with these numbers. Files: `data/audit/v4n_brainlike8_20261108/diagnosis/{leakage,efficiency}_after_test.txt`, `site/test/{leakage,efficiency}.mjs`.

## 2026-11-08 README, supplement rounding, bibliography

- 2026-11-08T13:17Z. Supplement, "the same networks in the animals' own mazes": the stage-7 old-goal figure read 67%; the registered record (data/audit/v4n_shared_mazes_20261107/analysis.json, bl7/human_bc familiar 66.45) rounds to 66%. Corrected to 66% (70% new goal unchanged). The page and README already showed 66. Bibliography: all 75 entries resolved against Crossref/arXiv or the publisher page; five entries gained exact pages/titles (de Cothi 2022 and Litwin-Kumar 2017 e-page suffixes, McClelland 1995 full title, Wan 2025 PMLR volume/pages, Akiba 2019 pages). No claim changed.
