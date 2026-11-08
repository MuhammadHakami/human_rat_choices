# Stage-8 diagnosis (descriptive, 2026-11-08, before the stage-8 test)

Run with the project page's step-exact JavaScript port of the paper's agents (`human_rat_choices/site/js/sim.js`; `site/test/loops.mjs`
is `loops_5seeds` there), seeds 1000-1004 x 3 priors, stage-7 weights as tested, 30-trial probe in each seed's trained and never-seen
fresh room, both goals. Outcomes seen before the stage-8 protocol was written; summarised in its design text.

| file | question |
|---|---|
| `loops_5seeds.txt` | failure types and, at trapped steps, habit vs goal-map drive, for goal map / mushroom body / PPO / frozen |
| `loops_goalmap.txt` (`loops_goalmap.mjs`) | goal map at trapped steps: belief spread, is the chosen link already depressed, is the remembered peak reachable |
| `oracle_visible_walls.txt` (`oracle_visible_walls.mjs`) | ceiling with the true walls in the visible directions pre-depressed (perception is not the cause) |
| `fixes_probe_time.txt` (`fixes_probe_time.mjs`) | probe-time test of three within-episode rules; thr = 0 (inhibition at every visited place) selected |

The scripts were run from the page repository against `site/js/sim.js`; their import path was made repository-relative afterwards (no other change), so they run from the public repository root.

## After the test (descriptive)
| file | question |
|---|---|
| `loops_stage8_after_test.txt` | P4: loop diagnostic on the stage-8 agents (`site/test/loops.mjs bl8,bl7`) |
| `leakage_after_test.txt` | leakage audit for stage 8 and stage 7 (`site/test/leakage.mjs bl8` / `bl7`): cue rows identical across rooms; goal far from every remembered place; memory re-centred on it; no memory |
| `efficiency_after_test.txt` | path efficiency (`site/test/efficiency.mjs`): decisions / shortest path, bumps, longest shortest path vs the 45-move budget |
