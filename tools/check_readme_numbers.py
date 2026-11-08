"""Every number in README.md must be the registered record, and the same number must be in the paper.
  python tools/check_readme_numbers.py        (CPU, seconds; exit 1 on any mismatch)
Each line below names a record, derives the number from it, and checks that the formatted number appears in README.md
and in the paper sources (main text, supplement, camera_numbers.tex)."""
import json, re, sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

R = Path(__file__).resolve().parent.parent
J = lambda p: json.loads((R / p).read_text())
README = (R / "README.md").read_text().replace("−", "-")
PAPER = "".join((R / "paper/wacv2027" / f).read_text() for f in ("vision4neuro_camera.tex", "vision4neuro_camera_supp.tex", "camera_numbers.tex"))
PAPER = PAPER.replace("\\%", "%").replace("$", "").replace("\\textbf{", "").replace("\\emph{", "").replace("−", "-")


def f(x, nd=1):  # round half up, as the paper does (not banker's rounding)
    q = Decimal(1).scaleb(-nd) if nd else Decimal(1)
    s = str(Decimal(str(x)).quantize(q, rounding=ROUND_HALF_UP))
    return s.lstrip("0") if s.startswith("0.") else s  # p-values and CKA appear as .02 / 0.12 -> check both below


fresh = J("data/audit/wacv_fresh_rooms_confirm_claude/analysis.json")
gate = J("data/audit/v4n_context_gating_20261103/analysis.json")["means"]
bl3 = J("data/audit/v4n_brainlike3_20261105/analysis.json")["none"]
bl7 = J("data/audit/v4n_brainlike7_20261107/analysis.json")["none"]
bl8 = J("data/audit/v4n_brainlike8_20261108/analysis.json")["none"]
shared = J("data/audit/v4n_shared_mazes_20261107/analysis.json")
brain = J("data/audit/v4n_brain_mazes_20261107/analysis.json")["families"]
site = J("site/data/results.json")
macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", (R / "paper/wacv2027/camera_numbers.tex").read_text()))
pp = lambda c: 100 * c / 30  # fresh-room contrasts are successes out of 30 trials
ppo_unseen_new = sum(site["kpi"][f"ppo/{p}/moved_heldout_room"][0] for p in ("human_bc", "human_legal_uniform", "cold_ppo")) / 3

CHECKS = [  # (label, formatted value, in paper too?)
    ("human − legal, fresh rooms (pp)", f(pp(fresh["primary"]["C1"]["mean"])), True),
    ("… CI low", f(pp(fresh["primary"]["C1"]["ci95"][0])), True),
    ("… CI high", f(pp(fresh["primary"]["C1"]["ci95"][1])), True),
    ("fewer revisits (pp)", macros["HLrevisit"], True),
    ("moved goal, human − legal (pp)", macros["PMvLegal"], True),
    ("rat − legal (pp)", f(pp(fresh["secondary"]["C3"]["mean"])), True),
    ("… CI low", f(pp(fresh["secondary"]["C3"]["ci95"][0])), True),
    ("… CI high", f(pp(fresh["secondary"]["C3"]["ci95"][1])), True),
    ("gate keeps old goal (%)", f(gate["G2"]["familiar_trained_room"], 0), True),
    ("no gate (%)", f(gate["G0"]["familiar_trained_room"]), True),
    ("circuit − PPO, new goal (pp)", f(bl3["moved_trained_room/ppo"]["mean"]), True),
    ("… CI low", f(bl3["moved_trained_room/ppo"]["ci95"][0]), True),
    ("… CI high", f(bl3["moved_trained_room/ppo"]["ci95"][1]), True),
    ("… old goal TOST p", f(bl3["familiar_trained_room/ppo"]["tost_p"], 2), False),
    ("goal map, never-seen room, new goal (%)", f(bl7["means"]["moved_heldout_room"], 0), True),
    ("PPO there (%)", f(ppo_unseen_new, 0), True),
    ("goal map keeps old goal (%)", f(bl7["means"]["familiar_trained_room"], 0), True),
    ("stage 7 old goal, never-seen room (%)", f(bl7["means"]["familiar_heldout_room"], 0), True),
    ("stage 8 old goal, never-seen room (%)", f(bl8["means"]["familiar_heldout_room"], 0), True),
    ("stage 8 − stage 7 (pp)", f(bl8["familiar_heldout_room/bl7/all"]["mean"]), True),
    ("… CI low", f(bl8["familiar_heldout_room/bl7/all"]["ci95"][0]), True),
    ("… CI high", f(bl8["familiar_heldout_room/bl7/all"]["ci95"][1]), True),
    ("stage 8 new goal, never-seen room (%)", f(bl8["means"]["moved_heldout_room"], 0), False),
    ("humans, own layouts (%)", f(shared["animals"]["human"]["success"][0], 0), True),
    ("rats (%)", f(shared["animals"]["rat"]["success"][0], 0), True),
    ("frozen rat-trained (%)", f(shared["agents"]["frozen/rat_bc"]["familiar"][0], 0), True),
    ("frozen human-trained (%)", f(shared["agents"]["frozen/human_bc"]["familiar"][0], 0), True),
    ("goal map, old goal (%)", f(shared["agents"]["bl7/human_bc"]["familiar"][0], 0), True),
    ("goal map, new goal (%)", f(shared["agents"]["bl7/human_bc"]["moved"][0], 0), True),
    ("human hippocampus ↔ human-trained hidden (ρ beyond space)", f(brain["human/all/frozen/human_bc/hidden"]["partial"][0], 2), True),
    ("… ↔ view", f(site["brain_rosetta"]["towns_human"]["fam"]["frozen/human_bc"]["partial"][0][4], 2), True),
    ("seeds", str(len(J("data/wacv/seeds30/preregistration.json")["seeds"])), True),
    ("fresh rooms", str(len(J("site/data/env.json")["rooms"])), True),
]

bad = 0
for label, val, in_paper in CHECKS:
    alts = {val, "0" + val if val.startswith(".") else val}
    ok_r = any(a in README for a in alts)
    ok_p = (not in_paper) or any(a in PAPER for a in alts)
    bad += (not ok_r) + (not ok_p)
    print(f"{'ok ' if ok_r and ok_p else 'MISSING'} {val:>6}  {label}" + ("" if ok_r else "  [not in README]") + ("" if ok_p else "  [not in paper]"))
print(f"\n{len(CHECKS) - bad}/{len(CHECKS)} numbers in README match the records" + ("" if not bad else f"; {bad} MISSING"))
sys.exit(1 if bad else 0)
