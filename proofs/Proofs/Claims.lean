/-
  Vision4Neuro camera-ready (2026-10-31): machine-checked logic and arithmetic behind the paper's claims.
  What Lean can check: definitions, identities and reported arithmetic. What it cannot: the empirical effects
  (they are statistical estimates). Core Lean only; numbers are in tenths of a percentage point (pp) where needed.
-/

namespace V4N

/-! ## 1. The legal-uniform control keeps the states (coverage) and changes only the labels -/

/-- A BC dataset: (state, action) rows. Relabelling gives row i a new label labs[i] at the SAME state (legal-uniform
labels are drawn per row, so two visits to one state may get different labels). -/
def relabel {S A : Type} : List (S × A) → List A → List (S × A)
  | r :: rs, a :: as => (r.1, a) :: relabel rs as
  | _, _ => []

theorem relabel_keeps_states {S A : Type} (rows : List (S × A)) (labs : List A) (h : labs.length = rows.length) :
    (relabel rows labs).map Prod.fst = rows.map Prod.fst := by
  induction rows generalizing labs with
  | nil => cases labs <;> rfl
  | cons r rs ih => cases labs with
    | nil => simp at h
    | cons a as => simp [relabel, ih as (by simpa using h)]

theorem relabel_keeps_size {S A : Type} (rows : List (S × A)) (labs : List A) (h : labs.length = rows.length) :
    (relabel rows labs).length = rows.length := by
  have := congrArg List.length (relabel_keeps_states rows labs h); simpa using this

/-! ## 2. The paired contrast: sum of per-seed differences = difference of sums (so mean diff = diff of means) -/

def sumZ : List Int → Int
  | [] => 0
  | x :: xs => x + sumZ xs

theorem paired_sum (xs ys : List Int) (h : xs.length = ys.length) :
    sumZ (List.zipWith (· - ·) xs ys) = sumZ xs - sumZ ys := by
  induction xs generalizing ys with
  | nil => cases ys with
    | nil => rfl
    | cons _ _ => simp at h
  | cons x xs ih => cases ys with
    | nil => simp at h
    | cons y ys =>
      simp only [List.zipWith, sumZ]
      have := ih ys (by simpa using h); omega

/-! ## 3. Species gap = coverage part + choice part (exact identity), and the reported numbers -/

/-- H, Hl, R, Rl: success of human-choice, human-legal, rat-choice, rat-legal agents. -/
theorem species_gap_decomposition (H Hl R Rl : Int) : H - R = (Hl - Rl) + ((H - Hl) - (R - Rl)) := by omega

/-- Reported (tenths of pp): human−rat 13.7 = legal-label gap 4.7 + interaction 9.0. -/
example : (137 : Int) = 47 + 90 := by decide

/-- The coverage share is about a third: 3·47 = 141 lies within ±10% of 137 (pp tenths). -/
example : 3 * 47 ≤ 137 + 14 ∧ 137 - 14 ≤ 3 * 47 := by decide

/-! ## 4. Units: successes out of 30 → percentage points (×100/30) -/

/-- A gain of 3 successes per 30 trials is exactly 10 pp: 3·100 = 10·30. -/
example : 3 * 100 = 10 * 30 := by decide
/-- One success per 30 trials is 10/3 pp, i.e. 3.33 pp: 100 = 30 · (10/3). -/
example : 1 * 100 * 3 = 10 * 30 := by decide

/-! ## 5. Operation counts used in the spiking comparison (counts, not energy) -/

def denseMACs (inp h1 h2 out : Nat) : Nat := inp * h1 + h1 * h2 + h2 * out

/-- The 64×64 actor–critic with 252 inputs and 4 actions + 1 value: 20,544 multiply-accumulates per decision. -/
example : denseMACs 252 64 64 5 = 20544 := by decide
/-- Hidden-layer part (what the spiking network replaces with spike-driven accumulates): 4,416. -/
example : 64 * 64 + 64 * 5 = 4416 := by decide

/-- Spike-driven synaptic operations: each hidden-1 spike fans out to 64 units, each hidden-2 spike to 5 outputs. -/
def synOps (spikes1 spikes2 : Nat) : Nat := spikes1 * 64 + spikes2 * 5

/-- Fewer spikes ⇒ fewer operations: the count is monotone in both spike totals. -/
theorem synOps_mono (a b c d : Nat) (h1 : a ≤ c) (h2 : b ≤ d) : synOps a b ≤ synOps c d := by
  unfold synOps; have := Nat.mul_le_mul_right 64 h1; have := Nat.mul_le_mul_right 5 h2; omega

end V4N
