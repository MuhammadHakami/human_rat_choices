/-
  Vision4Neuro camera-ready (2026-10-31): why a greedy memoryless policy gets trapped (reviewer FtUu).

  Model. The complete simulator state (cell, heading, last action, contact, reward, Bayes-filter posterior, texture)
  evolves deterministically once the policy is deterministic (greedy argmax of a frozen network): s_{t+1} = f s_t.
  The type of states is arbitrary (the posterior is real-valued), so no finiteness is assumed.

  Theorem `trapped_forever`: if the full state ever repeats (state at step i+p equals state at step i, p > 0) and
  none of the p states of that cycle is a goal state, then no later step reaches the goal. Sampled actions break the
  premise (f is no longer a function of the state), which is why the registered probes sample.

  Core Lean only (no Mathlib); `lake build` checks it.
-/

namespace V4N

/-- n deterministic steps from s. -/
def run {α : Type} (f : α → α) : Nat → α → α
  | 0, s => s
  | n + 1, s => f (run f n s)

theorem run_add {α : Type} (f : α → α) (s : α) (a b : Nat) : run f (a + b) s = run f b (run f a s) := by
  induction b with
  | zero => rfl
  | succ b ih => show f (run f (a + b) s) = f (run f b (run f a s)); rw [ih]

/-- After a repeat, every later state is one of the p states of the cycle. -/
theorem stays_in_cycle {α : Type} (f : α → α) (s : α) (i p : Nat) (hp : 0 < p)
    (h : run f (i + p) s = run f i s) : ∀ n, ∃ k, k < p ∧ run f (i + n) s = run f (i + k) s := by
  intro n
  induction n with
  | zero => exact ⟨0, hp, rfl⟩
  | succ n ih =>
    obtain ⟨k, hk, he⟩ := ih
    have step : run f (i + (n + 1)) s = f (run f (i + k) s) := by
      show f (run f (i + n) s) = f (run f (i + k) s); rw [he]
    by_cases hlt : k + 1 < p
    · exact ⟨k + 1, hlt, step⟩
    · have hkp : k + 1 = p := by omega
      refine ⟨0, hp, ?_⟩
      rw [step]
      show run f (i + (k + 1)) s = run f (i + 0) s
      rw [hkp, Nat.add_zero, h]

/-- Trapped forever: a repeated full state with no goal in its cycle means the goal is never reached later. -/
theorem trapped_forever {α : Type} (f : α → α) (goal : α → Prop) (s : α) (i p : Nat) (hp : 0 < p)
    (h : run f (i + p) s = run f i s) (hnot : ∀ k, k < p → ¬ goal (run f (i + k) s)) :
    ∀ n, ¬ goal (run f (i + n) s) := by
  intro n
  obtain ⟨k, hk, he⟩ := stays_in_cycle f s i p hp h n
  rw [he]; exact hnot k hk

/-- Sanity instance: a two-state flip-flop that never visits state 2 never reaches it. -/
def flip : Nat → Nat := fun x => if x = 0 then 1 else 0

example : ∀ n, ¬ (run flip (0 + n) 0 = 2) :=
  trapped_forever flip (· = 2) 0 0 2 (by decide) (by decide)
    (by intro k hk; match k, hk with | 0, _ => decide | 1, _ => decide)

end V4N
