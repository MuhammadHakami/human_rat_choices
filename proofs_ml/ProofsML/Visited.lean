/-
  Vision4Neuro stage 8 (2026-11-08): inhibition of return at every visited place. Stage 7 inhibited the goal field only where it
  exceeded 70% of its maximum, so the blurred field around a remembered goal could leave the agent's own cell a local maximum of
  the wavefront value next to a wall, where "staying" (bumping) was the drive's best option. Stage 8 applies the dopamine dip at
  every visited place: G' x = G x · (1 − K x) with a place-field kernel K that equals 1 at the visited cell c.
  1. visited_not_source: after the dip the visited cell carries no expectation, G' c = 0.
  2. visited_has_better_neighbour: hence, while some remembered place is still reachable (v c > 0), wave_ascent applies at c:
     an open neighbour has strictly larger value, so the drive can never prefer staying at a cell it has already visited.
  Standard axioms only.
-/
import ProofsML.Wavefront

theorem visited_not_source {S : Type*} (G K : S → ℝ) (c : S) (hK : K c = 1) :
    G c * (1 - K c) = 0 := by
  rw [hK]; ring

theorem visited_has_better_neighbour {S : Type*} (v G' : S → ℝ) (N : S → Set S) (γ : ℝ) (hγ0 : 0 < γ) (hγ1 : γ < 1)
    (hG : ∀ x, 0 ≤ G' x) (hrec : ∀ x, G' x < v x → ∃ y ∈ N x, v x = γ * v y) (c : S) (hc : G' c = 0) (hv : 0 < v c) :
    ∃ y ∈ N c, v c < v y :=
  wave_ascent v G' N γ hγ0 hγ1 hG hrec c (by rw [hc]; exact hv)
