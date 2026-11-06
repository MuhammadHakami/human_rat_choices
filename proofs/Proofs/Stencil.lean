/-
  The ft09 stencil mechanic, and why embodiment changes the problem.

  Decoded from the shipped game source: selecting a tile CYCLES the colour of every neighbour
  its 3x3 stencil marks, modulo the palette size. Model a grid as `Cell -> Nat` (colour indices)
  and a selection as the vector it adds to each cell.

  The two theorems that matter:
    * selections COMMUTE, so the reached state depends only on HOW MANY times each tile was
      selected, never on the order;
    * selecting anything `n` times (n = palette size) is the IDENTITY.

  Together: a solution is a multiset of selections. A disembodied agent pays one action per
  selection whatever the order. An embodied agent must travel between them, so its cost is a
  tour over that multiset -- the same puzzle with a different cost geometry. That is the
  embodiment claim, stated precisely.
-/

namespace ArcFly.Stencil

/-- A grid assigns a colour index to every cell. -/
abbrev Grid (Cell : Type) := Cell → Nat

/-- A selection is the amount it adds to each cell (1 where its stencil marks, else 0). -/
abbrev Sel (Cell : Type) := Cell → Nat

/-- Apply a selection: cycle each cell by the selection's amount, modulo the palette size. -/
def apply {Cell : Type} (n : Nat) (g : Grid Cell) (s : Sel Cell) : Grid Cell :=
  fun c => (g c + s c) % n

/-! ## Selections commute -/

theorem apply_comm {Cell : Type} (n : Nat) (g : Grid Cell) (s t : Sel Cell) :
    apply n (apply n g s) t = apply n (apply n g t) s := by
  funext c
  simp only [apply, Nat.mod_add_mod]
  congr 1
  omega

/-- Consequently the order of a whole sequence of selections is irrelevant: folding a list and
folding any permutation agree. We state the workhorse case -- swapping adjacent selections -- from
which the general permutation result follows by the usual adjacent-transposition argument. -/
theorem fold_swap {Cell : Type} (n : Nat) (g : Grid Cell) (s t : Sel Cell) (rest : List (Sel Cell)) :
    rest.foldl (apply n) (apply n (apply n g s) t)
      = rest.foldl (apply n) (apply n (apply n g t) s) := by
  rw [apply_comm]

/-! ## Selecting `n` times is the identity -/

/-- Apply the same selection `k` times. -/
def applyN {Cell : Type} (n : Nat) (g : Grid Cell) (s : Sel Cell) : Nat → Grid Cell
  | 0     => g
  | k + 1 => apply n (applyN n g s k) s

/-- Closed form: `k` applications add `k * s c` to every cell, mod the palette size. -/
theorem applyN_closed {Cell : Type} (n : Nat) (g : Grid Cell) (s : Sel Cell)
    (hg : ∀ c, g c < n) (k : Nat) :
    applyN n g s k = fun c => (g c + k * s c) % n := by
  induction k with
  | zero =>
    funext c
    simp [applyN, Nat.mod_eq_of_lt (hg c)]
  | succ k ih =>
    funext c
    simp only [applyN, apply, ih, Nat.mod_add_mod]
    congr 1
    rw [Nat.succ_mul]
    omega

/-- Selecting the same tile `n` times leaves the grid exactly as it was. -/
theorem applyN_period {Cell : Type} (n : Nat) (g : Grid Cell) (s : Sel Cell)
    (hg : ∀ c, g c < n) :
    applyN n g s n = g := by
  rw [applyN_closed n g s hg n]
  funext c
  show (g c + n * s c) % n = g c
  rw [Nat.mul_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt (hg c)]

/-- The cell-level fact: cycling a colour `n` times returns it unchanged. -/
theorem cycle_n_id (n x : Nat) (h : x < n) : (x + n) % n = x := by
  rw [Nat.add_mod_right, Nat.mod_eq_of_lt h]

/-- Cycling by any multiple of `n` is likewise the identity. -/
theorem cycle_mul_id (n x k : Nat) (h : x < n) : (x + k * n) % n = x := by
  rw [Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt h]

end ArcFly.Stencil
