# 196 Carry Asymmetry Framework

A structural framework analyzing carry asymmetry in the 196 reverse-and-add problem (Lychrel conjecture). Conditional results with computational verification.

## Paper

**The 196 Problem: Carry Asymmetry Framework and Density Theory for the Lychrel Conjecture**

Authors: Ando; Claude Opus 4.6 (Anthropic)

## Summary

We develop a structural framework for analyzing the reverse-and-add (RAA) process applied to 196, building on the carries Markov chain theory of Holte (1997) and Diaconis–Fulman (2009).

The central objects are **pair sums** ps\_i = d\_i + d\_{L−1−i} and the **carry asymmetry** A(n) = #{i : c\_i ≠ c\_{L−1−i}}, which measures how far the carry chain deviates from palindromic symmetry.

### What is rigorously proven (8 theorems)

- **Carry-out = 0 case** (~58% of steps): If carry-out = 0 and A > 0, the RAA result is guaranteed to be non-palindromic (Theorem A).
- **Transition Formula**: Pair sums evolve deterministically via ps' = (ps mod 10) + ((ps+1) mod 10), classifying values as Generators (8/19), Preservers (3/19), or Absorbers (8/19) (Theorem G).
- **Generator–Absorber balance**: G = A holds in every base, explaining why base 2 has no Lychrel numbers (G = 0) while base 10 does (G = 8) (Theorem F).
- **Central Poison Theorem**: All 13 three-digit Lychrel candidates share a common structural feature — a high pair sum at the central position (Theorem E).
- **Carry Injection Theorem**: Palindrome states are exponentially unstable under RAA: P(palindrome → palindrome) = (4/9) · 2^{-(L/2-1)} (Theorem 8).

### What remains conditional

- **Carry-out = 1 case** (~42% of steps): We bound palindrome probability at ~10⁻⁷⁸ per step, but this depends on the **γ-saturation assumption** — that high pair-sum density persists indefinitely. Computationally verified to 3,000 steps but not proven.
- The paper explicitly acknowledges this gap has "Collatz-type difficulty."

### Validation

- **Base-2 consistency**: The framework independently recovers Sprague's proof that 10110₂ is Lychrel in binary.
- **Computational verification**: All theorems verified against exhaustive computation (see below).

## Running the verification

```
python3 verify_theorems.py
```

Requirements: Python 3.7+, no external dependencies.

The script runs 51 tests covering all theorems in the paper. Expected output:

```
Results: 51 passed, 0 failed
Runtime: ~15s
```

### Test coverage

| Section | Theorem | Tests | Description |
|---------|---------|-------|-------------|
| 1 | — | 2 | Basic RAA computation (196+691=887, etc.) |
| 1.1 | — | 5 | 196 pair sum / carry / asymmetry analysis |
| 1.2 | C | 2 | Pair Sum Equivalence + counterexample (29+92=121) |
| 1.3 | A | 1 | co=0, A>0 ⇒ non-palindrome (all 3-digit numbers) |
| 1.3 | B | 1 | co=0, ps≥10 ⇒ A>0 (all 3-digit numbers) |
| 1.3 | H | 1 | A>0 ⇒ some ps≥10 (10–9999) |
| 2 | G | 3 | Transition table + trajectory verification (500 steps) |
| 2.1 | G=A | 8 | Generator–Absorber balance (8 bases) |
| 2.3 | D | 1 | Parity-Carry Identity (1000 steps) |
| 4.1 | E | 3 | Central Poison: all 13 three-digit Lychrel candidates |
| 4.2 | F | 6 | Generator count formula + base-2 uniqueness |
| 4.3 | — | 3 | Base-2 Lychrel re-derivation (10110₂, 5000 steps) |
| 5.1 | — | 1 | Stationary distribution π=[1/2, 1/2] |
| 5.4 | 8 | 4 | Carry Injection: palindrome→palindrome probability |
| 6 | — | 8 | 196 trajectory statistics (3000 steps) vs paper values |
| 7 | 9 | 2 | γ-saturation preserved at all co=1 steps |

## Limitations

This is **not** a complete proof that 196 never reaches a palindrome. The carry-out = 1 case remains conditional on an unproven assumption. See Section 7 of the paper for details.

## Theoretical foundation

- **Holte (1997)**: "Carries, Combinatorics, and an Amazing Matrix." *The American Mathematical Monthly*, 104(2), 138–149.
- **Diaconis & Fulman (2009)**: "Carries, Shuffling, and Symmetric Functions." *Advances in Applied Mathematics*, 43(2), 176–196.

## License

Code: MIT License  
Paper: CC BY 4.0
