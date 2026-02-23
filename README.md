# 196 Carry Asymmetry Framework

A structural framework analyzing carry asymmetry in the 196 reverse-and-add problem (Lychrel conjecture), with the first complete characterization of when a single RAA step produces a palindrome.

## Paper

**Carry Asymmetry, Palindrome Characterization, and Conditional Non-Convergence in the Reverse-and-Add Process**

Author: Ando  
Date: February 2026

## Summary

We study the reverse-and-add (RAA) process n → n + rev(n) in arbitrary bases b ≥ 2, with particular attention to the trajectory of 196—the smallest suspected Lychrel number in base 10. We establish three main results.

### Unconditional results (Section 3)

**Theorem 3.1 (Carry Asymmetry).** For inputs of *any* digit-length L with at least one pair-sum exceeding b−1 and final carry c_L = 0, the carry chain of n + rev(n) is necessarily left-right asymmetric.

**Proposition 3.2.** Under the same hypotheses, n + rev(n) is not a palindrome.

**Theorem 3.7 (Complete Palindrome Characterization).** For any base b ≥ 3 and any L-digit input n, the sum n + rev(n) is a palindrome if and only if one of the following holds:

- **(I) Carry-free:** all pair-sums satisfy ps_i < b (with c_L = 0).
- **(II) Pair-sum degeneracy:** all pair-sums lie in {0, b+1} (with c_L = 1).

This is the first necessary-and-sufficient characterization of RAA palindromes in the literature. It is assembled from Theorems 3.4 (Case I), 3.5 (Case II necessity), and 3.6 (Case II sufficiency).

**Corollary 3.9 (Equivalent Formulation of the 196 Conjecture).** The number 196 is Lychrel if and only if, at every step, at least one pair-sum satisfies ps_i ≥ 10 and at least one satisfies ps_i ∉ {0, 11}. Both conditions hold at all 2000 verified steps.

### Conditional result (Section 4)

**Theorem 4.4 (Conditional Non-Convergence).** Under a single digit non-degeneracy hypothesis (Condition W2), the probability of palindrome formation at step t decays exponentially:

P(palindrome at step t) ≤ ρ*^⌊L_t/4⌋

where ρ* ≈ 0.512 in base 10.

### Proof landscape (Section 5)

A systematic "wall map" classifying 10 proof approaches and identifying four distinct barriers (W1–W4). All approaches except modular arithmetic (W3, provably impossible) and palindrome characterization (W4, unconditional) reduce to a single wall: digit non-degeneracy (W2).

### Base comparison (Section 6)

The carry asymmetry framework extends to bases 2–16, connecting to Sprague's 1963 proof for base 2. In base 2, Case (II) is algebraically impossible (since b+1 = 3 > 2(b−1) = 2), explaining why Lychrel proofs exist only in specific bases.

### Limitations

This is **not** a complete proof that 196 never reaches a palindrome. The conditional result depends on Condition W2 (digit non-degeneracy), which is computationally verified but formally unproven. The paper explicitly acknowledges this gap as having "Collatz-type difficulty."

## Running the verification

```
python3 verify_theorems.py
```

Requirements: Python 3.7+, no external dependencies.

The script runs 70 tests covering all theorems in the paper. Expected output:

```
Results: 70 passed, 0 failed
Runtime: ~16s
```

### Test coverage

| Section | Result | Tests | Description |
|---------|--------|-------|-------------|
| 1 | — | 3 | Basic RAA computation (196+691=887, 89→palindrome) |
| 2 | Defs 2.1–2.6 | 7 | Digits, pair sums, carry chains, output digits |
| 3.1 | Theorem 3.1 | 1 | Carry asymmetry (bases 2,3,5,10; L=2–6) |
| 3.1 | Proposition 3.2 | 1 | c_L=0, generator ⇒ non-palindrome |
| 3.2 | Theorem 3.4 | 2 | Case I: carry-free palindrome (both directions) |
| 3.2 | Theorem 3.5 | 1 | Case II necessity: c_L=1 palindrome ⇒ ps ∈ {0,b+1} |
| 3.2 | Theorem 3.6 | 1 | Case II sufficiency: ps ∈ {0,b+1} ⇒ palindrome |
| 3.2 | Theorem 3.7 | 1 | Complete characterization (bases 3,5,10) |
| 3.2 | Corollary 3.9 | 3 | 196 trajectory: 2000 steps, all non-palindrome |
| 3.2 | Remark 3.8 | 4 | Case II example: 9002+2009=11011 |
| 4 | Proposition 4.1 | 1 | Carry match probability ρ ≈ 0.502 |
| 5 | Wall W3 | 4 | Modular impossibility (mod 3,7,9,11) |
| 6 | Proposition 6.1 | 15 | Carry Markov chain (5 bases × 3 checks) |
| 6.2 | Base 2 | 3 | Sprague connection: 10110₂ non-convergence |
| — | Statistics | 6 | 196 trajectory statistics vs paper values |
| — | Definitions | 15 | Generator/absorber/neutral counts (5 bases) |
| — | Multibase | 2 | Complete characterization exhaustive (base 3,5) |

## Theoretical foundation

- **Holte (1997):** "Carries, Combinatorics, and an Amazing Matrix." *The American Mathematical Monthly*, 104(2), 138–149.
- **Diaconis & Fulman (2009):** "Carries, Shuffling, and Symmetric Functions." *Advances in Applied Mathematics*, 43(2), 176–196.
- **Sprague (1963):** *Recreation in Mathematics*. Blackie, London.

## Files

| File | Description |
|------|-------------|
| `paper.pdf` | Paper (PDF) |
| `paper.tex` | Paper (LaTeX source) |
| `verify_theorems.py` | Computational verification (70 tests) |
| `LICENSE` | MIT (code), CC BY 4.0 (paper) |

## License

Code: MIT License  
Paper: CC BY 4.0
