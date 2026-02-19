#!/usr/bin/env python3
"""
196 Carry Asymmetry Framework — Theorem Verification Script

Computationally verifies the theorems in:
"The 196 Problem: Carry Asymmetry Framework and Density Theory
 for the Lychrel Conjecture"

Authors: Ando; Claude Opus 4.6 (Anthropic)

Usage:
    python3 verify_theorems.py

Requirements: Python 3.7+, no external dependencies.
Expected runtime: ~15 seconds.
"""

import sys
import time

# ============================================================
# 共通関数
# ============================================================

def raa(n: int) -> int:
    """Reverse-and-Add: n + reverse(n)"""
    return n + int(str(n)[::-1])


def raa_base2(n: int) -> int:
    """基数2でのReverse-and-Add"""
    bits = bin(n)[2:]
    return n + int(bits[::-1], 2)


def analyze(n: int, base: int = 10):
    """
    数値 n を解析し、ペア和・キャリーチェーン・非対称度を返す。

    戻り値:
        digits  : 桁列 (LSBファースト)
        ps      : ペア和 ps_i = d_i + d_{L-1-i}
        carry   : キャリーチェーン c_0, ..., c_L
        co      : キャリーアウト c_L
        A       : 非対称度 (carry[i] != carry[L-1-i] となる位置の数)
    """
    if base == 10:
        digits = list(map(int, str(n)))[::-1]
    elif base == 2:
        digits = list(map(int, bin(n)[2:]))[::-1]
    else:
        digits = []
        tmp = n
        while tmp > 0:
            digits.append(tmp % base)
            tmp //= base
        if not digits:
            digits = [0]

    L = len(digits)
    ps = [digits[i] + digits[L - 1 - i] for i in range(L)]

    carry = [0] * (L + 1)
    for i in range(L):
        carry[i + 1] = (ps[i] + carry[i]) // base
    co = carry[L]

    A = sum(1 for i in range(L // 2) if carry[i] != carry[L - 1 - i])

    return digits, ps, carry, co, A


def transition(ps_val: int, base: int = 10) -> int:
    """遷移公式: ps' = (ps mod b) + ((ps+1) mod b)"""
    return (ps_val % base) + ((ps_val + 1) % base)


def is_palindrome(s: str) -> bool:
    return s == s[::-1]


# ============================================================
# テストフレームワーク
# ============================================================

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    """テスト結果を記録・表示"""
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}  {detail}")


# ============================================================
# Section 1: Carry Asymmetry Theory
# ============================================================

def test_basic_raa():
    """論文Section 1冒頭の計算例"""
    print("\n=== Section 1: Basic RAA computation ===")
    check("196 + 691 = 887", raa(196) == 887)
    check("887 + 788 = 1675", raa(887) == 1675)


def test_example_196():
    """論文Section 1.1のExample: 196の解析"""
    print("\n=== Section 1.1: Example — 196 analysis ===")
    digits, ps, carry, co, A = analyze(196)
    check("digits = [6,9,1] (LSB first)", digits == [6, 9, 1])
    check("ps = [7,18,7]", ps == [7, 18, 7])
    check("carry = [0,0,1,0]", carry == [0, 0, 1, 0])
    check("carry-out = 0", co == 0)
    check("A = 1 (pos 0: carry[0]=0 != carry[2]=1)", A == 1)


def test_theorem_C():
    """Theorem C (Pair Sum Equivalence): (all ps<10) <=> (A=0, co=0)
    論文Section 1.2"""
    print("\n=== Theorem C: Pair Sum Equivalence (10-9999) ===")
    violations = 0
    tested = 0
    for n in range(10, 10000):
        _, ps, _, co, A = analyze(n)
        all_ps_lt10 = all(p < 10 for p in ps)
        cond = (A == 0 and co == 0)
        tested += 1
        if all_ps_lt10 != cond:
            violations += 1
    check(f"(all ps<10) <=> (A=0, co=0): {tested} cases, 0 violations",
          violations == 0, f"violations={violations}")

    # 論文の反例: 29+92=121 は回文だが ps=[11,11], co=1
    _, ps_29, _, co_29, _ = analyze(29)
    result_29 = raa(29)
    check("Counterexample: 29+92=121 palindrome with ps=[11,11], co=1",
          co_29 == 1 and ps_29 == [11, 11] and result_29 == 121
          and is_palindrome(str(result_29)))


def test_theorem_A():
    """Theorem A (Non-palindrome): co=0, A>0 => non-palindrome
    論文Section 1.3"""
    print("\n=== Theorem A: co=0, A>0 => non-palindrome (all 3-digit) ===")
    violations = 0
    tested = 0
    for n in range(100, 1000):
        _, _, _, co, A = analyze(n)
        if co == 0 and A > 0:
            tested += 1
            if is_palindrome(str(raa(n))):
                violations += 1
    check(f"{tested} cases with co=0, A>0: 0 palindromes",
          violations == 0, f"violations={violations}")


def test_theorem_B():
    """Theorem B: co=0, ps>=10 exists => A>0 (same step)
    論文Section 1.3"""
    print("\n=== Theorem B: co=0, ps>=10 => A>0 (all 3-digit) ===")
    violations = 0
    tested = 0
    for n in range(100, 1000):
        _, ps, _, co, A = analyze(n)
        if co == 0 and any(p >= 10 for p in ps):
            tested += 1
            if A == 0:
                violations += 1
    check(f"{tested} cases with co=0, ps>=10: all have A>0",
          violations == 0, f"violations={violations}")


def test_theorem_H():
    """Theorem H (Converse): A>0 => some ps>=10 exists
    論文Section 1.3"""
    print("\n=== Theorem H: A>0 => some ps>=10 (10-9999) ===")
    violations = 0
    tested = 0
    for n in range(10, 10000):
        _, ps, _, _, A = analyze(n)
        if A > 0:
            tested += 1
            if not any(p >= 10 for p in ps):
                violations += 1
    check(f"{tested} cases with A>0: all have some ps>=10",
          violations == 0, f"violations={violations}")


# ============================================================
# Section 2: Poison Feedback Loop
# ============================================================

def test_theorem_G_table():
    """Theorem G (Transition Formula): 遷移テーブル
    論文Section 2"""
    print("\n=== Theorem G: Transition table ===")
    paper_values = [1, 3, 5, 7, 9, 11, 13, 15, 17,
                    9, 1, 3, 5, 7, 9, 11, 13, 15, 17]
    computed = [transition(p) for p in range(19)]
    check("Transition table matches paper", computed == paper_values)

    generators = sum(1 for p in range(19) if transition(p) >= 10)
    preservers = sum(1 for p in range(19) if transition(p) == 9)
    absorbers = sum(1 for p in range(19) if transition(p) < 9)
    check("G=8, P=3, A=8",
          generators == 8 and preservers == 3 and absorbers == 8)


def test_theorem_G_trajectory():
    """Theorem G: 196軌道上で遷移公式を検証"""
    print("\n=== Theorem G: Trajectory verification (500 steps) ===")
    n = 196
    violations = 0
    tested = 0
    for _ in range(500):
        d1, ps1, c1, co1, _ = analyze(n)
        L1 = len(d1)
        asym1 = [i for i in range(L1 // 2) if c1[i] != c1[L1 - 1 - i]]

        n_next = raa(n)
        _, ps2, _, _, _ = analyze(n_next)
        L2 = len(str(n_next))

        if co1 == 0:
            for i in asym1:
                if i < L2:
                    tested += 1
                    if transition(ps1[i]) != ps2[i]:
                        violations += 1
        n = n_next
    check(f"Transition holds at {tested} asymmetry positions",
          violations == 0, f"violations={violations}")


def test_ga_equilibrium():
    """G=A Equilibrium: Generator = Absorber in every base
    論文Section 2.1"""
    print("\n=== G=A Equilibrium (multiple bases) ===")
    for b in [2, 3, 5, 7, 10, 13, 16, 20]:
        g, a = 0, 0
        for ps_val in range(2 * b - 1):
            t = transition(ps_val, b)
            if t >= b:
                g += 1
            elif t < b - 1:
                a += 1
        check(f"Base {b}: G={g} == A={a}", g == a)


def test_theorem_D():
    """Theorem D (Parity-Carry Identity): #{odd ps'} = 2A(n) when co=0
    論文Section 2.3"""
    print("\n=== Theorem D: Parity-Carry Identity (1000 steps) ===")
    n = 196
    violations = 0
    tested = 0
    for _ in range(1000):
        _, _, _, co1, A1 = analyze(n)
        n_next = raa(n)
        _, ps2, _, _, _ = analyze(n_next)
        if co1 == 0:
            odd_count = sum(1 for p in ps2 if p % 2 == 1)
            tested += 1
            if odd_count != 2 * A1:
                violations += 1
        n = n_next
    check(f"#(odd ps') = 2*A(n) at {tested} co=0 steps",
          violations == 0, f"violations={violations}")


# ============================================================
# Section 4: Why Base 10 Is Singular
# ============================================================

def test_theorem_E():
    """Theorem E (Central Poison): All 13 three-digit Lychrel candidates
    have central poison (ps[1] >= 10).
    論文Section 4.1"""
    print("\n=== Theorem E: Central Poison (3-digit Lychrel) ===")
    print("  (Identifying Lychrel candidates... ~15s)")
    lychrel = []
    for n in range(100, 1000):
        x = n
        found = False
        for _ in range(500):
            x = raa(x)
            if is_palindrome(str(x)):
                found = True
                break
        if not found:
            lychrel.append(n)

    expected = [196, 295, 394, 493, 592, 689, 691,
                788, 790, 879, 887, 978, 986]
    check("3-digit Lychrel candidates = 13", lychrel == expected,
          f"got {lychrel}")

    all_central = all(analyze(n)[1][1] >= 10 for n in lychrel)
    check("All 13 have central poison (ps[1] >= 10)", all_central)

    # 論文Table: Edge-only poison → 0 Lychrel candidates
    edge_only_lychrel = 0
    for n in lychrel:
        _, ps, _, _, _ = analyze(n)
        if ps[0] >= 10 and ps[1] < 10:
            edge_only_lychrel += 1
    check("Edge-only poison => 0 Lychrel candidates",
          edge_only_lychrel == 0)


def test_theorem_F():
    """Theorem F (Poison Count Formula): Generator counts by base
    論文Section 4.2"""
    print("\n=== Theorem F: Generator count formula ===")
    for b in [2, 3, 5, 10, 16]:
        g = sum(1 for ps_val in range(2 * b - 1)
                if transition(ps_val, b) >= b)
        if b == 2:
            expected = 0
        elif b % 2 == 1:
            expected = b - 1
        else:
            expected = b - 2
        check(f"Base {b}: G={g} (formula={expected})", g == expected)

    # Base 2 uniqueness
    check("Base 2 is unique with G=0 (among bases 2-30)",
          all(
              sum(1 for p in range(2 * b - 1)
                  if transition(p, b) >= b) > 0
              for b in range(3, 31)
          ))


def test_base2_lychrel():
    """Section 4.3: Base-2 Lychrel re-derivation (10110_2 = 22)"""
    print("\n=== Section 4.3: Base-2 Lychrel re-derivation ===")

    # 基数2: 全遷移がPreserver (ps'=1)
    all_preservers = all(transition(p, 2) == 1 for p in range(3))
    check("Base 2: all transitions => Preserver (ps'=1)", all_preservers)

    g = sum(1 for p in range(3) if transition(p, 2) >= 2)
    a = sum(1 for p in range(3) if transition(p, 2) < 1)
    check("Base 2: G=0, Absorber=0", g == 0 and a == 0)

    # 論文: verified 5,000 steps, minimum ps=2 count = 1
    n = 22
    min_ps2 = float('inf')
    for _ in range(5000):
        _, ps, _, _, _ = analyze(n, base=2)
        ps2_count = sum(1 for p in ps if p == 2)
        min_ps2 = min(min_ps2, ps2_count)
        n = raa_base2(n)
    check(f"5000 steps: min #(ps=2) = {min_ps2} >= 1", min_ps2 >= 1)


# ============================================================
# Section 5: Probabilistic Density Theory
# ============================================================

def test_stationary_distribution():
    """Stationary distribution pi = [1/2, 1/2]
    論文Section 5.1"""
    print("\n=== Section 5.1: Stationary distribution ===")
    n = 196
    carry1 = 0
    positions = 0
    for _ in range(1000):
        _, _, carry, _, _ = analyze(n)
        L = len(carry) - 1
        for i in range(1, L):
            carry1 += carry[i]
            positions += 1
        n = raa(n)
    ratio = carry1 / positions
    check(f"carry=1 fraction = {ratio:.4f} (theory: 0.5000)",
          abs(ratio - 0.5) < 0.01)


def test_theorem_8():
    """Theorem 8 (Carry Injection):
    P(palindrome -> palindrome) = (4/9) * 2^{-(L/2 - 1)}
    論文Section 5.4, verified exactly for L=2,...,8"""
    print("\n=== Theorem 8: Carry Injection (even L = 2,4,6,8) ===")
    for L in [2, 4, 6, 8]:
        total = 0
        pal_after = 0
        for n in range(10 ** (L - 1), 10 ** L):
            s = str(n)
            if is_palindrome(s):
                total += 1
                if is_palindrome(str(raa(n))):
                    pal_after += 1
        prob = pal_after / total
        theory = (4 / 9) * 2 ** (-(L // 2 - 1))
        check(f"L={L}: measured={prob:.6f}, theory={theory:.6f}",
              abs(prob - theory) < 1e-6)


# ============================================================
# Section 6: Measured Statistics
# ============================================================

def test_trajectory_statistics():
    """196 trajectory statistics (3000 steps)
    論文Section 6"""
    print("\n=== Section 6: Trajectory statistics (3000 steps) ===")
    n = 196
    co0 = 0
    co1 = 0
    A_min = float('inf')
    A_max = 0
    A_sum = 0
    A_positive_all = True
    ps_ge10_densities = []

    for _ in range(3000):
        d, ps, _, co, A = analyze(n)
        L = len(d)
        if co == 0:
            co0 += 1
        else:
            co1 += 1
        A_min = min(A_min, A)
        A_max = max(A_max, A)
        A_sum += A
        if A == 0:
            A_positive_all = False
        ps_ge10 = sum(1 for p in ps if p >= 10)
        ps_ge10_densities.append(ps_ge10 / L)
        n = raa(n)

    check(f"co=0 steps: {co0} (paper: 1735)", co0 == 1735)
    check(f"co=1 steps: {co1} (paper: 1265)", co1 == 1265)
    check(f"A min={A_min} (paper: 1)", A_min == 1)
    check(f"A max={A_max} (paper: 350)", A_max == 350)
    check(f"A mean={A_sum/3000:.1f} (paper: 156.6)",
          abs(A_sum / 3000 - 156.6) < 0.1)
    check("A > 0 at all 3000 steps", A_positive_all)

    mean_density = sum(ps_ge10_densities) / len(ps_ge10_densities)
    check(f"ps>=10 density={mean_density:.3f} (paper: 0.449)",
          abs(mean_density - 0.449) < 0.002)

    final_digits = len(str(n))
    check(f"Final digit count={final_digits} (paper: 1268)",
          final_digits == 1268)


# ============================================================
# Section 7: Conditional Theorem
# ============================================================

def test_gamma_saturation():
    """Theorem 9: gamma-saturation (gamma=0.10) preserved at co=1 steps
    論文Section 7.2"""
    print("\n=== Theorem 9: gamma-saturation at co=1 steps ===")
    n = 196
    violations = 0
    co1_steps = 0
    min_density = 1.0

    for _ in range(3000):
        _, _, _, co, _ = analyze(n)
        n_next = raa(n)

        if co == 1:
            co1_steps += 1
            d_next, ps_next, _, _, _ = analyze(n_next)
            L_next = len(d_next)
            density = sum(1 for p in ps_next if p >= 10) / L_next
            min_density = min(min_density, density)
            if density < 0.10:
                violations += 1

        n = n_next

    check(f"gamma=0.10 preserved at all {co1_steps} co=1 steps",
          violations == 0, f"violations={violations}")
    check(f"Min ps>=10 density after co=1: {min_density:.3f} (paper min: 0.29 for L>=50)",
          min_density >= 0.10)


# ============================================================
# メイン
# ============================================================

def main():
    global passed, failed
    start = time.time()

    print("=" * 60)
    print("196 Carry Asymmetry Framework — Theorem Verification")
    print("=" * 60)

    # Section 1: Carry Asymmetry Theory
    test_basic_raa()
    test_example_196()
    test_theorem_C()
    test_theorem_A()
    test_theorem_B()
    test_theorem_H()

    # Section 2: Poison Feedback Loop
    test_theorem_G_table()
    test_theorem_G_trajectory()
    test_ga_equilibrium()
    test_theorem_D()

    # Section 4: Why Base 10 Is Singular
    test_theorem_E()
    test_theorem_F()
    test_base2_lychrel()

    # Section 5: Probabilistic Density Theory
    test_stationary_distribution()
    test_theorem_8()

    # Section 6: Measured Statistics
    test_trajectory_statistics()

    # Section 7: Conditional Theorem
    test_gamma_saturation()

    elapsed = time.time() - start

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print(f"Runtime: {elapsed:.1f}s")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
