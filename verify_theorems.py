#!/usr/bin/env python3
"""
196 Carry Asymmetry Framework — 定理検証スクリプト
==================================================

論文 "Carry Asymmetry, Palindrome Characterization, and Conditional
Non-Convergence in the Reverse-and-Add Process" (Ando, 2026) の
全定理を計算的に検証する。

依存ライブラリ: なし（Python 3.7+ 標準ライブラリのみ）
実行: python3 verify_theorems.py
"""

import time
from typing import List, Tuple, Dict

# ============================================================
# 基本関数群
# ============================================================

def digits(n: int, base: int = 10) -> List[int]:
    """整数 n の桁表現を返す（d[0] が最下位桁）"""
    if n == 0:
        return [0]
    ds = []
    while n > 0:
        ds.append(n % base)
        n //= base
    return ds


def from_digits(ds: List[int], base: int = 10) -> int:
    """桁表現から整数を復元"""
    return sum(d * base**i for i, d in enumerate(ds))


def reverse_number(n: int, base: int = 10) -> int:
    """桁を逆転した数を返す"""
    ds = digits(n, base)
    return from_digits(ds[::-1], base)


def raa(n: int, base: int = 10) -> int:
    """Reverse-and-Add: n + rev(n)"""
    return n + reverse_number(n, base)


def is_palindrome(n: int, base: int = 10) -> bool:
    """回文判定"""
    ds = digits(n, base)
    return ds == ds[::-1]


def pair_sums(n: int, base: int = 10) -> List[int]:
    """ペアサム列 ps_i = d_i + d_{L-1-i} を返す"""
    ds = digits(n, base)
    L = len(ds)
    return [ds[i] + ds[L - 1 - i] for i in range(L)]


def carry_chain(n: int, base: int = 10) -> List[int]:
    """n + rev(n) のキャリーチェーン c_0, c_1, ..., c_L を返す"""
    ps = pair_sums(n, base)
    L = len(ps)
    c = [0] * (L + 1)
    for i in range(L):
        c[i + 1] = (ps[i] + c[i]) // base
    return c


def carry_symmetric(carries: List[int]) -> bool:
    """キャリーチェーンが左右対称か（c_i == c_{L-i}）"""
    L = len(carries) - 1
    return all(carries[i] == carries[L - i] for i in range(L + 1))


def output_digits(n: int, base: int = 10) -> List[int]:
    """n + rev(n) の出力桁列"""
    ps = pair_sums(n, base)
    c = carry_chain(n, base)
    L = len(ps)
    out = [(ps[i] + c[i]) % base for i in range(L)]
    if c[L] == 1:
        out.append(1)
    return out


# ============================================================
# テスト基盤
# ============================================================

passed = 0
failed = 0
test_details = []


def test(name: str, condition: bool, detail: str = ""):
    """テスト結果を記録"""
    global passed, failed
    if condition:
        passed += 1
        status = "PASS"
    else:
        failed += 1
        status = "FAIL"
    test_details.append((name, status, detail))
    if not condition:
        print(f"  *** FAIL: {name} — {detail}")


# ============================================================
# Section 1: 基本的な RAA 計算
# ============================================================

def test_basic_raa():
    """基本的な RAA 計算の確認"""
    # 196 + 691 = 887
    test("RAA(196)=887", raa(196) == 887, f"got {raa(196)}")
    # 887 + 788 = 1675
    test("RAA(887)=1675", raa(887) == 1675, f"got {raa(887)}")
    # 89 は 24 回で回文に到達
    n = 89
    for _ in range(24):
        n = raa(n)
    test("89→24steps→palindrome",
         is_palindrome(n), f"step 24: {n}")


# ============================================================
# Section 2: 定義の検証
# ============================================================

def test_definitions():
    """Definition 2.1–2.6 の基本動作確認"""
    # 196: digits = [6, 9, 1], pair sums = [6+1, 9+9, 1+6] = [7, 18, 7]
    ds = digits(196)
    test("digits(196)=[6,9,1]", ds == [6, 9, 1], f"got {ds}")

    ps = pair_sums(196)
    test("pair_sums(196)=[7,18,7]", ps == [7, 18, 7], f"got {ps}")

    # ペアサム対称性: ps_i == ps_{L-1-i}
    test("pair_sum_symmetry",
         all(ps[i] == ps[len(ps) - 1 - i] for i in range(len(ps))))

    # キャリーチェーン: c_0=0, c_1=floor(7/10)=0, c_2=floor(18/10)=1, c_3=floor(7+1)/10=0
    c = carry_chain(196)
    test("carry_chain(196)=[0,0,1,0]",
         c == [0, 0, 1, 0], f"got {c}")

    # 出力桁: o_0=(7+0)%10=7, o_1=(18+0)%10=8, o_2=(7+1)%10=8, c_L=0
    out = output_digits(196)
    test("output_digits(196)=[7,8,8]",
         out == [7, 8, 8], f"got {out}")

    # 887 = 7+8+8*10+... → digits [7,8,8]
    test("196+691=887", from_digits(out) == 887)

    # 196 のキャリーチェーンは左右非対称
    test("196_carry_not_symmetric", not carry_symmetric(c))


# ============================================================
# Section 3.1: Theorem 3.1 — Carry Asymmetry
# ============================================================

def test_theorem_3_1():
    """
    Theorem 3.1 (Carry Asymmetry):
    ps_j >= b かつ c_L = 0 ならば、キャリーチェーンは左右非対称。
    複数の基数・桁数で網羅的に検証。
    """
    violations = 0
    total = 0
    for base in [2, 3, 5, 10]:
        for L in range(2, 7):
            lo = base ** (L - 1)
            hi = base ** L
            for n in range(lo, min(hi, lo + 2000)):
                ps = pair_sums(n, base)
                c = carry_chain(n, base)
                has_generator = any(p >= base for p in ps)
                cL_zero = (c[-1] == 0)
                if has_generator and cL_zero:
                    total += 1
                    if carry_symmetric(c):
                        violations += 1
    test("Thm3.1_carry_asymmetry",
         violations == 0,
         f"{violations}/{total} violations")


# ============================================================
# Section 3.1: Proposition 3.2 — c_L=0 で非回文
# ============================================================

def test_proposition_3_2():
    """
    Proposition 3.2:
    ps_j >= b かつ c_L = 0 ならば n + rev(n) は回文でない。
    """
    violations = 0
    total = 0
    for base in [3, 5, 10]:
        for L in range(2, 6):
            lo = base ** (L - 1)
            hi = base ** L
            for n in range(lo, min(hi, lo + 3000)):
                ps = pair_sums(n, base)
                c = carry_chain(n, base)
                has_generator = any(p >= base for p in ps)
                cL_zero = (c[-1] == 0)
                if has_generator and cL_zero:
                    total += 1
                    s = raa(n, base)
                    if is_palindrome(s, base):
                        violations += 1
    test("Prop3.2_cL0_not_palindrome",
         violations == 0,
         f"{violations}/{total} violations")


# ============================================================
# Section 3.2: Theorem 3.4 — Carry-Free Palindrome (Case I)
# ============================================================

def test_theorem_3_4():
    """
    Theorem 3.4 (Carry-Free Palindrome):
    (前方) 全ての ps_i < b ⟹ c_L=0 かつ palindrome
    (逆方) c_L=0 かつ palindrome ⟹ 全ての ps_i < b
    """
    fwd_violations = 0
    fwd_total = 0
    rev_violations = 0
    rev_total = 0
    for base in [3, 5, 10]:
        for L in range(2, 6):
            lo = base ** (L - 1)
            hi = base ** L
            for n in range(lo, min(hi, lo + 3000)):
                ps = pair_sums(n, base)
                c = carry_chain(n, base)
                s = raa(n, base)
                pal = is_palindrome(s, base)
                all_ps_lt_b = all(p < base for p in ps)

                # 前方向: 全 ps_i < b → 回文
                if all_ps_lt_b:
                    fwd_total += 1
                    if not (c[-1] == 0 and pal):
                        fwd_violations += 1

                # 逆方向: c_L=0 かつ回文 → 全 ps_i < b
                if c[-1] == 0 and pal:
                    rev_total += 1
                    if not all_ps_lt_b:
                        rev_violations += 1

    test("Thm3.4_forward_all_ps_lt_b→pal",
         fwd_violations == 0,
         f"{fwd_violations}/{fwd_total} violations")
    test("Thm3.4_converse_cL0_pal→all_ps_lt_b",
         rev_violations == 0,
         f"{rev_violations}/{rev_total} violations")


# ============================================================
# Section 3.2: Theorem 3.5 — Pair-Sum Degeneracy (Necessity)
# ============================================================

def test_theorem_3_5():
    """
    Theorem 3.5 (Pair-Sum Degeneracy: Necessity):
    c_L = 1 かつ palindrome ⟹ 全 ps_i ∈ {0, b+1}
    """
    violations = 0
    palindromes_found = 0
    for base in [3, 5, 10, 16]:
        for L in range(2, 7):
            lo = base ** (L - 1)
            hi = base ** L
            for n in range(lo, min(hi, lo + 5000)):
                c = carry_chain(n, base)
                if c[-1] != 1:
                    continue
                s = raa(n, base)
                if not is_palindrome(s, base):
                    continue
                palindromes_found += 1
                ps = pair_sums(n, base)
                if not all(p in (0, base + 1) for p in ps):
                    violations += 1
    test("Thm3.5_cL1_pal→ps_in_{0,b+1}",
         violations == 0,
         f"{violations} violations among {palindromes_found} cL=1 palindromes")


# ============================================================
# Section 3.2: Theorem 3.6 — Pair-Sum Degeneracy (Sufficiency)
# ============================================================

def test_theorem_3_6():
    """
    Theorem 3.6 (Pair-Sum Degeneracy: Sufficiency):
    全 ps_i ∈ {0, b+1} かつ c_L = 1 ⟹ palindrome
    """
    violations = 0
    total = 0
    for base in [2, 3, 5, 10, 16]:
        bp1 = base + 1
        max_ps = 2 * (base - 1)
        if bp1 > max_ps:
            continue  # base=2: b+1=3 > 2, Case II は不可能
        for L in range(2, 9):
            # ps_i ∈ {0, b+1} のパターンを列挙
            # L 桁なのでペアサムは対称: ps_i = ps_{L-1-i}
            # 独立なのは ceil(L/2) 個
            half = (L + 1) // 2
            for mask in range(2**half):
                # 各独立位置のペアサムを決定
                ps_half = []
                for j in range(half):
                    if mask & (1 << j):
                        ps_half.append(bp1)
                    else:
                        ps_half.append(0)
                # 対称にして全 ps を構成
                ps_full = ps_half + ps_half[:L - half][::-1]
                # キャリーチェーンを計算
                c = [0] * (L + 1)
                for i in range(L):
                    c[i + 1] = (ps_full[i] + c[i]) // base
                if c[L] != 1:
                    continue
                total += 1
                # 出力桁を計算
                out = [(ps_full[i] + c[i]) % base for i in range(L)]
                out.append(1)  # c_L = 1
                # 回文判定
                if out != out[::-1]:
                    violations += 1
    test("Thm3.6_ps_degen_suf→pal",
         violations == 0,
         f"{violations}/{total} violations")


# ============================================================
# Section 3.2: Theorem 3.7 — Complete Palindrome Characterization
# ============================================================

def test_theorem_3_7():
    """
    Theorem 3.7 (Complete Palindrome Characterization):
    n + rev(n) が回文 ⟺ Case I (全 ps_i < b) または Case II (全 ps_i ∈ {0, b+1})
    全方向を検証。
    """
    violations = 0
    total = 0
    for base in [3, 5, 10]:
        for L in range(2, 6):
            lo = base ** (L - 1)
            hi = base ** L
            for n in range(lo, min(hi, lo + 5000)):
                total += 1
                ps = pair_sums(n, base)
                s = raa(n, base)
                pal = is_palindrome(s, base)
                case_I = all(p < base for p in ps)
                case_II = all(p in (0, base + 1) for p in ps)
                char = case_I or case_II

                # 回文 ⟺ (Case I or Case II)
                if pal != char:
                    violations += 1
    test("Thm3.7_complete_char",
         violations == 0,
         f"{violations}/{total} violations")


# ============================================================
# Section 3.2: Corollary 3.9 — 196 軌道の非回文性
# ============================================================

def test_corollary_3_9():
    """
    Corollary 3.9 (Equivalent Formulation):
    196 軌道の最初 2000 ステップが全て非回文であることを
    Theorem 3.7 の条件で確認。
    """
    n = 196
    steps = 2000
    cL0_count = 0
    cL1_count = 0
    all_non_pal = True

    for t in range(steps):
        ps = pair_sums(n)
        c = carry_chain(n)
        has_gen = any(p >= 10 for p in ps)
        has_non_degen = any(p not in (0, 11) for p in ps)

        if c[-1] == 0:
            cL0_count += 1
            if not has_gen:
                all_non_pal = False
        else:
            cL1_count += 1
            if not has_non_degen:
                all_non_pal = False
        n = raa(n)

    test("Cor3.9_196_2000steps_all_nonpal",
         all_non_pal,
         f"cL=0: {cL0_count}, cL=1: {cL1_count}")

    # 論文記載値との照合: cL=0 が 1169, cL=1 が 831
    test("Cor3.9_cL0_count=1169",
         cL0_count == 1169,
         f"got {cL0_count}")
    test("Cor3.9_cL1_count=831",
         cL1_count == 831,
         f"got {cL1_count}")


# ============================================================
# Section 3.2: Remark 3.8 — Case II 回文の実例
# ============================================================

def test_remark_case_II_example():
    """
    Remark 3.8: n=9002 → ps=(11,0,0,11) → 9002+2009=11011（回文、c_L=1）
    """
    n = 9002
    ps = pair_sums(n)
    test("CaseII_example_ps",
         ps == [11, 0, 0, 11], f"got {ps}")
    s = raa(n)
    test("CaseII_example_sum",
         s == 11011, f"got {s}")
    test("CaseII_example_palindrome",
         is_palindrome(s))
    c = carry_chain(n)
    test("CaseII_example_cL=1",
         c[-1] == 1, f"c_L={c[-1]}")


# ============================================================
# Section 4: Proposition 4.1 — キャリー一致確率
# ============================================================

def test_proposition_4_1():
    """
    Proposition 4.1: c_L=0 ステップでの I_k = 1[c_k == c_{L-1-k}] の
    無条件一致確率 ρ ≈ 0.502
    """
    n = 196
    match_count = 0
    total_positions = 0
    for t in range(2000):
        c = carry_chain(n)
        L = len(c) - 1
        if c[-1] != 0:
            n = raa(n)
            continue
        for k in range(L // 2):
            total_positions += 1
            if c[k] == c[L - 1 - k]:
                match_count += 1
        n = raa(n)

    rho = match_count / total_positions if total_positions > 0 else 0
    test("Prop4.1_rho≈0.502",
         abs(rho - 0.502) < 0.01,
         f"ρ = {rho:.4f}")


# ============================================================
# Section 6: Proposition 6.1 — Carry Markov Chain
# ============================================================

def test_proposition_6_1():
    """
    Proposition 6.1: キャリーマルコフ連鎖の遷移確率と第二固有値 λ₂ = 1/b
    """
    for base in [2, 3, 5, 10, 16]:
        # 理論値
        p01_theory = (base - 1) / (2 * base)  # P(0→1)
        p11_theory = (base + 1) / (2 * base)  # P(1→1)
        lambda2_theory = 1 / base

        # 数値計算: 全 (d_i, d_{L-1-i}) ペアでの遷移
        transitions = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
        for d1 in range(base):
            for d2 in range(base):
                ps = d1 + d2
                for c_in in [0, 1]:
                    c_out = (ps + c_in) // base
                    transitions[(c_in, c_out)] += 1

        total_from_0 = transitions[(0, 0)] + transitions[(0, 1)]
        total_from_1 = transitions[(1, 0)] + transitions[(1, 1)]
        p01_empirical = transitions[(0, 1)] / total_from_0
        p11_empirical = transitions[(1, 1)] / total_from_1

        test(f"Prop6.1_base{base}_P(0→1)",
             abs(p01_empirical - p01_theory) < 1e-10,
             f"theory={p01_theory:.4f}, got={p01_empirical:.4f}")
        test(f"Prop6.1_base{base}_P(1→1)",
             abs(p11_empirical - p11_theory) < 1e-10,
             f"theory={p11_theory:.4f}, got={p11_empirical:.4f}")

        # 第二固有値: λ₂ = P(0→0) - P(0→1) = (b+1)/(2b) - (b-1)/(2b) = 1/b
        p00 = 1 - p01_empirical
        lambda2_empirical = p00 - p01_empirical
        test(f"Prop6.1_base{base}_λ₂=1/{base}",
             abs(lambda2_empirical - lambda2_theory) < 1e-10,
             f"theory={lambda2_theory:.4f}, got={lambda2_empirical:.4f}")


# ============================================================
# Section 6.2: Base 2 — Sprague 接続
# ============================================================

def test_base_2_sprague():
    """
    Base 2: Case II は不可能（b+1=3 > 2(b-1)=2）。
    よって回文は Case I のみ。
    10110₂ (=22) が Lychrel であることを確認。
    """
    # b+1 = 3 > max pair sum = 2 → Case II impossible
    test("Base2_CaseII_impossible",
         3 > 2 * (2 - 1))

    # 22 (=10110₂) の軌道 5000 ステップ
    n = 22  # 10110 in binary
    ds = digits(n, 2)
    test("22_binary=10110",
         ds == [0, 1, 1, 0, 1], f"got {ds}")

    never_palindrome = True
    for t in range(5000):
        n = raa(n, 2)
        if is_palindrome(n, 2):
            never_palindrome = False
            break
    test("Base2_10110_5000steps_no_palindrome",
         never_palindrome)


# ============================================================
# Section 5: Wall W3 — Modular impossibility
# ============================================================

def test_wall_W3():
    """
    Wall W3: n + rev(n) mod m と回文 mod m は同じ残基集合を含む。
    """
    for m in [3, 7, 9, 11]:
        raa_residues = set()
        pal_residues = set()
        for n in range(100, 10000):
            raa_residues.add(raa(n) % m)
        for n in range(100, 10000):
            if is_palindrome(n):
                pal_residues.add(n % m)
        # 回文残基 ⊆ RAA残基
        test(f"W3_mod{m}_pal⊆raa",
             pal_residues.issubset(raa_residues),
             f"pal={pal_residues}, raa={raa_residues}")


# ============================================================
# 196 軌道の統計量（論文 Section 3.3 の値との照合）
# ============================================================

def test_trajectory_statistics():
    """
    196 軌道 2000 ステップの統計を論文記載値と照合。
    """
    n = 196
    cL0_with_gen = 0
    cL0_even = 0
    cL0_odd = 0
    cL1_with_nondegen = 0
    satisfies_both = 0

    for t in range(2000):
        ps = pair_sums(n)
        c = carry_chain(n)
        L = len(ps)
        has_gen = any(p >= 10 for p in ps)
        has_nondegen = any(p not in (0, 11) for p in ps)
        # 十分条件: ps_i ∈ {10,12,...,18} が存在
        has_sufficient = any(p >= 10 and p != 11 for p in ps)

        if c[-1] == 0 and has_gen:
            cL0_with_gen += 1
            if L % 2 == 0:
                cL0_even += 1
            else:
                cL0_odd += 1
        elif c[-1] == 1 and has_nondegen:
            cL1_with_nondegen += 1

        if has_sufficient:
            satisfies_both += 1

        n = raa(n)

    # 論文記載値
    test("traj_cL0_gen=1169",
         cL0_with_gen == 1169, f"got {cL0_with_gen}")
    test("traj_cL0_even=587",
         cL0_even == 587, f"got {cL0_even}")
    test("traj_cL0_odd=582",
         cL0_odd == 582, f"got {cL0_odd}")
    test("traj_cL1_nondegen=831",
         cL1_with_nondegen == 831, f"got {cL1_with_nondegen}")
    test("traj_total=2000",
         cL0_with_gen + cL1_with_nondegen == 2000,
         f"got {cL0_with_gen + cL1_with_nondegen}")
    test("traj_sufficient_condition_all_2000",
         satisfies_both == 2000,
         f"got {satisfies_both}")


# ============================================================
# Generator / Absorber / Neutral の数え上げ
# ============================================================

def test_generator_absorber_count():
    """
    Definition 2.6 の確認:
    各 base b で generator = absorber = b-1 個、neutral = 1 個
    """
    for base in [2, 3, 5, 10, 16]:
        max_ps = 2 * (base - 1)
        generators = sum(1 for ps in range(max_ps + 1) if ps >= base)
        absorbers = sum(1 for ps in range(max_ps + 1) if ps <= base - 2)
        neutrals = sum(1 for ps in range(max_ps + 1) if ps == base - 1)
        test(f"GA_base{base}_gen={base - 1}",
             generators == base - 1,
             f"got {generators}")
        test(f"GA_base{base}_abs={base - 1}",
             absorbers == base - 1,
             f"got {absorbers}")
        test(f"GA_base{base}_neutral=1",
             neutrals == 1,
             f"got {neutrals}")


# ============================================================
# Theorem 3.7 の追加検証: 異なる基数での完全性
# ============================================================

def test_complete_char_multibase():
    """
    Theorem 3.7 の完全性を base 3, 5 で追加検証。
    全 L=2,3,4 桁の数に対して例外なし。
    """
    for base in [3, 5]:
        violations = 0
        total = 0
        for L in range(2, 5):
            lo = base ** (L - 1)
            hi = base ** L
            for n in range(lo, hi):
                total += 1
                ps = pair_sums(n, base)
                s = raa(n, base)
                pal = is_palindrome(s, base)
                case_I = all(p < base for p in ps)
                case_II = all(p in (0, base + 1) for p in ps)
                if pal != (case_I or case_II):
                    violations += 1
        test(f"Thm3.7_base{base}_exhaustive",
             violations == 0,
             f"{violations}/{total} violations")


# ============================================================
# メイン
# ============================================================

def main():
    global passed, failed

    start = time.time()
    print("=" * 60)
    print("196 Carry Asymmetry — 定理検証")
    print("=" * 60)

    print("\n[Section 1] 基本 RAA 計算")
    test_basic_raa()

    print("\n[Section 2] 定義の検証")
    test_definitions()

    print("\n[Section 3.1] Theorem 3.1 — Carry Asymmetry")
    test_theorem_3_1()

    print("\n[Section 3.1] Proposition 3.2 — c_L=0 非回文")
    test_proposition_3_2()

    print("\n[Section 3.2] Theorem 3.4 — Carry-Free Palindrome (Case I)")
    test_theorem_3_4()

    print("\n[Section 3.2] Theorem 3.5 — Pair-Sum Degeneracy (Necessity)")
    test_theorem_3_5()

    print("\n[Section 3.2] Theorem 3.6 — Pair-Sum Degeneracy (Sufficiency)")
    test_theorem_3_6()

    print("\n[Section 3.2] Theorem 3.7 — Complete Palindrome Characterization")
    test_theorem_3_7()

    print("\n[Section 3.2] Corollary 3.9 — 196 軌道 2000 ステップ")
    test_corollary_3_9()

    print("\n[Section 3.2] Remark 3.8 — Case II 実例")
    test_remark_case_II_example()

    print("\n[Section 4] Proposition 4.1 — キャリー一致確率 ρ")
    test_proposition_4_1()

    print("\n[Section 6] Proposition 6.1 — Carry Markov Chain")
    test_proposition_6_1()

    print("\n[Section 6.2] Base 2 — Sprague 接続")
    test_base_2_sprague()

    print("\n[Section 5] Wall W3 — Modular impossibility")
    test_wall_W3()

    print("\n[Statistics] 196 軌道統計")
    test_trajectory_statistics()

    print("\n[Definitions] Generator/Absorber/Neutral 数え上げ")
    test_generator_absorber_count()

    print("\n[Multibase] Theorem 3.7 異基数完全検証")
    test_complete_char_multibase()

    elapsed = time.time() - start

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print(f"Runtime: {elapsed:.1f}s")
    print("=" * 60)

    if failed > 0:
        print("\n--- Failed tests ---")
        for name, status, detail in test_details:
            if status == "FAIL":
                print(f"  {name}: {detail}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
