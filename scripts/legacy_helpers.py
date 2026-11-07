import math,statistics
def _ols_optimal_timing_weights(papers: list[dict]) -> dict:
    """
    OLS 最优 Timing 权重推导。

    推导过程：
      设权重为 relative_pos 的线性函数：w_j = a * rel_pos_j + b
      论文 i 的加权均分：ŝ_i = Σ_j w_j * r_ij / Σ_j w_j
      目标：min_{a,b} Σ_i (ŝ_i - y_i)²，其中 y_i ∈ {0,1} 为 AC 决策

    这等价于：固定 b=1（归一化约束），只对 a 做线性搜索。
    闭合解：a* = argmin_{a} Σ_i (ŝ_i(a) - y_i)²

    由于 ŝ_i(a) 是 a 的有理函数（非线性），用网格搜索 + 精细二分法找 a*。
    实践中搜索范围 a ∈ [-2, 2]，b 固定为 1。

    最终权重归一化使均值=1（保持评分量级）。
    """
    paper_data = []
    for paper in papers:
        rps = [r['relative_pos'] for r in paper['reviews']]
        rats = [r['rating'] for r in paper['reviews']]
        dec = float(paper['decision_binary'])
        paper_data.append((rps, rats, dec))

    def paper_score(rps, rats, a, b=1.0):
        """给定 a, b，计算一篇论文的加权均分。"""
        ws = [a * rp + b for rp in rps]
        if any((w <= 0 for w in ws)):
            return None
        total_w = sum(ws)
        return sum((r * w for r, w in zip(rats, ws))) / total_w

    def mse(a):
        """计算 a 对应的全集 MSE。"""
        total = 0.0
        n = 0
        for rps, rats, dec in paper_data:
            s = paper_score(rps, rats, a)
            if s is None:
                continue
            total += (s - dec) ** 2
            n += 1
        return total / n if n > 0 else float('inf')
    best_a = 0.0
    best_mse = mse(0.0)
    grid = [i * 0.1 for i in range(-20, 21)]
    for a in grid:
        m = mse(a)
        if m < best_mse:
            best_mse = m
            best_a = a
    lo, hi = (best_a - 0.1, best_a + 0.1)
    for _ in range(20):
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        if mse(m1) < mse(m2):
            hi = m2
        else:
            lo = m1
    a_star = (lo + hi) / 2
    b_star = 1.0
    rep_points = {'Early': 0.88, 'Mid': 0.42, 'Late': 0.05}
    raw_weights = {k: a_star * v + b_star for k, v in rep_points.items()}
    w_mean = sum(raw_weights.values()) / len(raw_weights)
    opt_weights = {k: v / w_mean for k, v in raw_weights.items()}
    return {'a_star': round(a_star, 6), 'b_star': round(b_star, 6), 'mse_uniform': round(mse(0.0), 6), 'mse_optimal': round(mse(a_star), 6), 'mse_reduction_pct': round((mse(0.0) - mse(a_star)) / mse(0.0) * 100, 4), 'weights': {k: round(v, 4) for k, v in opt_weights.items()}, 'formula': f'w(rel_pos) = {a_star:.4f} * rel_pos + {b_star:.4f}  [归一化后]', 'interpretation': f"OLS 最优线性权重函数：w(t) = {a_star:.4f}·t + 1.0，t=relative_pos（0=最晚，1=最早）。{('a>0 → 越早提交权重越高' if a_star > 0 else 'a<0 → 越晚提交权重越高')}，MSE 降低 {(mse(0.0) - mse(a_star)) / mse(0.0) * 100:.4f}%"}
