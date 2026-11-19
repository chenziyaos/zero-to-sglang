"""量化的数值账：同一块权重、同一批激活，换格式、换缩放粒度以后误差差多少。

只用标准库。FP8 / FP4 的取值表按 OCP 8-bit FP 规范和 OCP MX 规范逐位生成，缩放方式按
sglang 8fabf5e 的 fp8_utils.py（input_to_float8、per_block_cast_to_fp8、ceil_to_ue8m0）
和 GPTQ / AWQ / SmoothQuant 论文的做法重写。

  python3 quant_sim.py                          # 同一块权重，十种格式和粒度的误差与每参数位数
  python3 quant_sim.py --scenario formats       # E4M3 / E5M2 / E2M1 的取值范围和相对步长
  python3 quant_sim.py --scenario activation    # 带离群通道的激活：FP8、INT8、SmoothQuant
  python3 quant_sim.py --scenario awq           # AWQ 按激活幅度放大显著通道，扫 alpha
  python3 quant_sim.py --scenario gptq          # RTN 和 GPTQ（逐列误差补偿）对比，3 bit 和 4 bit
  python3 quant_sim.py --scenario kv            # FP8 / FP4 KV cache：scale 缺省为 1 时什么情况会出事

口径：
  相对误差 = ||W - Q(W)||_F / ||W||_F；输出误差 = ||XW^T - XQ(W)^T||_F / ||XW^T||_F。
  合成数据的分布照 LLM.int8、SmoothQuant 论文报告的现象构造：权重近似高斯、各输出通道幅度
  不同、零星离群值；激活有少数固定的离群通道，幅度比其他通道大一个多数量级。
  数字说明的是量级和排序，不能替代在真实模型上跑评测。
"""
import argparse
import bisect
import math
import random
from typing import Callable, List, Sequence, Tuple

Matrix = List[List[float]]


# ---------------------------------------------------------------------------
# 浮点格式：按位枚举所有正的可表示值
# ---------------------------------------------------------------------------
def fp_grid(ebits: int, mbits: int, bias: int, max_val: float) -> List[float]:
    vals = set()
    for e in range(2 ** ebits):
        for m in range(2 ** mbits):
            if e == 0:
                v = m / 2 ** mbits * 2.0 ** (1 - bias)       # 次正规数
            else:
                v = (1 + m / 2 ** mbits) * 2.0 ** (e - bias)
            if v <= max_val:
                vals.add(v)
    return sorted(vals)


# E4M3FN：没有 inf，S.1111.111 是 NaN，所以最大值是 1.75 × 2^8 = 448
E4M3 = fp_grid(4, 3, 7, 448.0)
# E5M2：保留 IEEE 的 inf/NaN，最大 1.75 × 2^15 = 57344
E5M2 = fp_grid(5, 2, 15, 57344.0)
# E2M1（FP4）：0, 0.5, 1, 1.5, 2, 3, 4, 6
E2M1 = fp_grid(2, 1, 1, 6.0)
FMT = {'e4m3': E4M3, 'e5m2': E5M2, 'e2m1': E2M1}


def round_to_grid(x: float, grid: Sequence[float]) -> float:
    """就近舍入到格点，平局取尾数为偶的那个，超出最大值饱和（sglang 量化前先 clamp）。"""
    a = abs(x)
    if a >= grid[-1]:
        return math.copysign(grid[-1], x)
    i = bisect.bisect_left(grid, a)
    if grid[i] == a:
        return x
    lo, hi = grid[i - 1], grid[i]
    if a - lo < hi - a:
        v = lo
    elif hi - a < a - lo:
        v = hi
    else:
        v = lo if (i - 1) % 2 == 0 else hi                 # 下标奇偶就是尾数最低位
    return math.copysign(v, x)


def ceil_to_ue8m0(x: float) -> float:
    """把缩放因子向上取到 2 的整数次幂（DeepGEMM 的 UE8M0 缩放，fp8_utils.py:2411）。"""
    if x <= 0:
        return 2.0 ** -126
    m, e = math.frexp(x)               # x = m × 2^e，0.5 <= m < 1
    return 2.0 ** (e - 1) if m == 0.5 else 2.0 ** e


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
def make_weight(n: int, k: int, seed: int = 0, outlier_frac: float = 0.001) -> Matrix:
    """近似 LLM 线性层：高斯，按输出通道有幅度差（对数正态），零星 10 倍离群值。"""
    rng = random.Random(seed)
    w = []
    for _ in range(n):
        row_scale = 0.02 * math.exp(rng.gauss(0, 0.4))
        row = [rng.gauss(0, row_scale) for _ in range(k)]
        for j in range(k):
            if rng.random() < outlier_frac:
                row[j] *= 10
        w.append(row)
    return w


def make_act(t: int, k: int, seed: int = 1, n_outlier: int = 4, mag: float = 30.0) -> Tuple[Matrix, List[int]]:
    """激活：每 token 幅度有差异，少数固定通道幅度大 mag 倍（LLM.int8 观察到的现象）。"""
    rng = random.Random(seed)
    cols = rng.sample(range(k), n_outlier)
    x = []
    for _ in range(t):
        tok = math.exp(rng.gauss(0, 0.5))
        row = [rng.gauss(0, tok) for _ in range(k)]
        for c in cols:
            row[c] = rng.gauss(0, 1) * tok * mag + tok * mag * 0.5
        x.append(row)
    return x, cols


# ---------------------------------------------------------------------------
# 量化器：输入矩阵，返回反量化后的矩阵（便于直接比误差）
# ---------------------------------------------------------------------------
def _fp_block(vals: List[float], grid: Sequence[float], scale: float) -> List[float]:
    return [round_to_grid(v / scale, grid) * scale for v in vals]


def amax(vals: Sequence[float]) -> float:
    return max((abs(v) for v in vals), default=0.0) or 1e-12


def fp8_per_tensor(w: Matrix, grid=E4M3) -> Matrix:
    """input_to_float8：scale = amax / 448，整张量一个数。"""
    s = max(amax(r) for r in w) / grid[-1]
    return [_fp_block(r, grid, s) for r in w]


def fp8_per_row(w: Matrix, grid=E4M3) -> Matrix:
    """权重按输出通道（per-channel）、激活按 token（per-token）都是一行一个 scale。"""
    return [_fp_block(r, grid, amax(r) / grid[-1]) for r in w]


def fp8_block(w: Matrix, bn: int = 128, bk: int = 128, ue8m0: bool = False, grid=E4M3) -> Matrix:
    """DeepSeek-V3 的 128×128 块缩放；ue8m0=True 时 scale 取 2 的幂（per_block_cast_to_fp8）。"""
    n, k = len(w), len(w[0])
    out = [[0.0] * k for _ in range(n)]
    for i0 in range(0, n, bn):
        for j0 in range(0, k, bk):
            blk = [w[i][j] for i in range(i0, min(i0 + bn, n)) for j in range(j0, min(j0 + bk, k))]
            s = max(amax(blk), 1e-4) / grid[-1]
            if ue8m0:
                s = ceil_to_ue8m0(s)
            for i in range(i0, min(i0 + bn, n)):
                for j in range(j0, min(j0 + bk, k)):
                    out[i][j] = round_to_grid(w[i][j] / s, grid) * s
    return out


def int_group(w: Matrix, bits: int, group: int, sym: bool) -> Matrix:
    """按输入维分组的整数量化。

    sym=True 对应 GPTQ 的对称格式（Marlin 的 uint4b8 / uint8b128：存无符号数，固定偏置 2^(b-1)）；
    sym=False 对应 AWQ（uint4 + 每组一个零点）。group=-1 表示整行一组（per-channel）。
    """
    maxq = 2 ** bits - 1
    out = []
    for row in w:
        k = len(row)
        g = k if group == -1 else group
        q = []
        for j0 in range(0, k, g):
            seg = row[j0:j0 + g]
            if sym:
                a = amax(seg)
                s = 2 * a / maxq
                z = (maxq + 1) / 2
            else:
                lo, hi = min(min(seg), 0.0), max(max(seg), 0.0)
                s = (hi - lo) / maxq or 1e-12
                z = round(-lo / s)
            q += [(min(max(round(v / s) + z, 0), maxq) - z) * s for v in seg]
        out.append(q)
    return out


def int8_per_tensor(x: Matrix) -> Matrix:
    s = max(amax(r) for r in x) / 127
    return [[max(-127, min(127, round(v / s))) * s for v in r] for r in x]


def int8_per_row(x: Matrix) -> Matrix:
    out = []
    for r in x:
        s = amax(r) / 127
        out.append([max(-127, min(127, round(v / s))) * s for v in r])
    return out


def mxfp4(w: Matrix, block: int = 32) -> Matrix:
    """OCP MXFP4：32 个 E2M1 共享一个 E8M0 缩放，scale = 2^(floor(log2 amax) − 2)，E2M1 的 emax 是 2。"""
    out = []
    for row in w:
        q = []
        for j0 in range(0, len(row), block):
            seg = row[j0:j0 + block]
            a = amax(seg)
            s = 2.0 ** (math.floor(math.log2(a)) - 2)
            q += _fp_block(seg, E2M1, s)
        out.append(q)
    return out


def nvfp4(w: Matrix, block: int = 16) -> Matrix:
    """NVFP4：16 个 E2M1 共享一个 E4M3 缩放，外加整张量一个 FP32 缩放。"""
    g = max(amax(r) for r in w) / (448 * 6)          # 让块 scale 落在 E4M3 范围内
    out = []
    for row in w:
        q = []
        for j0 in range(0, len(row), block):
            seg = row[j0:j0 + block]
            bs = round_to_grid(amax(seg) / 6 / g, E4M3) * g or 1e-12
            q += _fp_block(seg, E2M1, bs)
        out.append(q)
    return out


# ---------------------------------------------------------------------------
# 度量
# ---------------------------------------------------------------------------
def rel_err(a: Matrix, b: Matrix) -> float:
    num = sum((x - y) ** 2 for ra, rb in zip(a, b) for x, y in zip(ra, rb))
    den = sum(x * x for r in a for x in r)
    return math.sqrt(num / den)


def sqnr_db(a: Matrix, b: Matrix) -> float:
    e = rel_err(a, b)
    return -20 * math.log10(e) if e > 0 else float('inf')


def matmul_t(x: Matrix, w: Matrix) -> Matrix:
    """Y = X W^T，W 是 [out, in]，和 nn.Linear 一致。"""
    return [[sum(a * b for a, b in zip(xr, wr)) for wr in w] for xr in x]


# ---------------------------------------------------------------------------
# 每参数位数（含缩放因子的开销）
# ---------------------------------------------------------------------------
def bits_per_param(kind: str, k: int = 4096) -> float:
    return {
        'bf16': 16.0,
        'fp8_tensor': 8.0,
        'fp8_channel': 8 + 32 / k,
        'fp8_block128': 8 + 32 / (128 * 128),
        'int8_channel': 8 + 32 / k,
        'int4_g128_sym': 4 + 16 / 128,               # 每组一个 FP16 scale
        'int4_g128_asym': 4 + 16 / 128 + 4 / 128,    # 再加一个 4 bit 零点
        'mxfp4': 4 + 8 / 32,                         # E8M0 缩放，4.25
        'nvfp4': 4 + 8 / 16,                         # E4M3 缩放，4.5（整张量 FP32 忽略）
    }[kind]


# ---------------------------------------------------------------------------
# 场景
# ---------------------------------------------------------------------------
def scenario_formats():
    print('三种小浮点格式（OCP 规范）')
    print(f'  {"格式":<6} {"正值个数":>8} {"最大值":>10} {"最小正规数":>12} {"最小次正规":>12} {"1 附近步长":>10}')
    for name, g, mn in (('E4M3', E4M3, 2 ** -6), ('E5M2', E5M2, 2 ** -14), ('E2M1', E2M1, 1.0)):
        i = bisect.bisect_left(g, 1.0)
        step = g[i + 1] - g[i]
        print(f'  {name:<6} {len(g) - 1:>8} {g[-1]:>10g} {mn:>12.3g} {g[1]:>12.3g} {step:>10g}')
    print()
    print('  E4M3 每个 2 的幂区间里只有 8 个值，相对步长 1/8 = 12.5%，最坏舍入误差 6.25%。')
    print('  E5M2 多一位指数、少一位尾数：范围大 128 倍，步长翻倍。推理里权重和激活用 E4M3，')
    print('  E5M2 留给梯度。KV cache 两种都能选，E4M3 精度更好，E5M2 不需要 scale 也很难溢出。')
    print()
    print('  E2M1 全部取值：', E2M1, '，只有 8 个非负值，必须配小块缩放才能用。')
    print()
    print('  在 [1, 2) 里均匀取 10000 个数，舍入到各格式后的平均相对误差：')
    xs = [1 + i / 10000 for i in range(10000)]
    for name, g in (('E4M3', E4M3), ('E5M2', E5M2)):
        e = sum(abs(round_to_grid(x, g) - x) / x for x in xs) / len(xs)
        print(f'    {name}: {e * 100:.2f}%')
    xs4 = [0.5 + i / 2000 for i in range(10000)]   # [0.5, 5.5)
    e = sum(abs(round_to_grid(x, E2M1) - x) / x for x in xs4) / len(xs4)
    print(f'    E2M1（[0.5, 5.5)）: {e * 100:.2f}%')


GRAN_CASES: List[Tuple[str, str, Callable[[Matrix], Matrix]]] = [
    ('FP8 per-tensor', 'fp8_tensor', fp8_per_tensor),
    ('FP8 per-channel', 'fp8_channel', fp8_per_row),
    ('FP8 块 128×128', 'fp8_block128', fp8_block),
    ('FP8 块 128×128 UE8M0', 'fp8_block128', lambda w: fp8_block(w, ue8m0=True)),
    ('INT8 per-tensor', 'fp8_tensor', int8_per_tensor),
    ('INT8 per-channel', 'int8_channel', int8_per_row),
    ('INT4 g128 对称（GPTQ）', 'int4_g128_sym', lambda w: int_group(w, 4, 128, True)),
    ('INT4 g128 零点（AWQ）', 'int4_g128_asym', lambda w: int_group(w, 4, 128, False)),
    ('MXFP4 g32', 'mxfp4', mxfp4),
    ('NVFP4 g16', 'nvfp4', nvfp4),
]


def scenario_granularity(n: int = 256, k: int = 512):
    w = make_weight(n, k)
    print(f'合成权重 {n}×{k}（高斯 + 通道幅度差 + 0.1% 的 10 倍离群值）')
    print(f'  {"格式":<24} {"位/参数":>8} {"相对误差":>9} {"SQNR dB":>8}')
    rows = []
    for name, kind, fn in GRAN_CASES:
        q = fn(w)
        e = rel_err(w, q)
        rows.append((name, e))
        print(f'  {name:<24} {bits_per_param(kind):>8.3f} {e * 100:>8.2f}% {sqnr_db(w, q):>8.1f}')
    print()
    print('  读法：FP8 的格点按指数分布，scale 只决定数落在哪一段，相对精度基本不变，所以权重上')
    print('  per-tensor、per-channel、128×128 块差不多；INT8 是均匀格点，per-tensor 时离群值把 scale')
    print('  拉大，小数值丢精度。分布正常时 INT8 per-channel 比 FP8 准，INT8 有 7 位有效精度，E4M3 只有')
    print('  3 位尾数。细粒度缩放对 FP8 真正起作用的地方在激活（见 --scenario activation）。')
    print('  4 bit 档里小块缩放决定一切：零点比对称多用一个格点，NVFP4 块只有 16 个元素、缩放是 E4M3，')
    print('  比 MXFP4 的 32 元素、2 的幂缩放细，误差最低。')
    return rows


def _w8a8(x: Matrix, w: Matrix, qx: Callable, qw: Callable) -> float:
    y = matmul_t(x, w)
    return rel_err(y, matmul_t(qx(x), qw(w)))


def smooth(x: Matrix, w: Matrix, alpha: float) -> Tuple[Matrix, Matrix]:
    """SmoothQuant：s_j = max|X_j|^α / max|W_j|^(1−α)，X 除以 s、W 乘以 s，乘积不变。"""
    k = len(x[0])
    ax = [max(abs(r[j]) for r in x) for j in range(k)]
    aw = [max(abs(r[j]) for r in w) for j in range(k)]
    s = [max(ax[j], 1e-5) ** alpha / max(aw[j], 1e-5) ** (1 - alpha) for j in range(k)]
    return [[v / s[j] for j, v in enumerate(r)] for r in x], [[v * s[j] for j, v in enumerate(r)] for r in w]


def scenario_activation(t: int = 32, k: int = 512, n: int = 128):
    x, cols = make_act(t, k)
    w = make_weight(n, k, seed=2)
    ax = [max(abs(r[j]) for r in x) for j in range(k)]
    normal = sorted(ax[j] for j in range(k) if j not in cols)
    print(f'合成激活 {t}×{k}，离群通道 {sorted(cols)}，最大幅度 {max(ax):.0f}，普通通道中位数 {normal[len(normal) // 2]:.1f}')
    print()
    print(f'  {"方案":<40} {"输出误差":>8}')
    cases = [
        ('W8A8 INT8，激活 per-tensor', lambda: _w8a8(x, w, int8_per_tensor, int8_per_row)),
        ('W8A8 INT8，激活 per-token', lambda: _w8a8(x, w, int8_per_row, int8_per_row)),
    ]
    for a in (0.5, 0.75):
        xs, ws = smooth(x, w, a)
        cases.append((f'W8A8 INT8 + SmoothQuant α={a}，per-tensor',
                      lambda xs=xs, ws=ws: rel_err(matmul_t(x, w), matmul_t(int8_per_tensor(xs), int8_per_row(ws)))))
    cases += [
        ('W8A8 FP8，激活 per-tensor', lambda: _w8a8(x, w, fp8_per_tensor, fp8_per_row)),
        ('W8A8 FP8，激活 per-token（sglang 默认）', lambda: _w8a8(x, w, fp8_per_row, fp8_per_row)),
        ('W8A8 FP8，激活 1×128 + 权重 128×128', lambda: _w8a8(x, w, lambda m: fp8_block(m, 1, 128), fp8_block)),
        ('W8A16 FP8（Marlin，激活不量化）', lambda: _w8a8(x, w, lambda m: m, fp8_per_row)),
    ]
    out = {}
    for name, fn in cases:
        e = fn()
        out[name] = e
        print(f'  {name:<40} {e * 100:>7.2f}%')
    print()
    print('  INT8 是均匀格点：离群通道把 scale 拉大三十多倍，普通通道只剩 2 到 3 个有效 bit。')
    print('  SmoothQuant 把一部分幅度挪给权重，per-tensor 的 INT8 也能用。FP8 的格点按指数分布，')
    print('  小数值的相对精度不随 scale 变差，所以 per-token 动态 FP8 不做平滑也够用。')
    return out


def scenario_awq(t: int = 64, k: int = 512, n: int = 128):
    x, cols = make_act(t, k, seed=3, n_outlier=5, mag=20)
    w = make_weight(n, k, seed=4)
    y = matmul_t(x, w)
    sx = [sum(abs(r[j]) for r in x) / t for j in range(k)]
    print(f'INT4 g128（零点）权重 {n}×{k}，校准激活 {t} 个 token，显著通道 {sorted(cols)}')
    print()

    def q4(m):
        return int_group(m, 4, 128, False)

    base = rel_err(y, matmul_t(x, q4(w)))
    print(f'  RTN（直接舍入）                     输出误差 {base * 100:.2f}%')
    # 论文第 3.1 节：显著通道留 FP16
    keep = sorted(range(k), key=lambda j: -sx[j])[:max(1, k // 100)]
    qw = q4(w)
    mixed = [[w[i][j] if j in keep else qw[i][j] for j in range(k)] for i in range(n)]
    print(f'  1% 显著通道保留 FP16（混合精度）     输出误差 {rel_err(y, matmul_t(x, mixed)) * 100:.2f}%')
    best = (base, 0.0)
    print('  按激活幅度缩放 s_j = mean|X_j|^α，W 乘 s、X 除 s：')
    for i in range(0, 11):
        a = i / 10
        s = [max(v, 1e-4) ** a for v in sx]
        mean_s = math.sqrt(max(s) * min(s))
        s = [v / mean_s for v in s]
        ws = [[v * s[j] for j, v in enumerate(r)] for r in w]
        xs = [[v / s[j] for j, v in enumerate(r)] for r in x]
        e = rel_err(y, matmul_t(xs, q4(ws)))
        best = min(best, (e, a))
        print(f'    α={a:.1f}  输出误差 {e * 100:.2f}%')
    print(f'  最好的 α={best[1]:.1f}，误差从 {base * 100:.2f}% 降到 {best[0] * 100:.2f}%，权重仍然全是 INT4。')
    print('  α 太大时显著通道被放得太大，同组其他权重的 scale 跟着变粗，误差又回升，所以要搜。')
    return base, best


# ---- GPTQ：纯 Python 的矩阵工具 ----
def _gram(x: Matrix) -> Matrix:
    k = len(x[0])
    h = [[0.0] * k for _ in range(k)]
    for r in x:
        for i in range(k):
            ri = r[i]
            if ri == 0:
                continue
            hi = h[i]
            for j in range(i, k):
                hi[j] += ri * r[j]
    for i in range(k):
        for j in range(i):
            h[i][j] = h[j][i]
    return h


def _inv(a: Matrix) -> Matrix:
    n = len(a)
    m = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(a)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        m[c], m[p] = m[p], m[c]
        pv = m[c][c]
        m[c] = [v / pv for v in m[c]]
        for r in range(n):
            if r != c and m[r][c] != 0:
                f = m[r][c]
                mr, mc = m[r], m[c]
                m[r] = [u - f * v for u, v in zip(mr, mc)]
    return [row[n:] for row in m]


def _chol_upper(a: Matrix) -> Matrix:
    """返回上三角 U，满足 U^T U = A。"""
    n = len(a)
    low = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(low[i][t] * low[j][t] for t in range(j))
            if i == j:
                low[i][j] = math.sqrt(max(a[i][i] - s, 1e-12))
            else:
                low[i][j] = (a[i][j] - s) / low[j][j]
    return [[low[j][i] for j in range(n)] for i in range(n)]


def _qsym(v: float, s: float, maxq: int) -> float:
    z = (maxq + 1) / 2
    return (min(max(round(v / s) + z, 0), maxq) - z) * s


def gptq(w: Matrix, h: Matrix, bits: int, group: int = -1, act_order: bool = False,
         damp: float = 0.01) -> Matrix:
    """GPTQ 算法 1：按列量化，误差乘 H^-1 的 Cholesky 行，摊到还没量化的列上。"""
    k = len(w[0])
    maxq = 2 ** bits - 1
    perm = list(range(k))
    if act_order:                                   # desc_act：按 H 对角线从大到小处理
        perm.sort(key=lambda j: -h[j][j])
    hp = [[h[i][j] for j in perm] for i in perm]
    d = damp * sum(hp[i][i] for i in range(k)) / k
    for i in range(k):
        hp[i][i] += d
    u = _chol_upper(_inv(hp))
    g = k if group == -1 else group
    out = []
    for row in w:
        r = [row[j] for j in perm]
        q = [0.0] * k
        s = 1.0
        for j in range(k):
            if j % g == 0:                          # 组的 scale 用补偿后的当前权重算
                s = 2 * amax(r[j:j + g]) / maxq
            q[j] = _qsym(r[j], s, maxq)
            err = (r[j] - q[j]) / u[j][j]
            uj = u[j]
            for t in range(j + 1, k):
                r[t] -= err * uj[t]
        inv = [0.0] * k
        for pos, j in enumerate(perm):
            inv[j] = q[pos]
        out.append(inv)
    return out


def out_err_h(w: Matrix, q: Matrix, h: Matrix) -> float:
    """用 H = X^T X 算输出误差：sum_i d_i^T H d_i / sum_i w_i^T H w_i。"""
    def quad(v):
        return sum(v[i] * sum(h[i][j] * v[j] for j in range(len(v))) for i in range(len(v)))
    num = sum(quad([a - b for a, b in zip(wr, qr)]) for wr, qr in zip(w, q))
    den = sum(quad(wr) for wr in w)
    return math.sqrt(num / den)


def scenario_gptq(n: int = 32, k: int = 128, t: int = 256):
    x, cols = make_act(t, k, seed=5, n_outlier=3, mag=10)
    w = make_weight(n, k, seed=6, outlier_frac=0.0)
    h = _gram(x)
    print(f'权重 {n}×{k}，校准 {t} 个 token（{len(cols)} 个离群通道），对称量化，H = X^T X，阻尼 1%')
    print(f'  {"bit":>3} {"分组":>6} {"RTN":>8} {"GPTQ":>8} {"GPTQ+desc_act":>14}')
    res = {}
    for bits in (4, 3):
        for group in (-1, 32):
            rtn = int_group(w, bits, group, True)
            e0 = out_err_h(w, rtn, h)
            e1 = out_err_h(w, gptq(w, h, bits, group), h)
            e2 = out_err_h(w, gptq(w, h, bits, group, act_order=True), h)
            res[(bits, group)] = (e0, e1, e2)
            gname = '整行' if group == -1 else f'g{group}'
            print(f'  {bits:>3} {gname:>6} {e0 * 100:>7.2f}% {e1 * 100:>7.2f}% {e2 * 100:>13.2f}%')
    print()
    print('  GPTQ 看的是输出误差：先量化的列留下的误差，按 H^-1 摊给后面和它相关的列去抵消。')
    print('  desc_act 先处理激活能量大的列，让它们的误差有更多列来摊。代价是推理时')
    print('  组和列的顺序对不上，sglang 的 Marlin 需要额外的 g_idx 重排（gptq.py 的 desc_act）。')
    return res


# ---- KV cache ----
def make_kv(seq: int, d: int, seed: int, k_outlier: float) -> Tuple[Matrix, Matrix, Matrix]:
    """K 有少数固定的大通道（KIVI 观察到 key 按通道离群），V 按 token 波动。"""
    rng = random.Random(seed)
    big = rng.sample(range(d), 4)
    k = [[rng.gauss(0, 1) * (k_outlier if j in big else 1.0) for j in range(d)] for _ in range(seq)]
    v = [[rng.gauss(0, math.exp(rng.gauss(0, 0.3))) for _ in range(d)] for _ in range(seq)]
    q = [[rng.gauss(0, 1) for _ in range(d)] for _ in range(8)]
    return q, k, v


def attention(q: Matrix, k: Matrix, v: Matrix) -> Matrix:
    d = len(q[0])
    out = []
    for qr in q:
        s = [sum(a * b for a, b in zip(qr, kr)) / math.sqrt(d) for kr in k]
        m = max(s)
        p = [math.exp(x - m) for x in s]
        z = sum(p)
        out.append([sum(p[t] * v[t][j] for t in range(len(k))) / z for j in range(len(v[0]))])
    return out


def kv_fp8(m: Matrix, grid, scale: float) -> Matrix:
    """sglang 的 FP8 KV：整层一个 k_scale / v_scale（kv_cache.py 只支持 per-tensor）。"""
    return [_fp_block(r, grid, scale) for r in m]


def kv_fp4_block16(m: Matrix) -> Matrix:
    """每 token 沿 head_dim 每 16 个元素一个缩放（nvfp4 / fp4_mx_block16 的块大小）。"""
    return nvfp4(m, 16)


def scenario_kv(seq: int = 256, d: int = 128):
    print(f'单头 attention，head_dim {d}，上下文 {seq}，8 个 query；比较 attention 输出的相对误差')
    print(f'  {"K 离群通道幅度":>14} {"E4M3 scale=1":>13} {"E4M3 校准":>10} {"E5M2 scale=1":>13} {"FP4 块16":>9}')
    res = {}
    for mag in (8, 60, 600):
        q, k, v = make_kv(seq, d, 7, mag)
        ref = attention(q, k, v)
        ka, va = max(amax(r) for r in k), max(amax(r) for r in v)
        row = (
            rel_err(ref, attention(q, kv_fp8(k, E4M3, 1.0), kv_fp8(v, E4M3, 1.0))),
            rel_err(ref, attention(q, kv_fp8(k, E4M3, ka / 448), kv_fp8(v, E4M3, va / 448))),
            rel_err(ref, attention(q, kv_fp8(k, E5M2, 1.0), kv_fp8(v, E5M2, 1.0))),
            rel_err(ref, attention(q, kv_fp4_block16(k), kv_fp4_block16(v))),
        )
        res[mag] = row
        print(f'  {mag:>14} ' + ' '.join(f'{e * 100:>{w}.2f}%' for e, w in zip(row, (12, 9, 12, 8))))
    print()
    print('  scale=1 是没给 --quantization-param-path、checkpoint 里也没有 k_scale 时的默认值。')
    print('  K 的离群通道超过 448 以后 E4M3 直接饱和，误差跳一个量级；E5M2 范围到 57344，不怕饱和，')
    print('  代价是平时多一倍舍入误差。FP4 块 16 每块自带缩放，不怕离群值，但只有 8 个非负格点。')
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', default='granularity',
                    choices=['granularity', 'formats', 'activation', 'awq', 'gptq', 'kv'])
    a = ap.parse_args()
    {'granularity': scenario_granularity, 'formats': scenario_formats,
     'activation': scenario_activation, 'awq': scenario_awq, 'gptq': scenario_gptq,
     'kv': scenario_kv}[a.scenario]()


if __name__ == '__main__':
    main()
