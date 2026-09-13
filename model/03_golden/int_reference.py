"""Golden integer reference cho eMamba-MARS — chạy model bằng SỐ NGUYÊN thuần.

Khác eval_int8.py (float fake-quant), file này tính đúng như phần cứng: mọi tensor là
mã INT, matmul = MAC nguyên, requant = DỊCH PHẢI (scale pow2), phi tuyến = giá trị thực
trên mã (LUT trên HW), SSM giữ state nhiều bit + dịch theo scale dA.

Mục đích: (1) đặc tả bit-chính-xác cho RTL, (2) sinh golden vectors, (3) export hằng số.
Load best.pt + calib (numpy thuần, không cần torch).

    python int_reference.py --ckpt results_qat/best.pt --calib results_qat/calib_paper.pt \
        --hidden 80 --n 512        # đo RMSE integer-reference trên test
"""
import argparse, zipfile, pickle, math, os
import numpy as np
from collections import OrderedDict

QM = 127


def load_pt(path):
    zf = zipfile.ZipFile(path); pkl = [n for n in zf.namelist() if n.endswith("data.pkl")][0]
    pre = pkl[:-9]
    DT = {'FloatStorage': np.float32, 'DoubleStorage': np.float64, 'LongStorage': np.int64,
          'IntStorage': np.int32, 'HalfStorage': np.float16, 'BoolStorage': np.bool_,
          'ByteStorage': np.uint8, 'CharStorage': np.int8, 'ShortStorage': np.int16}

    def rt(s, o, sz, st, rg, h=None):
        n = int(np.prod(sz)) if len(sz) else 1
        return np.array(s[o:o + n]).reshape(sz)

    class U(pickle.Unpickler):
        def find_class(self, m, n):
            if m == 'torch._utils' and n == '_rebuild_tensor_v2': return rt
            if m == 'torch._utils' and n == '_rebuild_parameter': return lambda d, r, h: d
            if m == 'collections' and n == 'OrderedDict': return OrderedDict
            if m == 'torch' and n.endswith('Storage'): return n
            return super().find_class(m, n)

        def persistent_load(self, pid):
            _, st, key, _, _ = pid
            return np.frombuffer(zf.read(f"{pre.rstrip('/')}/data/{key}"), dtype=DT[st]).copy()

    return U(zf.open(pkl)).load()


def pow2(s):
    return 2.0 ** round(math.log2(max(float(s), 1e-12)))


def clampq(v, bits=8):
    q = (1 << (bits - 1)) - 1
    return np.clip(np.round(v), -q, q)


S_EXPIN = 2.0 ** -5      # scale input exp PWL
S_H = 2.0 ** -3          # scale state SSM INT17 (đã quét tối ưu)
def e(s): return round(math.log2(s))
def rsh(x, s):           # round-half-up arithmetic shift (s>0 phải, s<0 trái)
    x = np.asarray(x, np.float64)
    if s > 0: return np.floor((x + (1 << (s - 1))) / (1 << s))
    if s < 0: return x * (1 << (-s))
    return x
def cl(x, b): q = (1 << (b - 1)) - 1; return np.clip(x, -q - 1, q)
def _silu(x): return x / (1 + np.exp(-x))


def fit_knots(f, lo, hi, tol, rel=False, maxk=200):
    k = [float(lo), float(hi)]
    for _ in range(maxk):
        w = (0.0, -1, None)
        for i in range(len(k) - 1):
            a, b = k[i], k[i + 1]; xs = np.linspace(a, b, 33)
            ip = f(a) + (xs - a) / (b - a) * (f(b) - f(a)); er = np.abs(f(xs) - ip)
            if rel: er = er / np.maximum(np.abs(f(xs)), 1e-6)
            j = int(er.argmax())
            if er[j] > w[0]: w = (er[j], i, float(xs[j]))
        if w[0] <= tol: break
        k.insert(w[1] + 1, w[2])
    return np.array(sorted(set(k)))


def fit_knots_n(f, lo, hi, N, rel=False):         # đặt ĐÚNG N đoạn (greedy) — khớp isa/gen_pwl_nl.py
    k = [float(lo), float(hi)]
    while len(k) - 1 < N:
        w = (-1.0, -1, None)
        for i in range(len(k) - 1):
            a, b = k[i], k[i + 1]; xs = np.linspace(a, b, 65)
            ip = f(a) + (xs - a) / (b - a) * (f(b) - f(a)); er = np.abs(f(xs) - ip)
            if rel: er = er / np.maximum(np.abs(f(xs)), 1e-6)
            j = int(er.argmax())
            if er[j] > w[0]: w = (er[j], i, float(xs[j]))
        k.insert(w[1] + 1, w[2])
    return np.array(sorted(set(k)))


def _exp_tab():                                   # exp PWL 11 đoạn [-4,1] (CHỐT — thay 24)
    kn = fit_knots_n(np.exp, -4.0, 1.0, 11, rel=True); n = len(kn) - 1
    bp = np.round(kn / S_EXPIN).astype(np.int64); sl = []; ic = []
    for i in range(n):
        a, b = kn[i], kn[i + 1]; ya, yb = np.exp(a) * 128, np.exp(b) * 128
        m = (yb - ya) / (b - a)
        sl.append(round(m * S_EXPIN * 128)); ic.append(round((ya - m * a) * 128))
    return bp, np.array(sl), np.array(ic)
_EXP_BP, _EXP_SL, _EXP_IC = _exp_tab()


def exp_pwl(x):                                   # INT8 dain → Ā UINT8 [0,127] (PWL, khớp piecewise_lut)
    x = np.asarray(x, np.int64)
    seg = np.clip(np.searchsorted(_EXP_BP, x, "right") - 1, 0, len(_EXP_SL) - 1)
    y = np.floor((_EXP_SL[seg] * x + _EXP_IC[seg] + 64) / 128.0)  # (sl·x+ic+2^6)>>7 round-half-up (KHỚP piecewise_lut.v)
    # ÉP 127 (thay 128): để exp tái dùng CHUNG requant_n INT8 [-128,127] trên HW (rangedet+array_dsp).
    #   Accuracy ≈ y hệt (MARS ΔMAE +0.003cm); requant_n thuần == golden cap127 (0/256 toàn dải x).
    _hi = int(os.environ.get("EXP_HI") or "127")              # TEST: EXP_HI=128 để khớp piecewise_lut (HI_CONST=128)
    y = np.where(x < _EXP_BP[0], 0, y); y = np.where(x > _EXP_BP[-1], _hi, y)
    return np.clip(y, 0, _hi)


def silu_lut256(real_code, s_real, s_in, s_out):  # CHỐT: PWL 17 đoạn [-7,7] round-half-up (= rangedet+array_dsp+requant_n HW), thay 256-LUT
    code = cl(rsh(real_code, e(s_in) - e(s_real)), 8).astype(np.int64)
    kn = fit_knots_n(_silu, -7.0, 7.0, 17); bp = np.round(kn / s_in)
    if os.environ.get("SILU_BP_INT8", "0") == "1": bp = np.clip(bp, -128, 127)  # khop silu_rom INT8 cua RTL (bp bi kep ±127)
    bp = bp.astype(np.int64)  # BP mac dinh 16-bit; SILU_BP_INT8=1 -> kep INT8 de bit-exact voi phan cung
    sl = []; ic = []
    for i in range(17):
        a, b = kn[i], kn[i + 1]; ya, yb = _silu(a) / s_out, _silu(b) / s_out
        m = (yb - ya) / (b - a); sl.append(round(m * s_in * 128)); ic.append(round((ya - m * a) * 128))
    sl = np.array(sl); ic = np.array(ic)
    seg = np.clip(np.searchsorted(bp, code, "right") - 1, 0, 16)
    y = np.floor((sl[seg] * code + ic[seg] + 64) / 128.0)   # (sl·x+ic+2^6)>>7 round-half-up
    return clampq(y).astype(np.int64)


class IntModel:
    """Mọi hằng số tính sẵn (pow2 scale, weight codes, bias_int). Forward = số nguyên."""

    def __init__(self, ckpt, calibf):
        self.W = {k: np.asarray(v, np.float64) for k, v in load_pt(ckpt).items()}
        ck = load_pt(calibf); self.calib = ck["calib"]; self.cfg = ck["cfg"]
        self.da_s = float(self.cfg.get("da_scale", 0)) or 2.0 ** -7
        self.state_bits = int(self.cfg.get("state_bits", 24))

    def asc(self, name, bits=8):                       # activation scale (pow2)
        st = self.calib[name]; thr = st.get("thr_fixed")
        thr = float(thr) if thr is not None else float(st["absmax"])
        return pow2(thr / ((1 << (bits - 1)) - 1))

    def wq(self, key):                                  # weight: codes + pow2 scale (per-tensor)
        Wm = self.W[key]
        _p = float(os.environ.get("WQ_PCT", "0"))       # TEST: WQ_PCT=99 → clip weight về phân vị 99 thay absmax
        _thr = np.percentile(np.abs(Wm), _p) if _p > 0 else np.abs(Wm).max()
        s = pow2(_thr / QM)
        return clampq(Wm / s), s

    def linear(self, qx, sx, key, bkey, sout):
        """MAC nguyên + bias INT16 + requant round-half-up >> + clamp[-128,127] (KHỚP RTL)."""
        qw, sw = self.wq(key)                            # (out,in)
        acc = qx @ qw.T                                  # INT32
        if bkey is not None and bkey in self.W:
            _bb = int(os.environ.get("BIAS_BITS", "16"))            # TEST: BIAS_BITS=32 để dò tràn INT16
            _bi = np.round(self.W[bkey] / (sx * sw))
            if os.environ.get("BIAS_DIAG") and np.abs(_bi).max() > 32767:
                print(f"  BIAS OVF {bkey}: max|bi|={np.abs(_bi).max():.0f} (>32767, x{np.abs(_bi).max()/32767:.1f}) sx*sw=2^{round(math.log2(sx*sw))}")
            acc = acc + cl(_bi, _bb)   # bias INT16 vào accumulator
        k = round(math.log2(sout / (sx * sw)))           # số bit dịch
        return cl(rsh(acc, k), 8), k                     # (acc+2^(k-1))>>k, clamp[-128,127]

    # ---- các phi tuyến: tính trên giá trị thực của mã rồi requant (HW dùng LUT) ----
    def silu_lut(self, qx, sx, sout):
        real = qx * sx
        return clampq((real / (1 + np.exp(-real))) / sout)

    def forward(self, Xn):                               # Xn: (N,8,8,5) đã chuẩn hoá (thực)
        W = self.W; N = Xn.shape[0]
        s_in = self.asc("a_in")
        q = clampq(Xn / s_in)                            # input → INT8
        # patchify (gom mã)
        t = q.reshape(N, 4, 2, 4, 2, 5).transpose(0, 1, 3, 5, 2, 4).reshape(N, 16, 20)
        h, _ = self.linear(t, s_in, "patch_embed.embed.weight", "patch_embed.embed.bias", self.asc("embed.out_q"))
        s_tok = self.asc("embed.out_q")
        for b in range(self.cfg["blocks"]):
            h, s_tok = self.block(h, s_tok, b)
        return self.head(h, s_tok)

    def block(self, x, sx, b):
        W = self.W; p = f"blocks.{b}."
        res, sres = x, sx
        # RangeNorm INTEGER (khớp range_norm.v): mean (sum·6554)>>17, divider sign-mag Q1.15
        s_n = self.asc(p + "a_norm")
        s_gam = pow2(np.abs(W[p + "norm.gamma"]).max() / 127)
        gam = clampq(W[p + "norm.gamma"] / s_gam)                 # γ INT8
        bet = np.round(W[p + "norm.beta"] / s_gam)                # β INT16 (scale s_gam)
        summ = x.sum(-1, keepdims=True)                          # Σ 20 code
        mean = np.floor((summ * 6554 + 65536) / 131072.0)        # (sum·6554+2^16)>>17
        cent = x - mean
        rng = np.maximum(cent.max(-1, keepdims=True) - cent.min(-1, keepdims=True), 1)
        num = cent * 32768
        _SH = int(os.environ.get("RNORM_RECIP_SH", "0"))            # 0 = divider gốc; >0 = recip-LUT
        _NOCORR = os.environ.get("RNORM_NOCORR", "0") == "1"        # 1 = KHỚP range_norm_recip.v (ceil, KHÔNG correction)
        if _SH > 0:                                                 # nhân-nghịch-đảo
            _RMAX = 1024
            _rc = (np.arange(_RMAX, dtype=np.int64)); _rc[0] = 1
            _r = np.clip(rng.astype(np.int64), 1, _RMAX - 1)
            _an = np.abs(num).astype(np.int64)
            if _NOCORR:                                             # = range_norm_recip.v: recip=ceil(2^SH/r), KHÔNG correction
                _recip = (((1 << _SH) + _rc - 1) // _rc).astype(np.int64)
                _q = (_an * _recip[_r]) >> _SH
            else:                                                   # round-up + 1 correction ⇒ BIT-EXACT divider
                _recip = ((1 << _SH) // _rc + 1).astype(np.int64)
                _q = (_an * _recip[_r]) >> _SH
                _q = _q - ((_q * _r) > _an)
            quot = np.clip(np.sign(num) * _q, -32768, 32767)
        else:
            quot = np.clip(np.sign(num) * (np.abs(num) // rng), -32768, 32767)   # divider sign-magnitude
        raw = np.floor((gam * quot + 16384) / 32768.0) + bet     # γ·quot>>15 + β
        qn = cl(rsh(raw, e(s_n) - e(s_gam)), 8)                   # GAMMA_SHIFT → INT8
        # in_proj
        s_xz = self.asc(p + "in_proj.out_q")
        qxz, _ = self.linear(qn, s_n, p + "in_proj.weight", p + "in_proj.bias", s_xz)
        qxx, qz = qxz[..., :40], qxz[..., 40:]
        # conv1d depthwise (MAC nguyên) → requant về scale silu input rồi SiLU-LUT
        cw, csw = self.wq(p + "conv1d.weight")           # (40,1,4)
        L = qxx.shape[1]; xt = np.pad(qxx.transpose(0, 2, 1), ((0, 0), (0, 0), (3, 3)))
        cacc = np.zeros((qxx.shape[0], 40, L))
        for tt in range(L):
            for k in range(4): cacc[:, :, tt] += cw[:, 0, k] * xt[:, :, tt + k]
        b_int = np.round(W[p + "conv1d.bias"] / (s_xz * csw))
        cacc = (cacc + b_int[None, :, None]).transpose(0, 2, 1)   # acc, scale s_xz*csw
        s_sl = self.asc(p + "a_silu1")
        qxx = silu_lut256(cacc, s_xz * csw, 2.0 ** -4, s_sl)   # conv→INT8(2^-4)→SiLU 256-LUT
        # delta = ReLU(dt_proj(dt_in(xx))) — 2 BƯỚC đúng model: dt_in→cắt INT8 (dt_in.out_q)→dt_proj
        s_dti = self.asc(p + "dt_in.out_q")
        qdt, _ = self.linear(qxx, s_sl, p + "dt_in.weight", p + "dt_in.bias", s_dti)
        s_del = self.asc(p + "a_delta")
        qd_acc, _ = self.linear(qdt, s_dti, p + "dt_proj.weight", p + "dt_proj.bias", s_del)
        qdelta = np.maximum(qd_acc, 0)
        # B, C
        s_B = self.asc(p + "B_proj.out_q"); s_C = self.asc(p + "C_proj.out_q")
        qB, _ = self.linear(qxx, s_sl, p + "B_proj.weight", None, s_B)
        qC, _ = self.linear(qxx, s_sl, p + "C_proj.weight", None, s_C)
        # ── SSM INTEGER-ADD (khớp ssm_core.v): term1=Ā·h + term2=B_bar·x, h24>>7 ──
        A = -np.exp(np.minimum(W[p + "A_log"], math.log(4.0)))
        sA = pow2(np.abs(A).max() / QM); qA = clampq(A / sA)         # (40,8) INT8
        D = W[p + "D"]; sD = pow2(np.abs(D).max() / 32767); qD = cl(np.round(D / sD), 16)  # CLAMP INT16 (khớp per16/RTL D_param; pow2 làm scale → qD có thể vượt 32767)
        s_y = self.asc(p + "a_y"); s_h24 = (2.0 ** -7) * S_H
        DA = e(S_EXPIN) - e(s_del) - e(sA)               # δ·A → exp input (scale 2^-5)
        DB = e(s_h24 / s_sl) - e(s_del) - e(s_B)         # δ·B → B_bar (term2 ↔ term1)
        DX = e(s_C * s_h24) - e(sD) - e(s_sl)            # D·x → y_acc
        Y = e(s_y) - e(s_C * s_h24)                      # y_acc → INT8
        Bn = qxx.shape[0]; hh = np.zeros((Bn, 40, 8)); ys = []
        for tt in range(L):
            xxt = qxx[:, tt]; dlt = qdelta[:, tt]; yacc = np.zeros((Bn, 40))
            for n in range(8):
                dain = cl(rsh(dlt * qA[:, n], DA), 8)                # exp input INT8
                Abar = exp_pwl(dain)                                # exp PWL 11 đoạn (khớp piecewise_lut N_SEG=11)
                Bbar = cl(rsh(dlt * qB[:, tt, n:n + 1], DB), 8)      # (B,40) INT8
                h24 = cl(Abar * hh[:, :, n] + Bbar * xxt, 24)        # INT24 scale 2^-7·s_h
                yacc = yacc + qC[:, tt, n:n + 1] * h24               # scale s_C·2^-7·s_h
                hh[:, :, n] = cl(rsh(h24, 7), 17)                    # store INT17 scale s_h
            yacc = yacc + rsh(qD * xxt, DX)                          # D·x aligned vào y_acc
            ys.append(cl(rsh(yacc, Y), 8))
        qy = np.stack(ys, 1)                              # INT8 scale s_y
        # gate: y · SiLU(z) → requant round-half-up (GATING_SHIFT)
        s_gz = self.asc(p + "a_gatez"); s_g = self.asc(p + "a_gate")
        qgz = silu_lut256(qz, s_xz, s_xz, s_gz)          # z đã INT8 (s_xz) → SiLU 256-LUT
        qg = cl(rsh(qy * qgz, e(s_g) - e(s_y) - e(s_gz)), 8)
        # out_proj + residual (align res về s_bo rồi cộng, clamp[-128,127])
        s_bo = self.asc(p + "a_block")
        qo_acc, _ = self.linear(qg, s_g, p + "out_proj.weight", p + "out_proj.bias", s_bo)
        out = cl(qo_acc + rsh(res, e(s_bo) - e(sres)), 8)
        return out, s_bo

    def head(self, x, sx):
        W = self.W
        # mean-pool dọc token
        s_p = self.asc("head.a_pool")
        pooled = (x * sx).mean(1)
        qp = cl(np.floor(pooled / s_p + 0.5), 8)          # round-half-up
        # FFN residual: x + ffn_out(relu(ffn_in(x)))
        s_fi = self.asc("head.ffn_in.out_q")
        qh, _ = self.linear(qp, s_p, "head.ffn_in.weight", "head.ffn_in.bias", s_fi)
        qh = np.maximum(qh, 0)
        s_tr = self.asc("head.a_trunk")
        fo, sfo = self.wq("head.ffn_out.weight")
        acc = qh @ fo.T + cl(np.round(W["head.ffn_out.bias"] / (s_fi * sfo)), 16)
        # ffn_out requant→INT8 (CLAMP, khớp u_ffn_out ra INT8) RỒI cộng pool residual + clamp (khớp residual_add)
        fo_out = cl(rsh(acc, e(s_tr) - e(s_fi * sfo)), 8)
        qt = cl(fo_out + rsh(qp, e(s_tr) - e(s_p)), 8)
        # proj → 57
        s_o = self.asc("head.a_out")
        qo, _ = self.linear(qt, s_tr, "head.proj.weight", "head.proj.bias", s_o)
        return qo * s_o                                   # trả giá trị thực (chuẩn hoá)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results_qat/best.pt")
    ap.add_argument("--calib", default="results_qat/calib_paper.pt")
    ap.add_argument("--feature", default="data/raw/MARS/feature/featuremap_test.npy")
    ap.add_argument("--label", default="data/raw/MARS/feature/labels_test.npy")
    ap.add_argument("--train-feature", default="data/raw/MARS/feature/featuremap_train.npy")
    ap.add_argument("--train-label", default="data/raw/MARS/feature/labels_train.npy")
    ap.add_argument("--hidden", type=int, default=80)
    ap.add_argument("--n", type=int, default=512, help="số frame test để đo")
    a = ap.parse_args()

    m = IntModel(a.ckpt, a.calib)
    fm_tr = np.load(a.train_feature).astype(np.float64)
    fm_te = np.load(a.feature).astype(np.float64)[:a.n]
    lab = np.load(a.label).astype(np.float64)
    mean = fm_tr.reshape(-1, 5).mean(0); std = fm_tr.reshape(-1, 5).std(0) + 1e-8
    to_jm = lambda Y: (Y.reshape(len(Y), 3, 19).transpose(0, 2, 1).reshape(len(Y), 57)) * 100.0
    Ytr = to_jm(np.load(a.train_label).astype(np.float64))
    y_mean = Ytr.mean(); y_std = Ytr.std() + 1e-8

    Xn = (fm_te - mean) / std
    out = m.forward(Xn)                                   # (n,57) chuẩn hoá
    pred = out * y_std + y_mean
    gt = to_jm(lab[:a.n])
    d = pred - gt
    dj = d.reshape(len(d), 19, 3)                          # (N,19,3) → tách trục x/y/z
    mpjpe = np.sqrt((dj ** 2).sum(-1)).mean()              # L2 mỗi khớp, TB
    print(f"INTEGER reference | {a.n} frame")
    print(f"  MAE  {np.abs(d).mean():.3f} cm   RMSE {np.sqrt((d**2).mean()):.3f} cm   MPJPE {mpjpe:.3f} cm")
    print(f"  MAE per-axis:  x {np.abs(dj[...,0]).mean():.3f}   "
          f"y {np.abs(dj[...,1]).mean():.3f}   z {np.abs(dj[...,2]).mean():.3f}  (cm)")


if __name__ == "__main__":
    main()
