"""
ssm.py — Integer-exact, 1-1 với SV (Fixed Gradient Flow with STE wrapper)
====================================
Nguyên tắc:
  - Mọi phép tính INTEGER dùng torch.long() (INT64) — không bao giờ float khi shift/div.
  - h_int lưu torch.long() (INTEGER đơn vị 2^7, khớp bài báo Ā scale 2^-7).
  - Gradient đi qua STE wrapper (Soft Path) để mô hình có thể học được.
  - Chỉ .float() khi output ra ngoài (dequant).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────
# --- CẤU HÌNH Q5 (Tăng từ Q4 lên Q5 để có đủ độ mịn gradient, tránh gradient crushing) ---
FRAC          = 5                  # Q5: 32 mức lượng tử, bước Δ=0.03125 (Q4 chỉ có 16 mức → dead zone lớn)
SCALE         = 1 << FRAC          # 32

A_FRAC        = 5                  # Đồng bộ A_SCALE với FRAC để tránh lệch biên độ dA_prod
A_SCALE       = 1 << A_FRAC        # 32

EXP_FRAC      = 7                  # Q7: khớp bài báo eMamba (Ā scale = 2^-7, h shift 7)
EXP_SCALE     = 1 << EXP_FRAC      # 128

D2_BIAS_FRAC  = 8                  # Giữ nguyên (Bias INT16 thường để Q8 là đẹp)
D2_BIAS_SCALE = 1 << D2_BIAS_FRAC  # 256

# ── BẢNG ROM NGHỊCH ĐẢO (MÔ PHỎNG HARDWARE LUT) ──────────────────────────────
# Tương đương với đoạn Verilog: inv_rom[i] = (16'sd1 << 15) / i;
# Dùng mảng tĩnh này để Python tra cứu thay vì thực hiện phép chia float/toán học
INV_ROM_ARRAY = [32767] + [min(32767, 32768 // i) for i in range(1, 256)]
INV_ROM = torch.tensor(INV_ROM_ARRAY, dtype=torch.long)


# ── Straight-Through Estimator ────────────────────────────────────────────────
class _STE(torch.autograd.Function):
    """Forward: round+clamp INT8. Backward: pass-through."""
    @staticmethod
    def forward(ctx, x, scale):
        return (x * scale).round().clamp(-128, 127) / scale
    @staticmethod
    def backward(ctx, g):
        return g, None

def fq(x, scale=SCALE):
    """Fake-quant về INT8 tại scale. x và output đều float."""
    return _STE.apply(x, float(scale))


# ── round_shift — INTEGER only ────────────────────────────────────────────────
def rshift(val_long: torch.Tensor, shift: int) -> torch.Tensor:
    """Round half to even - giống np.round và torch.round"""
    assert val_long.dtype == torch.long
    if shift <= 0:
        return val_long
    
    bias = (1 << (shift - 1))
    # Ties to even: + bias - 1 nếu bit thấp nhất là 1 (để round half to even)
    rounded = (val_long + bias - ((val_long >> shift) & 1)) >> shift
    return rounded


def to_long(x_float: torch.Tensor, scale: int) -> torch.Tensor:
    """float Q(scale) → long INTEGER: round(x * scale)"""
    return (x_float * scale).round().to(torch.long)


# ── exp_approx_core — 1-1 từ exp_approx_core.sv ──────────────────────────────
def _exp_core_np(x_arr: np.ndarray) -> np.ndarray:
    """Piecewise linear exp approximation — Q7 output (max=127).
    Input: INT8 Q2 (A_FRAC=2). Output: INT8 signed Q7 [0, 127].
    Khớp bài báo eMamba: Ā scale = 2^-7.
    Lưu ý: exp(0)=1.0=128 nhưng INT8 signed max=127, cap tại 127."""
    x   = x_arr.astype(np.int32)
    out = np.full_like(x, 127, dtype=np.int32)   # x>=0 → 127 (INT8 signed max, khớp SV)

    def seg(lo, hi, slope, bias):
        m = (x >= lo) & (x < hi)
        out[m] = (slope * x[m].astype(np.int32) + bias).astype(np.int32)

    # x < -16 → floor (exp(-4.0)*128 ≈ 2)
    out[x < -16] = 2

    # Piecewise linear segments (least-squares fit, max_err=0 per segment)
    seg(-16, -14,   1,   18)
    seg(-14, -12,   1,   18)
    seg(-12, -11,   2,   30)
    seg(-11,  -9,   3,   41)
    seg( -9,  -7,   4,   49)
    seg( -7,  -5,   7,   71)
    seg( -5,  -3,  10,   87)
    seg( -3,  -1,  18,  114)
    seg( -1,   0,  27,  127)

    return out.astype(np.int64)


def exp_approx_vec(x_long: torch.Tensor) -> torch.Tensor:
    x_np  = x_long.detach().cpu().numpy().astype(np.int32)
    y_np  = _exp_core_np(x_np)
    return torch.from_numpy(y_np).to(dtype=torch.long, device=x_long.device)

# ── RTLConv2D — 1-1 từ emamba_conv2d.sv ──────────────────────────────────────
class RTLConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, frac=FRAC):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.padding      = padding
        self.frac         = frac

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.xavier_uniform_(self.weight)
        self.bias   = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        scale = 1 << self.frac
        
        # 1. HARD PATH (Số nguyên chuẩn RTL)
        x_long = to_long(x, scale)
        W_long = to_long(self.weight, scale)
        b_long = to_long(self.bias, scale)

        # Chạy F.conv2d dưới kiểu double để không bị mất bit số nguyên khi tính nhân tích chập
        conv_out = F.conv2d(x_long.double(), W_long.double(), bias=None, 
                            stride=self.stride, padding=self.padding).long()
        
        # Dịch bit bias và cộng vào mạch tích chập
        b_shift = b_long.view(1, -1, 1, 1) << self.frac
        acc     = conv_out + b_shift
        
        # Làm tròn dịch bit (rshift) và bão hòa về INT8 chuẩn QAT
        y_long  = rshift(acc, self.frac).clamp(-128, 127)
        y_int_exact = y_long.float() / scale
        
        # 2. SOFT PATH (Số thực để lan truyền ngược Gradient)
        x_fq   = fq(x, scale)
        W_fq   = fq(self.weight, scale)
        b_fq   = fq(self.bias, scale)
        y_soft = fq(F.conv2d(x_fq, W_fq, b_fq, stride=self.stride, padding=self.padding), scale)
        
        # 3. KẾT HỢP STE
        return y_int_exact.detach() + (y_soft - y_soft.detach())
# ── RTLLinear — 1-1 từ emamba_linear.sv ──────────────────────────────────────
class RTLLinear(nn.Module):
    def __init__(self, in_dim, out_dim,
                 has_bias=True, frac=FRAC, bias_frac=FRAC):
        super().__init__()
        self.in_dim     = in_dim
        self.out_dim    = out_dim
        self.has_bias   = has_bias
        self.frac       = frac
        self.bias_frac  = bias_frac
        self.bias_shift = frac * 2 - bias_frac

        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.xavier_uniform_(self.weight)
        if has_bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        # 1. HARD PATH: Tính toán bằng số nguyên chuẩn RTL
        x_long = to_long(x, 1 << self.frac)
        W_long = to_long(self.weight, 1 << self.frac)

        acc = (x_long.double() @ W_long.double().t()).long()

        if self.has_bias:
            b_long = to_long(self.bias, 1 << self.bias_frac)
            b_shift = b_long << self.bias_shift
            acc = acc + b_shift

        y_long = rshift(acc, self.frac)
        y_sat  = y_long.clamp(-128, 127)
        y_int_exact = y_sat.float() / (1 << self.frac)

        # 2. SOFT PATH: Tính toán bằng số thực để lấy Gradient
        W_fq = fq(self.weight, 1 << self.frac)
        x_fq = fq(x, 1 << self.frac)
        if self.has_bias:
            b_fq = fq(self.bias, 1 << self.bias_frac)
            y_soft = fq(F.linear(x_fq, W_fq, b_fq), 1 << self.frac)
        else:
            y_soft = fq(F.linear(x_fq, W_fq), 1 << self.frac)

        # 3. KẾT HỢP (STE Detach)
        return y_int_exact.detach() + (y_soft - y_soft.detach())


# ── RTLConv1D — 1-1 từ emamba_conv1d.sv ──────────────────────────────────────
class RTLConv1D(nn.Module):
    def __init__(self, dim, kernel=4, frac=FRAC):
        super().__init__()
        self.dim    = dim
        self.kernel = kernel
        self.frac   = frac
        self.weight = nn.Parameter(torch.empty(dim, kernel))
        nn.init.xavier_uniform_(self.weight)
        self.bias   = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        scale = 1 << self.frac
        
        # 1. HARD PATH
        x_long = to_long(x, scale)
        W_long = to_long(self.weight, scale)
        b_long = to_long(self.bias, scale)

        batch, seq, dim = x.shape
        pad   = torch.zeros(batch, self.kernel - 1, dim,
                            device=x.device, dtype=torch.long)
        x_pad = torch.cat([pad, x_long], dim=1)

        outputs = []
        for t in range(seq):
            win     = x_pad[:, t:t + self.kernel, :]                     
            b_shift = b_long << self.frac                                
            mac     = (win.double() * W_long.double().t().unsqueeze(0)).sum(dim=1).long()        
            acc     = mac + b_shift
            acc_r   = rshift(acc, self.frac).clamp(-128, 127)            
            outputs.append(acc_r.float() / scale)

        y_int_exact = torch.stack(outputs, dim=1)
        
        # 2. SOFT PATH
        W_fq = fq(self.weight, scale)
        x_fq = fq(x, scale)
        b_fq = fq(self.bias, scale)

        x_conv = x_fq.transpose(1, 2) 
        W_conv = W_fq.unsqueeze(1) 
        p = self.kernel - 1
        y_soft = F.conv1d(F.pad(x_conv, (p, 0)), W_conv, b_fq, groups=self.dim)
        y_soft = fq(y_soft.transpose(1, 2), scale)

        # 3. KẾT HỢP
        return y_int_exact.detach() + (y_soft - y_soft.detach())


# ── RTLRangeNorm — 1-1 từ emamba_range_norm.sv ───────────────────────────────
class RTLRangeNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim   = dim
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta  = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        g_fq = fq(self.gamma)
        b_fq = fq(self.beta)

        # 1. HARD PATH
        x_long = to_long(x, SCALE)
        sum_l  = x_long.sum(dim=-1, keepdim=True)
        half_d = self.dim // 2
        
        # --- FIX TRÙM CUỐI: Ép phép chia số nguyên (Truncation) giống hệt Verilog ---
        # Verilog dùng toán tử '/' (cắt bỏ phần thập phân, vd: -15 / 20 = 0)
        # Python mặc định '//' là Floor (vd: -15 // 20 = -1) -> Phải dùng torch.div(..., rounding_mode='trunc')
        num_l  = torch.where(sum_l >= 0, sum_l + half_d, sum_l - half_d)
        mean_l = torch.div(num_l, self.dim, rounding_mode='trunc')

        xc_l   = x_long - mean_l
        xc_max = xc_l.max(dim=-1, keepdim=True).values
        xc_min = xc_l.min(dim=-1, keepdim=True).values
        
        # --- Ép kiểu và tra bảng ROM ---
        r_l    = (xc_max - xc_min).clamp(0, 255)         # Lấy Range từ 0 -> 255
        inv_rom_device = INV_ROM.to(x.device)            # Đưa bảng ROM vào cùng Device
        inv_l  = inv_rom_device[r_l]                     # Tra bảng LUT y hệt Verilog

        g_long   = to_long(g_fq, SCALE)
        scaled_l = (xc_l * g_long).long()
        normed_l = scaled_l * inv_l
        n_val_l  = rshift(normed_l, 15)

        b_long   = to_long(b_fq, SCALE)
        final_l  = (n_val_l + b_long).clamp(-128, 127)
        y_int_exact = final_l.float() / SCALE

        # 2. SOFT PATH
        x_fq = fq(x, SCALE)
        mean_soft = x_fq.mean(dim=-1, keepdim=True)
        xc_soft = x_fq - mean_soft
        xc_max_soft = xc_soft.max(dim=-1, keepdim=True).values
        xc_min_soft = xc_soft.min(dim=-1, keepdim=True).values
        r_soft = (xc_max_soft - xc_min_soft).clamp(min=1e-5)
        # Tương đương chuẩn hoá dải float
        normed_soft = xc_soft / r_soft
        y_soft = fq(normed_soft * g_fq + b_fq, SCALE)

        # 3. KẾT HỢP
        return y_int_exact.detach() + (y_soft - y_soft.detach())
    
# ── RTLSiLU — 1-1 từ emamba_silu.sv ─────────────────────────────────────────
class RTLSiLU(nn.Module):
    @staticmethod
    def _sb(b_x1000):
        s = b_x1000 * SCALE
        return int((s + 500) // 1000) if s >= 0 else int((s - 500) // 1000)

    @classmethod
    def _tbl(cls):
        O = SCALE; H = SCALE >> 1; sb = cls._sb
        return [
            (-7*O,  None, None),      
            (-5*O,  -3,   sb(-101)),
            (-4*O,  -10,  sb(-226)),
            (-3*O,  -18,  sb(-354)),
            (-2*O,  -25,  sb(-431)),
            (-O-H,  -18,  sb(-379)),
            (-O,     2,   sb(-260)),
            (-H,     41,  sb(-109)),
            (0,      97,  0),
            (H,      159, 0),
            (O,      215, sb(-109)),
            (O+H,    254, sb(-260)),
            (2*O,    274, sb(-380)),
            (3*O,    281, sb(-431)),
            (4*O,    274, sb(-353)),
            (5*O,    266, sb(-226)),
            (6*O,    261, sb(-127)),
            (7*O,    258, sb(-65)),
            (None,  None, None),      
        ]

    def forward(self, x):
        # 1. HARD PATH
        x_long = to_long(x, SCALE)    
        tbl    = self._tbl()
        out    = torch.zeros_like(x)

        for i, (thresh, slope, bias_i) in enumerate(tbl):
            if thresh is None:
                mask = x_long >= (7 * SCALE)
                out  = torch.where(mask, x, out)
            elif slope is None:
                mask = x_long < thresh
                out  = torch.where(mask, torch.zeros_like(x), out)
            else:
                prev = tbl[i-1][0]
                mask = (x_long >= prev) & (x_long < thresh)
                prod   = x_long * slope                          
                y_full = ((prod + 128) >> 8) + bias_i           
                y_sat  = y_full.clamp(-128, 127)
                out    = torch.where(mask, y_sat.float() / SCALE, out)
        y_int_exact = out

        # 2. SOFT PATH
        x_fq = fq(x, SCALE)
        y_soft = fq(x_fq * torch.sigmoid(x_fq), SCALE)

        # 3. KẾT HỢP
        return y_int_exact.detach() + (y_soft - y_soft.detach())


# ── RTLReLU ───────────────────────────────────────────────────────────────────
def rtl_relu(x):
    return torch.where(x > 0, x, torch.zeros_like(x))


# ── RTLSelection — 1-1 từ emamba_selection.sv ────────────────────────────────
class RTLSelection(nn.Module):
    def __init__(self, d_inner, dt_rank, n_state):
        super().__init__()
        self.proj_b       = RTLLinear(d_inner, n_state,  has_bias=False)
        self.proj_c       = RTLLinear(d_inner, n_state,  has_bias=False)
        self.proj_delta_1 = RTLLinear(d_inner, dt_rank,  has_bias=False)
        self.proj_delta_2 = RTLLinear(dt_rank, d_inner,
                                      has_bias=True, bias_frac=D2_BIAS_FRAC)
        with torch.no_grad():
            nn.init.uniform_(self.proj_delta_1.weight, -0.3, 0.3)
            nn.init.uniform_(self.proj_delta_2.weight, -0.3, 0.3)

    def forward(self, x):
        # Không cần thay đổi do bên trong dùng RTLLinear đã được gắn STE wrapper
        B     = self.proj_b(x)
        C     = self.proj_c(x)
        dt    = fq(rtl_relu(self.proj_delta_1(x)))
        delta = fq(rtl_relu(self.proj_delta_2(dt)))
        return B, C, delta


# ── SSM — 1-1 từ emamba_ssm.sv FSM ──────────────────────────────────────────
class SSM(nn.Module):
    def __init__(self, d_inner, n_state, dt_rank=3):
        super().__init__()
        self.d_inner   = d_inner
        self.n_state   = n_state
        self.selection = RTLSelection(d_inner, dt_rank, n_state)

        A_init = torch.arange(1, n_state+1, dtype=torch.float32).unsqueeze(0)
        A_init = A_init.expand(d_inner, -1).contiguous()
        self.A_log = nn.Parameter(torch.log(A_init))
        self.D     = nn.Parameter(torch.ones(d_inner))

        self.register_buffer('h_int',
            torch.zeros(d_inner, n_state, dtype=torch.long))

    def reset_state(self):
        self.h_int.zero_()

    def forward(self, x_seq):
        batch, seq, _ = x_seq.shape

        A_fq   = fq(-torch.exp(self.A_log), A_SCALE)   
        A_long = to_long(A_fq, A_SCALE)                  
        D_fq   = fq(self.D)
        D_long = to_long(D_fq, SCALE)                    

        # HARD PATH state
        h = self.h_int.unsqueeze(0).expand(batch, -1, -1).clone()  
        
        # SOFT PATH state (bắt đầu từ float tương đương để cộng dồn gradient)
        h_soft = h.float() / EXP_SCALE

        outputs = []
        outputs_soft = []

        for t in range(seq):
            x_t  = x_seq[:, t, :]                       
            B_t, C_t, delta_t = self.selection(x_t)     

            # 1. HARD PATH ────────────────────────────────────────────────
            # EXP_FRAC=7 alignment (khớp bài báo eMamba Section 4.6):
            #   A_bar: Q7 (frac=7)
            #   h_prev: Q7 (frac=7)
            #   r_Ah = A_bar * h → frac=14
            #   r_dx = delta(Q5) * x(Q5) → frac=10
            #   r_Bx = r_dx * B(Q5) → frac=15, rshift 1 → frac=14
            #   v_hn = r_Ah + r_Bx_aligned → frac=14
            #   h_store = rshift(v_hn, 7) → frac=7 → INT17 (matches paper!)
            #   r_Ch = C(Q5) * v_hn(frac14) → frac=19
            #   v_Dx = D(Q5) * x(Q5) → frac=10, <<9 → frac=19
            #   y = rshift(total, 14) → frac=5 → Q5 → INT8
            x_long     = to_long(x_t,     SCALE)        
            B_long     = to_long(B_t,     SCALE)        
            C_long     = to_long(C_t,     SCALE)        
            delta_long = to_long(delta_t, SCALE)        

            dA_prod = delta_long.unsqueeze(-1) * A_long.unsqueeze(0)  
            dA_raw  = rshift(dA_prod, 5).clamp(-128, 127)             

            A_bar = exp_approx_vec(dA_raw)               # Q7 output (max=127, INT8 signed)

            r_dx = delta_long * x_long                   # frac=10
            r_Ah = A_bar * h                             # frac=7+7=14
            r_Bx = r_dx.unsqueeze(-1) * B_long.unsqueeze(1)  # frac=15
            r_Bx_aligned = rshift(r_Bx, 1)              # frac=14 (align với r_Ah)
            v_hn_f  = r_Ah + r_Bx_aligned               # frac=14
            h = rshift(v_hn_f, EXP_FRAC).clamp(-65536, 65535)  # shift 7 → Q7 → INT17
            r_Ch = C_long.unsqueeze(1) * v_hn_f          # frac=5+14=19

            v_sum   = r_Ch.sum(dim=-1)                   # frac=19
            v_Dx    = D_long * x_long                    # frac=10
            v_Dx_sh = v_Dx * (1 << 9)                   # frac=19 (align với v_sum)
            total   = v_sum + v_Dx_sh                    # frac=19
            y_long  = rshift(total, 14).clamp(-128, 127) # frac=5 → Q5 → INT8
            y_int_exact = y_long.float() / SCALE
            outputs.append(y_int_exact)

            # 2. SOFT PATH ────────────────────────────────────────────────
            dA_soft = delta_t.unsqueeze(-1) * A_fq.unsqueeze(0) 
            A_bar_soft = torch.exp(dA_soft)
            
            r_Ah_soft = A_bar_soft * h_soft
            r_Bx_soft = (delta_t * x_t).unsqueeze(-1) * B_t.unsqueeze(1)
            h_soft = r_Ah_soft + r_Bx_soft
            
            r_Ch_soft = C_t.unsqueeze(1) * h_soft
            v_sum_soft = r_Ch_soft.sum(dim=-1)
            v_Dx_soft = D_fq * x_t
            
            y_soft_t = fq(v_sum_soft + v_Dx_soft, SCALE)
            outputs_soft.append(y_soft_t)

        self.h_int = h[0].detach().to(torch.long)

        y_int_exact_seq = torch.stack(outputs, dim=1)
        y_soft_seq = torch.stack(outputs_soft, dim=1)

        # 3. KẾT HỢP
        return y_int_exact_seq.detach() + (y_soft_seq - y_soft_seq.detach())