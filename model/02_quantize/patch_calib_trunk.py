"""patch_calib_trunk.py — nhân scale 1 activation (mặc định head.a_trunk) lên x lần.

Dùng khi bias 1 lớp vượt INT16 (vd head.proj.bias): tăng s_x (a_trunk) → bias/(s_x·s_w)
giảm → vừa INT16, KHÔNG cần sửa RTL. Scale pow2 nên x2 = +1 bit. gen_synth/int_reference/
export đều đọc lại calib → tự nhất quán (chỉ khác shift baked).

    python patch_calib_trunk.py --calib results_all/calib.pt --out results_all/calib_fix.pt
"""
import argparse, torch

ap = argparse.ArgumentParser()
ap.add_argument("--calib", default="results_all/calib.pt")
ap.add_argument("--out", default="results_all/calib_fix.pt")
ap.add_argument("--key", default="head.a_trunk", help="activation cần nới scale")
ap.add_argument("--mul", type=float, default=2.0, help="hệ số nhân scale (x2 = +1 bit)")
a = ap.parse_args()

ck = torch.load(a.calib, map_location="cpu", weights_only=False)
st = ck["calib"][a.key]
touched = []
for k in ("thr_fixed", "absmax"):
    if st.get(k) is not None:
        st[k] = st[k] * a.mul          # GIỮ kiểu (tensor→tensor cho observer QAT; float→float)
        touched.append(k)
torch.save(ck, a.out)
print(f"{a.key}: {touched} x{a.mul} -> {a.out}")
