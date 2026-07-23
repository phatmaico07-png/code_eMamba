#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""demo_viewer_live.py — 3-PANEL demo (style An&Ogras 2021) chạy LIVE trên KV260.

    [ 1. HW Accelerator eMamba ]   [ 2. fp32 Python ]   [ 3. Kinect Ground Truth ]

  • Panel 1 = output ACCELERATOR THẬT: mỗi frame đẩy input.bin -> FPGA (single-port
    streaming) -> 57 mã INT8 -> đổi sang cm (pose_cm = code·s_o·y_std + y_mean) -> vẽ.
  • Panel 2 = dự đoán model fp32 (PyTorch), đọc pred_cm từ bundle .npz tính sẵn.
  • Panel 3 = ground-truth Kinect (gt_cm) trong bundle.
  Cả 3 panel cùng hệ cm, cùng trục -> so trực tiếp. Có slider/play/pause/tốc-độ-fps.

Chạy trên board (HDMI mượt nhất; ssh -X cũng được nhưng chậm vì 3 panel 3D):
  sudo apt install -y python3-matplotlib python3-tk        # 1 lần
  # live FPGA (cần root cho UIO):
  sudo -E env DISPLAY=$DISPLAY XAUTHORITY=$HOME/.Xauthority \
      python3 demo_viewer_live.py --bundle results/bundle_fp32.npz --input input.bin
  # thử trên PC không cần FPGA (panel1 = fp32 tạm):
  python3 demo_viewer_live.py --sim --bundle results/bundle_fp32.npz
Phím: Space=play/pause · ←/→ frame · Q=thoát · nút fps đổi tốc độ đẩy frame.
"""
import sys, os, time, struct, glob, argparse, subprocess
import numpy as np

# ── hằng đổi mã HW -> cm (khớp make_bundle_rtl.py) ──
S_O, Y_MEAN, Y_STD = 0.015625, 66.482895, 95.577004     # head.a_out=2^-6 ; nhãn train
IN_WORDS, OUT_WORDS, OUT_BYTES = 80, 15, 57
PORT_OFF = 0x100
CMD_INPUT, CMD_START, CMD_READOUT = 0x02, 0x05, 0x06
S_DONE = 0x1
NJ = 19
SKELETON = [(0,1),(1,18),(18,2),(2,3), (18,4),(4,5),(5,6), (18,7),(7,8),(8,9),
            (0,10),(10,11),(11,12),(12,13), (0,14),(14,15),(15,16),(16,17)]
RIGHT = {4,5,6,10,11,12,13}; LEFT = {7,8,9,14,15,16,17}


def hw_to_cm(out57):
    """57 byte INT8 (joint-major x,y,z) -> (19,3) cm."""
    c = np.frombuffer(bytes(out57), dtype=np.int8).astype(np.float64)
    return (c * S_O * Y_STD + Y_MEAN).reshape(NJ, 3)


# ════════════════════════════ NGUỒN FPGA ════════════════════════════
class BoardSource:
    def __init__(self):
        import mmap
        uio = self._find_uio("MY_IP")
        if uio is None:
            raise RuntimeError("Khong tim thay UIO 'MY_IP' — da nap bitstream chua?")
        size = self._map_size(uio)
        self.fd = os.open("/dev/" + uio, os.O_RDWR | os.O_SYNC)
        self.m = mmap.mmap(self.fd, size, mmap.MAP_SHARED,
                           mmap.PROT_READ | mmap.PROT_WRITE, offset=0)
        self.stalls = 0
    @staticmethod
    def _find_uio(name):
        for p in sorted(glob.glob("/sys/class/uio/uio*")):
            try:
                if open(p + "/name").read().strip() == name:
                    return os.path.basename(p)
            except OSError:
                pass
        return None
    @staticmethod
    def _map_size(uio):
        try:
            return int(open("/sys/class/uio/%s/maps/map0/size" % uio).read().strip(), 16)
        except OSError:
            return 0x10000
    def put(self, v):  struct.pack_into("<I", self.m, PORT_OFF, v & 0xFFFFFFFF)
    def cmd(self, c, n=0):  self.put(((c & 0xFF) << 24) | (n & 0xFFFFFF))
    def get(self):  return struct.unpack_from("<I", self.m, PORT_OFF)[0]
    def load_param(self, exe, weights):
        r = subprocess.run([exe, weights], capture_output=True, text=True)
        sys.stdout.write(r.stdout)
        if "KHOP" not in r.stdout or "SAI" in r.stdout:
            raise RuntimeError("load_param bao SAI — xem log.")
    def infer(self, inwords):
        self.cmd(CMD_INPUT, IN_WORDS)
        for w in inwords:
            self.put(w)
        self.cmd(CMD_START, 0)
        ok = False
        for _ in range(4):
            s = 5_000_000
            while s > 0 and (self.get() & S_DONE):     s -= 1
            while s > 0 and not (self.get() & S_DONE): s -= 1
            if s > 0:
                ok = True; break
            self.stalls += 1; self.cmd(CMD_START, 0)
        if not ok:
            raise RuntimeError("STALL: accelerator khong tra done")
        self.cmd(CMD_READOUT, 0)
        out = bytearray()
        for _ in range(OUT_WORDS):
            v = self.get()
            out += bytes((v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF, (v >> 24) & 0xFF))
        return bytes(out[:OUT_BYTES])


def angle(a, b, c):
    v1, v2 = a - b, c - b
    cos = (v1 * v2).sum() / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


# ════════════════════════════ GUI 3-PANEL ════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="results/bundle_fp32.npz", help="npz có pred_cm + gt_cm")
    ap.add_argument("--input",  default="input.bin")
    ap.add_argument("--weights", default="weights.bin")
    ap.add_argument("--loadexe", default="./load_param")
    ap.add_argument("--no-load", action="store_true")
    ap.add_argument("--sim", action="store_true", help="không FPGA: panel1 = fp32 tạm (test PC)")
    ap.add_argument("--fps", type=float, default=10.0)
    a = ap.parse_args()

    d = np.load(a.bundle, allow_pickle=True)
    preds, gts = d["pred_cm"].astype(np.float64), d["gt_cm"].astype(np.float64)   # (N,19,3) cm
    N = len(preds)

    src = None; inputs = b""
    if not a.sim:
        inputs = open(a.input, "rb").read()
        N = min(N, len(inputs) // (IN_WORDS * 4))
        src = BoardSource()
        if not a.no_load:
            print("Nap tham so qua load_param ...")
            src.load_param(a.loadexe, a.weights)

    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    from matplotlib.widgets import Button, Slider

    # trục chung (từ GT) -> 3 panel cùng tỉ lệ
    allp = gts.reshape(-1, 3)
    m = 15.0
    lim = [(allp[:, k].min() - m, allp[:, k].max() + m) for k in range(3)]

    fig = plt.figure(figsize=(16, 6.2), facecolor="white")
    fig.subplots_adjust(left=0.02, right=0.99, bottom=0.16, top=0.90, wspace=0.04)
    axes = [fig.add_subplot(1, 3, i + 1, projection="3d") for i in range(3)]
    titles = ["1. HW Accelerator eMamba (KV260 FPGA)", "2. fp32 Python (PyTorch)", "3. Kinect Ground Truth"]
    colors = ["crimson", "#1f77b4", "steelblue"]

    txt_ang = fig.text(0.5, 0.085, "", ha="center", fontsize=12, family="monospace")
    txt_frm = fig.text(0.5, 0.03, "", ha="center", fontsize=11, fontweight="bold")
    txt_met = fig.text(0.99, 0.97, "", ha="right", va="top", fontsize=9, color="gray")

    def draw_skel(ax, j, color, title):
        ax.cla()
        ax.scatter(j[:, 0], j[:, 1], j[:, 2], c=color, s=55, edgecolors="white", linewidths=1.2, zorder=5)
        for x, y in SKELETON:
            ax.plot([j[x,0],j[y,0]], [j[x,1],j[y,1]], [j[x,2],j[y,2]], c=color, lw=2.0, alpha=0.75)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1]); ax.set_zlim(*lim[2])
        ax.tick_params(labelsize=7); ax.view_init(elev=15, azim=-60); ax.set_box_aspect([1.2,1,1])

    st = dict(idx=0, playing=True, t0=time.time(), fc=0, fps=0.0, us=0.0)

    def get_hw(i):
        if a.sim:
            return preds[i]                      # PC test: dùng fp32 tạm cho panel1
        base = i * IN_WORDS * 4
        iw = [struct.unpack_from("<I", inputs, base + 4*k)[0] for k in range(IN_WORDS)]
        t = time.time(); out = src.infer(iw); st["us"] = (time.time()-t)*1e6
        return hw_to_cm(out)

    def render(i):
        hw = get_hw(i)
        draw_skel(axes[0], hw,        colors[0], titles[0])
        draw_skel(axes[1], preds[i],  colors[1], titles[1])
        draw_skel(axes[2], gts[i],    colors[2], titles[2])
        try:
            aa = lambda P: (angle(P[4],P[5],P[6]), angle(P[7],P[8],P[9]),
                            angle(P[10],P[11],P[12]), angle(P[14],P[15],P[16]))
            le, re, lk, rk = aa(hw)
            txt_ang.set_text(f"HW góc — khuỷu T {le:5.0f}°  P {re:5.0f}°    gối T {lk:5.0f}°  P {rk:5.0f}°")
        except Exception:
            txt_ang.set_text("")
        dh = hw - gts[i]; df = preds[i] - gts[i]
        txt_met.set_text(f"HW vs GT: RMSE {np.sqrt((dh**2).mean()):.2f} cm   |   "
                         f"fp32 vs GT: RMSE {np.sqrt((df**2).mean()):.2f} cm")
        src_lbl = "SIM" if a.sim else "LIVE FPGA"
        extra = f" · 1 frame {st['us']:.0f} µs" if not a.sim else ""
        txt_frm.set_text(f"[{src_lbl}]  frame {i}/{N}  ·  {st['fps']:.0f} fps{extra}")
        slider.eventson = False; slider.set_val(i); slider.eventson = True
        fig.canvas.draw_idle()

    # ── controls (đáy) ──
    ax_sl = fig.add_axes([0.20, 0.005, 0.45, 0.02])
    slider = Slider(ax_sl, "", 0, N-1, valinit=0, valstep=1, color="#9ca3af"); slider.valtext.set_visible(False)
    b_prev = Button(fig.add_axes([0.13, 0.005, 0.03, 0.02]), "◄")
    b_play = Button(fig.add_axes([0.66, 0.005, 0.06, 0.02]), "❚❚")
    b_next = Button(fig.add_axes([0.16, 0.005, 0.03, 0.02]), "►")
    b_fps  = Button(fig.add_axes([0.73, 0.005, 0.06, 0.02]), f"{int(a.fps)} fps")

    timer = fig.canvas.new_timer(interval=int(1000/a.fps))
    def tick():
        if not st["playing"]: return
        render(st["idx"])
        st["fc"] += 1
        if st["fc"] >= 8:
            now=time.time(); st["fps"]=st["fc"]/(now-st["t0"]); st["t0"]=now; st["fc"]=0
        st["idx"] = (st["idx"] + 1) % N
    timer.add_callback(tick); timer.start()

    def on_slider(v):
        st["playing"]=False; b_play.label.set_text("▶"); st["idx"]=int(v); render(int(v))
    def on_prev(_): st["playing"]=False; b_play.label.set_text("▶"); st["idx"]=max(0,st["idx"]-1); render(st["idx"])
    def on_next(_): st["playing"]=False; b_play.label.set_text("▶"); st["idx"]=min(N-1,st["idx"]+1); render(st["idx"])
    def on_play(_):
        st["playing"]=not st["playing"]; b_play.label.set_text("❚❚" if st["playing"] else "▶")
    FPS=[5,10,20,30]
    def on_fps(_):
        cur=int(round(1000.0/timer.interval)); n=(FPS.index(cur)+1)%len(FPS) if cur in FPS else 1
        timer.interval=int(1000/FPS[n]); b_fps.label.set_text(f"{FPS[n]} fps")
    slider.on_changed(on_slider); b_prev.on_clicked(on_prev); b_next.on_clicked(on_next)
    b_play.on_clicked(on_play); b_fps.on_clicked(on_fps)

    def on_key(e):
        if e.key in ("q","escape"): plt.close(fig)
        elif e.key == " ": on_play(None)
        elif e.key == "left":  on_prev(None)
        elif e.key == "right": on_next(None)
    fig.canvas.mpl_connect("key_press_event", on_key)

    fig.suptitle("eMamba-MARS  ·  HW accelerator vs fp32 vs Kinect", x=0.01, ha="left",
                 fontsize=12, fontweight="bold", color="#374151")
    print(f"{N} frame | Space play/pause · ◄►/slider chọn frame · nút fps đổi tốc độ")
    plt.show()


if __name__ == "__main__":
    main()
