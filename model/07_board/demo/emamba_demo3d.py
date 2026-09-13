#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
emamba_demo3d.py — Demo 3D real-time eMamba accelerator (MARS mmWave pose) trên KV260.

  • BOARD (mặc định): mmap UIO "MY_IP" -> đẩy TỪNG frame qua FPGA (single-port streaming,
    giống main.c / emamba_demo.py) -> đọc 19 khớp × (x, depth, height) int8 -> vẽ
    skeleton 3D động ngay (real-time stream).
  • --sim : đọc thẳng golden.bin (KHÔNG cần FPGA) để thử GUI trên PC.

Hiển thị bằng matplotlib (backend TkAgg) -> chạy mượt + nhẹ qua  ssh -X.
Giải mã output: 57 byte = 19 khớp × (x, depth, height) int8 interleaved (validate bit-exact).

Chạy qua SSH (X forward):
  ssh -X debian@KV260
  sudo apt install -y python3-matplotlib python3-tk      # 1 lần
  cd ~/DATN
  # UIO cần root nhưng ssh -X mất X-cookie khi sudo -> giữ DISPLAY/XAUTHORITY:
  sudo -E env DISPLAY=$DISPLAY XAUTHORITY=$HOME/.Xauthority python3 emamba_demo3d.py
Thử trên PC (không cần board):
  python3 emamba_demo3d.py --sim
Phím: Space=play/pause  R=restart  Q/Esc=thoát  ←/→ xoay  ↑/↓ ngẩng
"""
import sys, os, time, struct, glob, argparse, subprocess

# ───────────────────── giao thức accelerator (khớp emamba_hw.h) ─────────────────────
PORT_OFF   = 0x100
CMD_WEIGHT, CMD_INPUT, CMD_SHIFT, CMD_SHCTRL = 0x01, 0x02, 0x03, 0x04
CMD_START, CMD_READOUT, CMD_RDSTAT, CMD_RDSUM, CMD_RDSHIFT = 0x05, 0x06, 0x07, 0x08, 0x09
S_DONE, S_BUSY, S_LOADDONE = 0x1, 0x2, 0x4
IN_WORDS, OUT_WORDS, OUT_BYTES = 80, 15, 57
NJ = 19

# ───────────── skeleton MARS 19 khớp (đã xác định thực nghiệm, từ emamba_demo.py) ─────────────
# 0 hông-gốc 1 lưng-giữa 2 cổ 3 đầu | 4-6 tay phải | 7-9 tay trái | 10-13 chân phải | 14-17 chân trái | 18 vai-cột-sống
EDGES = [(0,1),(1,18),(18,2),(2,3),
         (18,4),(4,5),(5,6),
         (18,7),(7,8),(8,9),
         (0,10),(10,11),(11,12),(12,13),
         (0,14),(14,15),(15,16),(16,17)]
RIGHT = {4,5,6,10,11,12,13}
LEFT  = {7,8,9,14,15,16,17}


def s8(b):
    return b - 256 if b > 127 else b


# ════════════════════════════ NGUỒN DỮ LIỆU ════════════════════════════
class BoardSource:
    """Đẩy frame qua FPGA thật (UIO mmap), trả 57 byte output."""
    def __init__(self):
        import mmap
        uio = self._find_uio("MY_IP")
        if uio is None:
            raise RuntimeError("Khong tim thay UIO 'MY_IP' — da nap bitstream + device tree chua?")
        size = self._map_size(uio)
        self.fd = os.open("/dev/" + uio, os.O_RDWR | os.O_SYNC)
        self.m  = mmap.mmap(self.fd, size, mmap.MAP_SHARED,
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
            raise RuntimeError("load_param bao SAI — xem log tren.")

    def infer(self, inwords):
        self.cmd(CMD_INPUT, IN_WORDS)
        for w in inwords:
            self.put(w)
        self.cmd(CMD_START, 0)
        ok = False
        for _ in range(4):
            s = 5_000_000
            while s > 0 and (self.get() & S_DONE):      s -= 1
            while s > 0 and not (self.get() & S_DONE):  s -= 1
            if s > 0:
                ok = True; break
            self.stalls += 1
            self.cmd(CMD_START, 0)
        if not ok:
            raise RuntimeError("STALL: accelerator khong tra done")
        self.cmd(CMD_READOUT, 0)
        out = bytearray()
        for _ in range(OUT_WORDS):
            v = self.get()
            out += bytes((v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF, (v >> 24) & 0xFF))
        return bytes(out[:OUT_BYTES])


class FileSource:
    """--sim: lấy frame từ golden.bin (không đụng FPGA)."""
    def __init__(self, golden):
        self.data = open(golden, "rb").read()
        self.stalls = 0
    def load_param(self, *a):  pass
    def infer_frame(self, idx):
        return self.data[idx * OUT_BYTES:(idx + 1) * OUT_BYTES]


def decode(out):
    """57 byte -> 3 list (x, depth, height) int8 cho 19 khớp."""
    xs = [s8(out[3 * j])     for j in range(NJ)]
    ds = [s8(out[3 * j + 1]) for j in range(NJ)]
    hs = [s8(out[3 * j + 2]) for j in range(NJ)]
    return xs, ds, hs


def scan_limits(golden_bytes):
    """Quét golden.bin lấy min/max từng trục -> view 3D ổn định, không nhảy scale."""
    n = len(golden_bytes) // OUT_BYTES
    lo = [127, 127, 127]; hi = [-128, -128, -128]
    for f in range(n):
        out = golden_bytes[f * OUT_BYTES:(f + 1) * OUT_BYTES]
        for j in range(NJ):
            for k in range(3):
                v = s8(out[3 * j + k]); lo[k] = min(lo[k], v); hi[k] = max(hi[k], v)
    # nới biên 8%
    for k in range(3):
        m = max(4, int((hi[k] - lo[k]) * 0.08)); lo[k] -= m; hi[k] += m
    return lo, hi


# ════════════════════════════ GUI 3D (matplotlib / ssh -X) ════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true", help="đọc golden.bin, không đụng FPGA")
    ap.add_argument("--input",   default="input.bin")
    ap.add_argument("--golden",  default="golden.bin")
    ap.add_argument("--weights", default="weights.bin")
    ap.add_argument("--loadexe", default="./load_param")
    ap.add_argument("--no-load", action="store_true", help="bỏ qua nạp weight (đã nạp trước)")
    ap.add_argument("--fps", type=float, default=20.0, help="tốc độ hiển thị mong muốn")
    ap.add_argument("--elev", type=float, default=12.0)
    ap.add_argument("--azim", type=float, default=-70.0)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("TkAgg")                      # backend nhẹ, hợp ssh -X
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D      # noqa: F401

    golden = open(a.golden, "rb").read() if os.path.exists(a.golden) else b""
    inputs = b""
    if a.sim:
        if not golden: sys.exit("--sim can golden.bin")
        src = FileSource(a.golden); nframes = len(golden) // OUT_BYTES
    else:
        inputs = open(a.input, "rb").read()
        src = BoardSource()
        if not a.no_load:
            print("Nạp tham số qua load_param ...")
            src.load_param(a.loadexe, a.weights)
        nframes = (len(golden) // OUT_BYTES) if golden else (len(inputs) // (IN_WORDS * 4))

    lo, hi = scan_limits(golden) if golden else ([-128]*3, [127]*3)

    # ── figure 3D ──
    plt.rcParams["toolbar"] = "None"
    fig = plt.figure(figsize=(7.2, 8.0), facecolor="#12141c")
    ax = fig.add_subplot(111, projection="3d", facecolor="#12141c")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.9)
    # trục: ngang=x(0), sâu=depth(1), cao=height(2)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    try: ax.set_box_aspect((hi[0]-lo[0], hi[1]-lo[1], hi[2]-lo[2]))
    except Exception: pass
    ax.view_init(elev=a.elev, azim=a.azim)
    ax.set_axis_off()

    # đường xương (mỗi cạnh 1 Line3D) + scatter khớp
    cols = []
    for _a, b in EDGES:
        c = "#50c878" if b in RIGHT else "#5a96ff" if b in LEFT else "#e6e6f0"
        cols.append(c)
    lines = [ax.plot([0,0],[0,0],[0,0], c=cols[i], lw=3, solid_capstyle="round")[0]
             for i in range(len(EDGES))]
    scat = ax.scatter([0]*NJ, [0]*NJ, [0]*NJ, s=28, c="#f0d25a", depthshade=False)
    head = ax.scatter([0],[0],[0], s=70, c="#ff5a5a", depthshade=False)
    title = ax.set_title("", color="#e8e8f0", fontsize=11, family="monospace", loc="left", pad=14)

    # ── trạng thái ──
    st = dict(idx=0, playing=True, mism=0, tot=0, t0=time.time(), fcnt=0, fps=0.0, us=0.0)

    def on_key(ev):
        k = ev.key
        if k in ("q", "escape"): plt.close(fig)
        elif k == " ": st["playing"] = not st["playing"]
        elif k == "r": st["idx"] = st["mism"] = st["tot"] = 0
        elif k == "left":  ax.view_init(elev=ax.elev, azim=ax.azim - 8)
        elif k == "right": ax.view_init(elev=ax.elev, azim=ax.azim + 8)
        elif k == "up":    ax.view_init(elev=ax.elev + 6, azim=ax.azim)
        elif k == "down":  ax.view_init(elev=ax.elev - 6, azim=ax.azim)
    fig.canvas.mpl_connect("key_press_event", on_key)

    def step(_frame):
        if not st["playing"]:
            return lines + [scat, head]
        i = st["idx"]
        t = time.time()
        if a.sim:
            out = src.infer_frame(i)
        else:
            base = i * IN_WORDS * 4
            iw = [struct.unpack_from("<I", inputs, base + 4 * k)[0] for k in range(IN_WORDS)]
            out = src.infer(iw)
        st["us"] = (time.time() - t) * 1e6

        if golden:
            g = golden[i * OUT_BYTES:(i + 1) * OUT_BYTES]
            d = sum(1 for k in range(OUT_BYTES) if out[k] != g[k])
            st["mism"] += d; st["tot"] += OUT_BYTES

        xs, ds, hs = decode(out)
        for li, (pa, pb) in enumerate(EDGES):
            lines[li].set_data([xs[pa], xs[pb]], [ds[pa], ds[pb]])
            lines[li].set_3d_properties([hs[pa], hs[pb]])
        import numpy as np
        scat._offsets3d = (np.array(xs), np.array(ds), np.array(hs))
        head._offsets3d = (np.array([xs[3]]), np.array([ds[3]]), np.array([hs[3]]))

        st["fcnt"] += 1
        if st["fcnt"] >= 10:
            now = time.time(); st["fps"] = st["fcnt"]/(now-st["t0"]); st["t0"]=now; st["fcnt"]=0
        mode = "SIM (golden)" if a.sim else "FPGA (KV260)"
        acc = ("BIT-EXACT (lech %d/%d)" % (st["mism"], st["tot"])) if (st["tot"] and st["mism"]==0) \
              else ("lech %d/%d" % (st["mism"], st["tot"])) if st["tot"] else "-"
        stall = ("  stall:%d" % src.stalls) if (not a.sim and src.stalls) else ""
        title.set_text("eMamba 3D  |  %s  frame %d/%d  |  %.0f fps  |  1 frame %.0f us%s\n"
                       "so golden: %s   [Space R Q  ←→↑↓ xoay]"
                       % (mode, i, nframes, st["fps"], st["us"], stall, acc))

        st["idx"] = (i + 1) % nframes
        if st["idx"] == 0: st["mism"] = st["tot"] = 0
        return lines + [scat, head]

    from matplotlib.animation import FuncAnimation
    interval = max(5, int(1000 / a.fps))
    anim = FuncAnimation(fig, step, interval=interval, blit=False, cache_frame_data=False)
    fig._keep_anim = anim       # giữ tham chiếu khỏi bị GC
    print("GUI 3D dang chay — dong cua so hoac Q de thoat.")
    plt.show()


if __name__ == "__main__":
    main()
