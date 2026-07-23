#!/bin/bash
# stress.sh — kiem thu do on dinh eMamba accelerator tren KV260.
#   Moi vong: reload bitstream -> load_param (verify WSUM/SHIFT KHOP) -> chay emamba N lan.
#   Dem PASS/FAIL, theo doi throughput (us/frame) de phat hien sai so / flaky timing.
# Chay tu ~/DATN:   sudo ./stress.sh [N_run_moi_vong] [N_reload]
#   vd: sudo ./stress.sh 20 5   -> 5 reload x (1 load_param + 20 emamba) = 100 lan inference
set -u

N=${1:-20}                      # so lan emamba moi vong reload
R=${2:-5}                       # so lan reload bitstream
BITNAME=emamba_soc_wrapper.bit
W=weights.bin
IN=input.bin
GOLD=golden.bin
OUT=/tmp/emamba_out.bin
FPGA=/sys/class/fpga_manager/fpga0

pass=0; fail=0; stall_tot=0
us_min=99999; us_max=0
declare -a md5s=()

need() { [ -f "$1" ] || { echo "THIEU file: $1"; exit 1; }; }
need "$W"; need "$IN"; need "$GOLD"; need ./load_param; need ./emamba

reload_bit() {
    echo 0          > "$FPGA/flags"
    echo "$BITNAME" > "$FPGA/firmware"
    sleep 0.3
    local st; st=$(cat "$FPGA/state")
    printf "  fpga_state=%s" "$st"
    [ "$st" = "operating" ] || { printf "  *** STATE LOI ***\n"; return 1; }
    printf "\n"; return 0
}

check_load() {                  # load_param + verify; tra 0 neu KHOP
    local lp; lp=$(./load_param "$W" 2>&1)
    if echo "$lp" | grep -q "WSUM doc lai = 0xBE8FCF11" && ! echo "$lp" | grep -q "SAI"; then
        echo "  load_param: WSUM+SHIFT KHOP"; return 0
    else
        echo "  load_param: *** SAI ***"; echo "$lp" | grep -E "WSUM|SHIFT"; return 1
    fi
}

run_emamba() {                  # 1 lan inference; tra 0 neu PASS bit-exact + file DU 455088 byte
    local em us sz st; em=$(./emamba "$IN" "$OUT" "$GOLD" 2>&1)
    sz=$(stat -c%s "$OUT" 2>/dev/null || echo 0)
    us=$(echo "$em" | grep -oE '[0-9]+\.[0-9]+ us/frame' | grep -oE '[0-9]+\.[0-9]+' | head -1)
    local m; m=$(md5sum "$OUT" | cut -d' ' -f1); md5s+=("$m")
    st=$(echo "$em" | grep -oE 'Phuc hoi [0-9]+' | grep -oE '[0-9]+'); [ -n "$st" ] && stall_tot=$((stall_tot+st))
    [ -n "$us" ] && { awk "BEGIN{exit !($us<$us_min)}" && us_min=$us; awk "BEGIN{exit !($us>$us_max)}" && us_max=$us; }
    if echo "$em" | grep -qE "TONG lech 0/[0-9]+ +-> PASS" && [ "$sz" -eq 455088 ]; then
        return 0
    else
        echo "    *** FAIL: size=$sz $(echo "$em" | grep -E 'TONG|STALL')"; return 1
    fi
}

echo "=========================================================="
echo " STRESS eMamba | $R reload x $N inference = $((R*N)) lan"
echo "=========================================================="
for r in $(seq 1 "$R"); do
    echo "[reload $r/$R]"
    reload_bit            || { fail=$((fail+1)); continue; }
    check_load            || { fail=$((fail+1)); continue; }
    okc=0
    for n in $(seq 1 "$N"); do
        if run_emamba; then okc=$((okc+1)); pass=$((pass+1)); else fail=$((fail+1)); fi
    done
    echo "  -> vong $r: $okc/$N PASS"
done

# kiem tra moi output file giong het nhau (deterministic)
uniq_md5=$(printf '%s\n' "${md5s[@]}" | sort -u | wc -l)

echo "=========================================================="
echo " TONG KET: PASS=$pass  FAIL=$fail  (tong $((R*N)) inference)"
echo " Throughput: $us_min - $us_max us/frame"
echo " Stall phuc hoi (re-arm START): $stall_tot lan / $((R*N)) run"
echo " Output md5 khac nhau: $uniq_md5 (1 = deterministic hoan toan)"
if [ "$fail" -eq 0 ] && [ "$uniq_md5" -eq 1 ]; then
    echo " ==> ON DINH 100% : bit-exact, lap lai, ben qua reload."
else
    echo " ==> CO VAN DE : xem log tren."
fi
echo "=========================================================="
