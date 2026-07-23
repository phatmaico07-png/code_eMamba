# Makefile — BAN MOI: weight NUNG SAN (LOAD_MODE=0), AXI-Lite. KHONG con load_param.
all: emamba

# CLI test (TXT): sudo ./emamba input.txt output.txt golden_rtl.txt
emamba: main.c emamba_hw.c emamba_hw.h
	gcc -O2 -o emamba main.c emamba_hw.c

# Qt demo 3-panel: sudo ./emamba_qt   (can Qt5: sudo apt install qtbase5-dev)
qt: emamba_demo3panel_qt.cpp
	g++ -O2 -fPIC -std=c++17 emamba_demo3panel_qt.cpp -o emamba_qt \
		`pkg-config --cflags --libs Qt5Widgets`

clean:
	rm -f emamba emamba_qt
