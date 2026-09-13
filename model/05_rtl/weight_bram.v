`timescale 1ns / 1ps
//============================================================================
// weight_bram.v — BRAM wrapper cho weight/bias storage
//
// Vivado infer BRAM nhờ (* ram_style = "block" *).  Synchronous read 1 cycle.
//
//   LOAD_MODE = 0 : $readmemh init (ROM) — đường ĐÃ VERIFY 0/455088, write-bus
//                   bị tham số hoá triệt tiêu hoàn toàn (synth bỏ).
//   LOAD_MODE = 1 : runtime load qua WRITE-BUS dùng chung (PS DMA nạp).
//
// Write-bus BYTE-SERIAL (dồn MSB-first): mỗi w_en đẩy 1 byte vào wacc; đủ
// WBYTES byte = 1 word → ghi mem[waddr], waddr++. w_seg_rst mở 1 segment mới
// (đặt waddr về nửa b0|b1).  Mọi BRAM nghe chung bus, chỉ BRAM có w_sel==MY_ID
// mới phản ứng.  Thứ tự byte khớp $readmemh (cặp hex đầu = MSB của word).
//============================================================================

module weight_bram #(
    parameter DEPTH        = 1024,
    parameter WIDTH        = 8,
    parameter SHARED       = 0,        // 1 = 2 block trong 1 BRAM (double-depth)
    parameter INIT_FILE    = "none",
    parameter INIT_FILE_B1 = "none",   // block1 (SHARED=1)
    parameter LOAD_MODE    = 0,        // 0 = $readmemh (verified) ; 1 = runtime
    parameter [4:0] MY_ID  = 5'd0      // id khớp load-map sel
)(
    input  wire                                          clk,
    input  wire [$clog2(DEPTH > 1 ? DEPTH : 2)-1:0]     addr,
    input  wire                                          block_sel, // 0=block0,1=block1
    output reg  [WIDTH-1:0]                              dout,
    // ── write-bus dùng chung (chỉ tác dụng khi LOAD_MODE=1) ──
    input  wire                                          w_en,      // 1 byte hợp lệ
    input  wire                                          w_seg_rst, // bắt đầu segment
    input  wire [4:0]                                    w_sel,     // chọn BRAM
    input  wire                                          w_half,    // 0=b0,1=b1 (SHARED)
    input  wire [7:0]                                    w_byte     // byte (MSB-first)
);
    localparam WBYTES = (WIDTH + 7) / 8;

    generate
        if (SHARED) begin : gen_shared
            // Double-depth: block0 [0..DEPTH-1], block1 [DEPTH..2*DEPTH-1]
            localparam TOTAL_DEPTH = DEPTH * 2;
            localparam ADDR_W = $clog2(TOTAL_DEPTH > 1 ? TOTAL_DEPTH : 2);

            (* ram_style = "distributed" *)   // LUTRAM (0 BRAM) — lean core fit thoai mai
            reg [WIDTH-1:0] mem [0:TOTAL_DEPTH-1];

            initial begin
                if (LOAD_MODE == 0 && INIT_FILE    != "none") $readmemh(INIT_FILE,    mem, 0,     DEPTH-1);
                if (LOAD_MODE == 0 && INIT_FILE_B1 != "none") $readmemh(INIT_FILE_B1, mem, DEPTH, TOTAL_DEPTH-1);
            end

            wire [ADDR_W-1:0] full_addr = (block_sel ? DEPTH[ADDR_W-1:0] : {ADDR_W{1'b0}}) + addr;

            // ── write FSM byte-serial ──
            reg  [WIDTH-1:0] wacc;
            reg  [$clog2(WBYTES + 1 > 1 ? WBYTES + 1 : 2)-1:0] bcnt;
            reg  [ADDR_W-1:0] waddr;
            wire             sel_me  = (w_sel == MY_ID);
            wire [WIDTH+7:0] acc_cat = {wacc, w_byte};            // dồn trái 1 byte (MSB-first)
            wire [WIDTH-1:0] nxt_acc = acc_cat[WIDTH-1:0];        // giữ WIDTH bit thấp — an toàn mọi WIDTH>=8

            always @(posedge clk) begin
                if (LOAD_MODE && sel_me) begin
                    if (w_seg_rst) begin
                        bcnt  <= 0;
                        waddr <= w_half ? DEPTH[ADDR_W-1:0] : {ADDR_W{1'b0}};
                    end else if (w_en) begin
                        wacc <= nxt_acc;
                        if (bcnt == WBYTES - 1) begin
                            mem[waddr] <= nxt_acc;
                            waddr <= waddr + 1'b1;
                            bcnt  <= 0;
                        end else begin
                            bcnt <= bcnt + 1'b1;
                        end
                    end
                end
                dout <= mem[full_addr];
            end
        end else begin : gen_single
            localparam ADDR_W = $clog2(DEPTH > 1 ? DEPTH : 2);

            (* ram_style = "distributed" *)   // LUTRAM (0 BRAM) — lean core fit thoai mai
            reg [WIDTH-1:0] mem [0:DEPTH-1];

            initial begin
                if (LOAD_MODE == 0 && INIT_FILE != "none") $readmemh(INIT_FILE, mem);
            end

            // ── write FSM byte-serial ──
            reg  [WIDTH-1:0] wacc;
            reg  [$clog2(WBYTES + 1 > 1 ? WBYTES + 1 : 2)-1:0] bcnt;
            reg  [ADDR_W-1:0] waddr;
            wire             sel_me  = (w_sel == MY_ID);
            wire [WIDTH+7:0] acc_cat = {wacc, w_byte};            // dồn trái 1 byte (MSB-first)
            wire [WIDTH-1:0] nxt_acc = acc_cat[WIDTH-1:0];        // giữ WIDTH bit thấp — an toàn mọi WIDTH>=8

            always @(posedge clk) begin
                if (LOAD_MODE && sel_me) begin
                    if (w_seg_rst) begin
                        bcnt  <= 0;
                        waddr <= {ADDR_W{1'b0}};
                    end else if (w_en) begin
                        wacc <= nxt_acc;
                        if (bcnt == WBYTES - 1) begin
                            mem[waddr] <= nxt_acc;
                            waddr <= waddr + 1'b1;
                            bcnt  <= 0;
                        end else begin
                            bcnt <= bcnt + 1'b1;
                        end
                    end
                end
                dout <= mem[addr];
            end
        end
    endgenerate

endmodule
