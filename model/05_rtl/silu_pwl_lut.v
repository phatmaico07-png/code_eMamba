`timescale 1ns / 1ps
//============================================================================
// silu_pwl_lut.v — SiLU bằng PWL (tra bp/sl/ic + nhân DSP, SONG SONG 1 chu kỳ).
//
//   Interface GIỐNG direct_lut (start/x_in/y_out/done, block_sel, write-bus) để
//   thay trực tiếp trong mamba_block mà KHÔNG đổi FSM. Khác direct_lut (256-ô,
//   đọc tuần tự ~N_ELEM chu kỳ) ở chỗ: tính PWL song song -> done sau 2 chu kỳ.
//
//   y[e] = clamp8( (sl[seg]*x[e] + ic[seg] + 2^(FRAC-1)) >>> FRAC )
//   seg  = chỉ số đoạn lớn nhất có bp[seg] <= x  (kẹp [0, N_SEG-1])
//   KHÔNG xử lý biên zero/identity -> KHỚP int_reference silu_lut256 (clip seg +
//   clampq). Do đó BIT-EXACT với bảng direct_lut sinh từ cùng silu_lut256.
//
//   SHARED=1: bảng block0 + block1, chọn theo block_sel.
//   LOAD_MODE=0: $readmemh (sim). LOAD_MODE=1: nạp runtime qua write-bus (board).
//============================================================================
module silu_pwl_lut #(
    parameter N_ELEM      = 40,
    parameter SHARED      = 0,
    parameter N_SEG       = 17,
    parameter FRAC        = 7,
    parameter BP_FILE     = "none", parameter SL_FILE     = "none", parameter IC_FILE     = "none",
    parameter BP_FILE_B1  = "none", parameter SL_FILE_B1  = "none", parameter IC_FILE_B1  = "none",
    parameter LOAD_MODE   = 0,
    parameter [4:0] MY_ID_BP = 5'd0,
    parameter [4:0] MY_ID_SL = 5'd0,
    parameter [4:0] MY_ID_IC = 5'd0
)(
    input  wire                clk,
    input  wire                rst_n,
    input  wire                block_sel,
    input  wire                start,
    input  wire [N_ELEM*8-1:0] x_in,
    output wire [N_ELEM*8-1:0] y_out,
    output reg                 done,
    // ── write-bus dùng chung (chỉ tác dụng khi LOAD_MODE=1) ──
    input  wire                w_en,
    input  wire                w_seg_rst,
    input  wire [4:0]          w_sel,
    input  wire                w_half,    // 1 = block1 (SHARED)
    input  wire [7:0]          w_byte
);
    // ───────────── BẢNG bp/sl/ic (block0 + block1) ─────────────
    reg signed [7:0]  bp0 [0:N_SEG];   reg signed [15:0] sl0 [0:N_SEG-1]; reg signed [15:0] ic0 [0:N_SEG-1];
    reg signed [7:0]  bp1 [0:N_SEG];   reg signed [15:0] sl1 [0:N_SEG-1]; reg signed [15:0] ic1 [0:N_SEG-1];
    initial begin
        if (LOAD_MODE==0 && BP_FILE   !="none") $readmemh(BP_FILE,    bp0);
        if (LOAD_MODE==0 && SL_FILE   !="none") $readmemh(SL_FILE,    sl0);
        if (LOAD_MODE==0 && IC_FILE   !="none") $readmemh(IC_FILE,    ic0);
        if (LOAD_MODE==0 && SHARED && BP_FILE_B1!="none") $readmemh(BP_FILE_B1, bp1);
        if (LOAD_MODE==0 && SHARED && SL_FILE_B1!="none") $readmemh(SL_FILE_B1, sl1);
        if (LOAD_MODE==0 && SHARED && IC_FILE_B1!="none") $readmemh(IC_FILE_B1, ic1);
    end

    // ───────────── nạp runtime (LOAD_MODE=1): w_half chọn block1 ─────────────
    generate if (LOAD_MODE) begin : gen_wr
        reg [$clog2(N_SEG+2)-1:0] bpa;
        always @(posedge clk) if (w_sel==MY_ID_BP) begin
            if (w_seg_rst) bpa <= 0;
            else if (w_en) begin if (w_half) bp1[bpa]<=w_byte; else bp0[bpa]<=w_byte; bpa<=bpa+1'b1; end
        end
        reg [$clog2(N_SEG+1)-1:0] sla; reg sbc; reg [7:0] shi;
        always @(posedge clk) if (w_sel==MY_ID_SL) begin
            if (w_seg_rst) begin sla<=0; sbc<=0; end
            else if (w_en) begin
                if (!sbc) begin shi<=w_byte; sbc<=1; end
                else begin if (w_half) sl1[sla]<={shi,w_byte}; else sl0[sla]<={shi,w_byte}; sla<=sla+1'b1; sbc<=0; end
            end
        end
        reg [$clog2(N_SEG+1)-1:0] ica; reg ibc; reg [7:0] ihi;
        always @(posedge clk) if (w_sel==MY_ID_IC) begin
            if (w_seg_rst) begin ica<=0; ibc<=0; end
            else if (w_en) begin
                if (!ibc) begin ihi<=w_byte; ibc<=1; end
                else begin if (w_half) ic1[ica]<={ihi,w_byte}; else ic0[ica]<={ihi,w_byte}; ica<=ica+1'b1; ibc<=0; end
            end
        end
    end endgenerate

    // ───────────── PWL song song (tổ hợp từ x_reg) ─────────────
    reg [N_ELEM*8-1:0] x_reg;
    wire [N_ELEM*8-1:0] y_comb;
    genvar e;
    generate for (e=0; e<N_ELEM; e=e+1) begin : gen_el
        wire signed [7:0] xv = $signed(x_reg[e*8 +: 8]);
        reg [$clog2(N_SEG>1?N_SEG:2)-1:0] seg;
        integer si;
        always @(*) begin
            seg = 0;
            for (si=0; si<N_SEG-1; si=si+1)
                if (xv >= (block_sel ? bp1[si+1] : bp0[si+1])) seg = si+1;
        end
        wire signed [15:0] sv = block_sel ? sl1[seg] : sl0[seg];
        wire signed [15:0] iv = block_sel ? ic1[seg] : ic0[seg];
        (* use_dsp = "yes" *) wire signed [23:0] prod = sv * xv;
        wire signed [23:0] sm = prod + {{8{iv[15]}}, iv};
        wire signed [23:0] sh = (sm + (1 <<< (FRAC-1))) >>> FRAC;
        assign y_comb[e*8 +: 8] = (sh > 24'sd127)  ?  8'sd127 :
                                  (sh < -24'sd128) ? -8'sd128 : sh[7:0];
    end endgenerate

    // ───────────── handshake: start -> 2 chu kỳ -> done (y giữ) ─────────────
    reg [N_ELEM*8-1:0] y_reg;
    reg st_d;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin done<=1'b0; st_d<=1'b0; x_reg<=0; y_reg<=0; end
        else begin
            done <= 1'b0;
            st_d <= start;
            if (start) x_reg <= x_in;
            if (st_d)  begin y_reg <= y_comb; done <= 1'b1; end
        end
    end
    assign y_out = y_reg;
endmodule
