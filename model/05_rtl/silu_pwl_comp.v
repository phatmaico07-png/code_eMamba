`timescale 1ns / 1ps
//============================================================================
// silu_pwl_comp.v — Bộ TÍNH SiLU PWL (KHÔNG chứa bảng; bảng vào qua port từ
//   silu_rom dùng chung). N_ELEM lane song song: tra đoạn + nhân DSP, 2 chu kỳ.
//   Interface start/done GIỐNG direct_lut/silu_pwl_lut -> giữ FSM mamba_block.
//   y[e] = clamp8( (sl[seg]*x[e] + ic[seg] + 2^(FRAC-1)) >>> FRAC )
//   seg  = đoạn lớn nhất có bp[seg] <= x  (kẹp [0,N_SEG-1]); KHÔNG biên 0/identity
//   => BIT-EXACT int_reference silu_lut256.
//============================================================================
module silu_pwl_comp #(
    parameter N_ELEM = 40,
    parameter N_SEG  = 17,
    parameter FRAC   = 7
)(
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    start,
    input  wire [N_ELEM*8-1:0]     x_in,
    // bảng từ silu_rom (đã chọn block) — bp INT16 (gate bp vượt ±127)
    input  wire [(N_SEG+1)*16-1:0] bp_bus,
    input  wire [N_SEG*16-1:0]     sl_bus,
    input  wire [N_SEG*16-1:0]     ic_bus,
    output wire [N_ELEM*8-1:0]     y_out,
    output reg                     done
);
    reg  [N_ELEM*8-1:0] x_reg;
    wire [N_ELEM*8-1:0] y_comb;

    genvar e;
    generate for (e=0; e<N_ELEM; e=e+1) begin : gen_el
        wire signed [7:0] xv = $signed(x_reg[e*8 +: 8]);
        reg [$clog2(N_SEG>1?N_SEG:2)-1:0] seg;
        integer si;
        always @(*) begin
            seg = 0;
            for (si=0; si<N_SEG-1; si=si+1)
                if ($signed({{8{xv[7]}}, xv}) >= $signed(bp_bus[(si+1)*16 +: 16])) seg = si+1;
        end
        wire signed [15:0] sv = $signed(sl_bus[seg*16 +: 16]);
        wire signed [15:0] iv = $signed(ic_bus[seg*16 +: 16]);
        (* use_dsp = "yes" *) wire signed [23:0] prod = sv * xv;
        wire signed [23:0] sm = prod + {{8{iv[15]}}, iv};
        wire signed [23:0] sh = (sm + (1 <<< (FRAC-1))) >>> FRAC;
        assign y_comb[e*8 +: 8] = (sh > 24'sd127)  ?  8'sd127 :
                                  (sh < -24'sd128) ? -8'sd128 : sh[7:0];
    end endgenerate

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
