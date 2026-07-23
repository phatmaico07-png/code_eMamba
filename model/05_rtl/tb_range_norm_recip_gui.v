`timescale 1ns / 1ps
// tb_range_norm_recip.v — range_norm (divider) vs range_norm_recip (FSMD)
//   1) Do LATENCY tung DUT RIENG (chi kich 1 DUT -> dem toi out_valid, khong dinh race).
//   2) Kiem MISMATCH tren NTOK token (cho divider xong, recip giu output o S_DONE -> so).
module tb_range_norm_recip_gui;
    localparam DIM = 20, N_CU = 20;
    localparam integer NTOK = 5000;
    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    reg  [DIM*8-1:0]  in_data;
    reg  signed [3:0] gamma_shift_in = -4'sd1;   // gamma_shift = -1 (ca divider lan recip)
    reg a_iv, b_iv, a_or, b_or;

    wire a_in_ready, a_out_valid;  wire [DIM*8-1:0] a_out_data;
    range_norm #(.DIM(DIM), .N_CU(N_CU), .GAMMA_SHIFT(0),
        .GAMMA_FILE("D:/DATN/eMamba_Accelerator/rtl/gamma_test.hex"), .BETA_FILE("D:/DATN/eMamba_Accelerator/rtl/beta_test.hex")) u_div (
        .clk(clk), .rst_n(rst_n), .block_sel(1'b0), .gamma_shift_in(gamma_shift_in),
        .in_valid(a_iv), .in_ready(a_in_ready), .in_data(in_data),
        .out_valid(a_out_valid), .out_ready(a_or), .out_data(a_out_data), .wbus(16'd0));

    wire b_in_ready, b_out_valid;  wire [DIM*8-1:0] b_out_data;
    range_norm_recip #(.GAMMA_FILE("D:/DATN/eMamba_Accelerator/rtl/gamma_test.hex"), .BETA_FILE("D:/DATN/eMamba_Accelerator/rtl/beta_test.hex")) u_rec (
        .clk(clk), .rst_n(rst_n), .block_sel(1'b0), .gamma_shift_in(gamma_shift_in),
        .in_valid(b_iv), .in_ready(b_in_ready), .in_data(in_data),
        .out_valid(b_out_valid), .out_ready(b_or), .out_data(b_out_data), .wbus(16'd0));

    integer seed = 32'hC0FFEE;
    function [DIM*8-1:0] rnd_token(input dummy);
        integer j; reg [DIM*8-1:0] t;
        begin t = 0; for (j=0;j<DIM;j=j+1) t[j*8 +: 8] = $random(seed); rnd_token = t; end
    endfunction

    integer i, mismatch, a_cyc, b_cyc;
    reg [DIM*8-1:0] a_cap, b_cap;

    initial begin
        in_data=0; a_iv=0; b_iv=0; a_or=0; b_or=0; mismatch=0;
        repeat (5) @(posedge clk); rst_n=1; @(posedge clk);

        // ===== LATENCY divider (chi kich u_div) =====
        in_data = rnd_token(1'b0);
        @(posedge clk); a_iv=1; @(posedge clk); a_iv=0;
        a_cyc=0; while (!a_out_valid) begin @(posedge clk); a_cyc=a_cyc+1; end
        a_or=1; @(posedge clk); a_or=0; @(posedge clk);

        // ===== LATENCY recip (chi kich u_rec) =====
        @(posedge clk); b_iv=1; @(posedge clk); b_iv=0;
        b_cyc=0; while (!b_out_valid) begin @(posedge clk); b_cyc=b_cyc+1; end
        b_or=1; @(posedge clk); b_or=0; @(posedge clk);

        // ===== MISMATCH NTOK token =====
        for (i = 0; i < NTOK; i = i + 1) begin
            in_data = rnd_token(1'b0);
            @(posedge clk); a_iv=1; b_iv=1;
            @(posedge clk); a_iv=0; b_iv=0;
            while (!a_out_valid) @(posedge clk);   // cho divider (cham hon); recip da xong, giu o S_DONE
            a_cap = a_out_data; b_cap = b_out_data;
            a_or=1; b_or=1; @(posedge clk); a_or=0; b_or=0;
            if (a_cap !== b_cap) begin
                mismatch=mismatch+1;
                if (mismatch<=5) $display("  MISMATCH tok %0d: div=%h recip=%h", i, a_cap, b_cap);
            end
            @(posedge clk);
        end

        $display("================================================");
        $display(" range_norm (divider) vs range_norm_recip (FSMD)");
        $display("   token kiem thu : %0d", NTOK);
        $display("   MISMATCH       : %0d  -> %s", mismatch, (mismatch==0)?"KHOP divider (0 lech)":"co lech");
        $display("   LATENCY divider : %0d cycle/token", a_cyc);
        $display("   LATENCY recip   : %0d cycle/token", b_cyc);
        $display("================================================");
        $finish;
    end
    initial begin #20000000; $display("TIMEOUT"); $finish; end
endmodule
