`timescale 1ns / 1ps
//============================================================================
// mean_pool.v — Mean Pool over L tokens: sum / L
//
// Accumulates L tokens (D INT8 each), then divides by L.
// For L=16: divide = shift right 4 (exact, no rounding needed).
//
// Interface: streaming tokens in, single vector out.
//   in_valid/in_ready: 1 token per handshake
//   frame_start: reset accumulator for new sequence
//   out_valid: goes high after L tokens accumulated
//
// Latency: L cycles (accumulate) + 1 cycle (shift) = 17 cycles
// Resource: D adders (12-bit) + D shift registers = minimal
//============================================================================

module mean_pool #(
    parameter D     = 20,
    parameter L     = 16,
    parameter SHIFT = 4     // log2(L) = log2(16) = 4
)(
    input  wire                clk,
    input  wire                rst_n,
    input  wire                frame_start,  // reset accumulator

    input  wire                in_valid,
    output wire                in_ready,
    input  wire [D*8-1:0]      in_data,      // 1 token = D INT8

    output reg                 out_valid,
    input  wire                out_ready,
    output reg  [D*8-1:0]      out_data      // mean = D INT8
);

    // Accumulator: D elements, ĐỦ chứa tổng L token INT8 = 8 + log2(L) bit.
    // PHẢI độc lập với SHIFT (SHIFT là dịch scale của mean, KHÁC với độ rộng tổng).
    // Trước đây = 8+SHIFT → khi SHIFT đổi 4→2 (chỉnh scale) acc bị hẹp INT10 → tràn
    // ở frame có |Σ16 token| > 511 (vd frame 7170 kênh 16 = 541). Nay cố định INT12.
    localparam ACC_W = 8 + $clog2(L);  // 12 for L=16
    reg signed [ACC_W-1:0] acc [0:D-1];
    reg [$clog2(L):0] tok_cnt;
    reg busy;

    assign in_ready = !busy && !out_valid;

    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tok_cnt   <= 0;
            busy      <= 0;
            out_valid <= 0;
            out_data  <= 0;
            for (i = 0; i < D; i = i + 1)
                acc[i] <= 0;
        end else begin
            // Output consumed
            if (out_valid && out_ready)
                out_valid <= 0;

            // Frame start: reset
            if (frame_start) begin
                tok_cnt <= 0;
                out_valid <= 0;
                for (i = 0; i < D; i = i + 1)
                    acc[i] <= 0;
            end

            // Accumulate token
            if (in_valid && in_ready) begin
                for (i = 0; i < D; i = i + 1)
                    acc[i] <= acc[i] + $signed(in_data[i*8 +: 8]);
                tok_cnt <= tok_cnt + 1;

                // Last token → compute mean
                if (tok_cnt == L - 1) begin
                    busy <= 1;
                end
            end

            // Shift (1 cycle after last accumulate)
            if (busy) begin
                for (i = 0; i < D; i = i + 1) begin : mean_shift
                    reg signed [ACC_W-1:0] shifted;
                    shifted = (acc[i] + (1 <<< (SHIFT-1))) >>> SHIFT;
                    // Clamp to INT8
                    if (shifted > 127)       out_data[i*8 +: 8] <= 8'sd127;
                    else if (shifted < -128) out_data[i*8 +: 8] <= -8'sd128;
                    else                     out_data[i*8 +: 8] <= shifted[7:0];
                end
                out_valid <= 1;
                busy <= 0;
            end
        end
    end

endmodule
