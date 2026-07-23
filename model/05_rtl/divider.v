`timescale 1ns / 1ps
//============================================================================
// divider.v — Non-restoring iterative divider (bit-exact với divider.py)
//
// Signed INT24 numerator / unsigned INT9 denominator → signed Q1.15 quotient.
// 24-cycle iteration (1 bit per cycle, MSB first), sign-magnitude.
//
// Algorithm per cycle:
//   rem = (rem << 1) | next_bit_of_abs_num
//   if rem >= den: rem -= den;  quot_bit = 1
//   else:                        quot_bit = 0
//
// After 24 iterations: apply sign, saturate to [-32768, 32767].
// Division-by-zero: quotient = 0 (bypass).
//
// Interface:
//   start (1 pulse) → 25 cycles → done (1 pulse) + quotient valid
//   Latency: 1 (latch) + 24 (iterate) = 25 cycles
//
// Usage in range_norm: N instances parallel, 1 per compute unit.
//   numerator = (x[i] - mean) << 15   (centered << 15 = INT24)
//   denominator = max - min            (range, positive INT9)
//============================================================================

module divider #(
    parameter NUM_BITS  = 24,    // numerator bit-width (signed)
    parameter DEN_BITS  = 9,     // denominator bit-width (unsigned, positive)
    parameter QUOT_BITS = 16     // quotient bit-width (signed Q1.15)
)(
    input  wire                         clk,
    input  wire                         rst_n,

    input  wire                         start,        // pulse to begin division
    input  wire signed [NUM_BITS-1:0]   numerator,    // signed dividend
    input  wire [DEN_BITS-1:0]          denominator,  // unsigned divisor (positive)

    output reg                          done,         // 1-cycle pulse when result ready
    output reg  signed [QUOT_BITS-1:0]  quotient      // valid when done=1, held until next start
);

    // ── Internal registers ────────────────────────────────────────
    reg [NUM_BITS-1:0]          abs_num;      // |numerator|
    reg                         sign_neg;     // original sign
    reg [DEN_BITS-1:0]          den_reg;      // latched denominator (safe from 0)
    reg                         den_zero;     // denominator was 0

    reg [DEN_BITS:0]            rem;          // remainder (+1 bit for shifted value)
    reg [NUM_BITS-1:0]          quot_raw;     // unsigned quotient being built

    localparam CYC_W = $clog2(NUM_BITS > 1 ? NUM_BITS : 2);
    localparam [CYC_W-1:0] CYC_LAST = NUM_BITS - 1;  // 23 for NUM_BITS=24
    reg [CYC_W-1:0]             cyc;          // iteration counter 0..NUM_BITS-1
    reg                         running;      // divider active

    // ── Saturation constants ─────────────────────────────────────
    localparam signed [NUM_BITS:0] SAT_POS = (1 << (QUOT_BITS - 1)) - 1;  // +32767
    localparam signed [NUM_BITS:0] SAT_NEG = -(1 << (QUOT_BITS - 1));     // -32768

    // ── Combinational: current iteration ──────────────────────────
    wire [CYC_W-1:0] bit_pos  = NUM_BITS - 1 - cyc;
    wire              next_bit = abs_num[bit_pos];
    wire [DEN_BITS:0] rem_shifted = {rem[DEN_BITS-1:0], next_bit};
    wire              ge = (rem_shifted >= {1'b0, den_reg});

    // ══════════════════════════════════════════════════════════════
    // DEBUG: trace divider start/done
    // ══════════════════════════════════════════════════════════════
    // synthesis translate_off
    // Debug prints disabled for clean output
    // always @(posedge clk) begin
    //     if (start) $display("  [DIV] t=%0t START num=%0d den=%0d", $time, numerator, denominator);
    //     if (done)  $display("  [DIV] t=%0t DONE  quot=%0d cyc_last=%0d", $time, quotient, CYC_LAST);
    // end
    // synthesis translate_on

    // ══════════════════════════════════════════════════════════════
    // SEQUENTIAL LOGIC
    // ══════════════════════════════════════════════════════════════
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            abs_num  <= {NUM_BITS{1'b0}};
            sign_neg <= 1'b0;
            den_reg  <= {{(DEN_BITS-1){1'b0}}, 1'b1};  // 1 (safe default)
            den_zero <= 1'b0;
            rem      <= {(DEN_BITS+1){1'b0}};
            quot_raw <= {NUM_BITS{1'b0}};
            cyc      <= {CYC_W{1'b0}};
            running  <= 1'b0;
            done     <= 1'b0;
            quotient <= {QUOT_BITS{1'b0}};
        end else begin
            // Default: clear done pulse
            done <= 1'b0;

            if (start) begin
                // ── Latch inputs ─────────────────────────────
                sign_neg <= numerator[NUM_BITS-1];
                abs_num  <= numerator[NUM_BITS-1] ? (-numerator) : numerator;
                den_reg  <= (denominator == {DEN_BITS{1'b0}})
                            ? {{(DEN_BITS-1){1'b0}}, 1'b1}   // safe: treat 0 as 1
                            : denominator;
                den_zero <= (denominator == {DEN_BITS{1'b0}});
                rem      <= {(DEN_BITS+1){1'b0}};
                quot_raw <= {NUM_BITS{1'b0}};
                cyc      <= {CYC_W{1'b0}};
                running  <= 1'b1;

            end else if (running) begin
                // ── Non-restoring iteration ───────────────────
                // Trial subtraction
                rem      <= ge ? (rem_shifted - {1'b0, den_reg}) : rem_shifted;
                quot_raw <= {quot_raw[NUM_BITS-2:0], ge};

                if (cyc == CYC_LAST) begin
                    // ── Last iteration: finalize ──────────────
                    running <= 1'b0;
                    done    <= 1'b1;

                    // Apply sign + saturate to Q1.15
                    // Final unsigned quotient = {quot_raw[NUM_BITS-2:0], ge}
                    // (current cycle's ge is the last bit)
                    begin : finalize_quot
                        reg [NUM_BITS-1:0] q_final;
                        reg signed [NUM_BITS:0] q_signed;

                        q_final = {quot_raw[NUM_BITS-2:0], ge};

                        if (den_zero) begin
                            quotient <= {QUOT_BITS{1'b0}};  // div-by-zero → 0
                        end else begin
                            q_signed = sign_neg
                                ? (-{1'b0, q_final})
                                :   {1'b0, q_final};

                            // Saturate to signed QUOT_BITS range
                            if (q_signed > SAT_POS)
                                quotient <= SAT_POS[QUOT_BITS-1:0];   // +32767
                            else if (q_signed < SAT_NEG)
                                quotient <= SAT_NEG[QUOT_BITS-1:0];   // -32768
                            else
                                quotient <= q_signed[QUOT_BITS-1:0];
                        end
                    end

                end else begin
                    cyc <= cyc + 1'b1;
                end
            end
        end
    end

endmodule
