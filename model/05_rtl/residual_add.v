`timescale 1ns / 1ps
//============================================================================
// residual_add.v — Element-wise scale-aware residual add → INT8
//
// Supports both directions of scale mismatch via separate left/right shifts.
// Math: result = clamp_INT8( (a <<< A_LSHIFT) + (b <<< B_LSHIFT - B_RSHIFT) )
//
// Parameters:
//   A_LSHIFT — left shift on a
//   B_LSHIFT — left shift on b (when b_scale > a_scale, typical Mamba B1)
//   B_RSHIFT — right shift on b with rounding (when b_scale < a_scale)
//
// Backward-compat: legacy port `b_shift` maps to B_RSHIFT when non-zero.
// Purely combinational. Latency: 0 cycles.
//============================================================================

module residual_add #(
    parameter N_ELEM   = 20,
    parameter A_LSHIFT = 0,
    parameter B_LSHIFT = 0,
    parameter B_RSHIFT = 0
)(
    input  wire [N_ELEM*8-1:0]  a_in,
    input  wire [N_ELEM*8-1:0]  b_in,
    // Runtime overrides (0 = use param defaults)
    input  wire [3:0]           a_lshift_rt,
    input  wire [3:0]           b_lshift_rt,
    input  wire [3:0]           b_rshift_rt,
    // Legacy back-compat — interpreted as right shift on b
    input  wire [4:0]           b_shift,
    output wire [N_ELEM*8-1:0]  y_out
);

    wire [3:0] a_ls = (a_lshift_rt != 0) ? a_lshift_rt : A_LSHIFT[3:0];
    wire [3:0] b_ls = (b_lshift_rt != 0) ? b_lshift_rt : B_LSHIFT[3:0];
    wire [3:0] b_rs = (b_rshift_rt != 0) ? b_rshift_rt :
                      (b_shift != 0)     ? b_shift[3:0] : B_RSHIFT[3:0];

    genvar i;
    generate
        for (i = 0; i < N_ELEM; i = i + 1) begin : gen_add
            wire signed [7:0]  av = $signed(a_in[i*8 +: 8]);
            wire signed [7:0]  bv = $signed(b_in[i*8 +: 8]);

            wire signed [15:0] av_ext = $signed({{8{av[7]}}, av});
            wire signed [15:0] bv_ext = $signed({{8{bv[7]}}, bv});

            wire signed [15:0] a_aligned = av_ext <<< a_ls;

            reg  signed [15:0] b_aligned;
            wire signed [15:0] bv_rnd = bv_ext +
                ((b_rs > 0) ? ($signed(16'sd1) <<< (b_rs - 1)) : 16'sd0);
            always @(*) begin
                if (b_ls > 0)
                    b_aligned = bv_ext <<< b_ls;
                else if (b_rs > 0)
                    b_aligned = bv_rnd >>> b_rs;
                else
                    b_aligned = bv_ext;
            end

            wire signed [16:0] sum = $signed({a_aligned[15], a_aligned}) +
                                      $signed({b_aligned[15], b_aligned});

            assign y_out[i*8 +: 8] = (sum >  17'sd127)  ? 8'sd127  :
                                      (sum < -17'sd128) ? -8'sd128 :
                                      sum[7:0];
        end
    endgenerate

endmodule
