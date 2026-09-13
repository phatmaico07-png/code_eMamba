`timescale 1ns / 1ps
//============================================================================
// elem_multiply.v — Element-wise INT8 × INT8 → shift → clamp INT8
//
// Used for gating: upper_silu × ssm_output → gated [ED] INT8
//
// Formula per element:
//   product = a[i] × b[i]          (INT8 × INT8 = INT16)
//   y[i] = clamp((product + rnd) >>> shift, INT8)
//
// Combinational (0 cycle) or 1-cycle pipeline.
// Resource: N_ELEM DSP (or LUT multipliers) + shift + clamp
//============================================================================

module elem_multiply #(
    parameter N_ELEM    = 40,
    parameter SHIFT     = 7,        // default requant shift
    parameter PIPELINE  = 1         // 0=combinational, 1=registered output
)(
    input  wire                     clk,
    input  wire [4:0]              shift_right,   // runtime shift (0 = use SHIFT parameter)
    input  wire [N_ELEM*8-1:0]    a_in,          // [N_ELEM] INT8
    input  wire [N_ELEM*8-1:0]    b_in,          // [N_ELEM] INT8
    output wire [N_ELEM*8-1:0]    y_out          // [N_ELEM] INT8
);

    wire [4:0] sh = (shift_right != 0) ? shift_right : SHIFT[4:0];

    // Combinational result
    wire [N_ELEM*8-1:0] y_comb;

    genvar i;
    generate
        for (i = 0; i < N_ELEM; i = i + 1) begin : gen_mul
            wire signed [7:0]  ai = $signed(a_in[i*8 +: 8]);
            wire signed [7:0]  bi = $signed(b_in[i*8 +: 8]);
            (* use_dsp = "yes" *)
            wire signed [15:0] prod = ai * bi;

            // Requant: tính rounding 1 LẦN -> MỘT phép cộng dùng chung (như linear_layer).
            // Trước: 12 nhánh case mỗi nhánh (prod+const) -> use_dsp map 12 cộng×40 = 480
            //        + 40 nhân = 520 DSP. Giờ 1 cộng -> chỉ ~40 DSP (1 nhân/phần tử).
            reg signed [15:0] rnd;
            always @(*) begin
                case (sh)
                    5'd1:    rnd = 16'sd1;
                    5'd2:    rnd = 16'sd2;
                    5'd3:    rnd = 16'sd4;
                    5'd4:    rnd = 16'sd8;
                    5'd5:    rnd = 16'sd16;
                    5'd6:    rnd = 16'sd32;
                    5'd7:    rnd = 16'sd64;
                    5'd8:    rnd = 16'sd128;
                    5'd9:    rnd = 16'sd256;
                    5'd10:   rnd = 16'sd512;
                    5'd11:   rnd = 16'sd1024;
                    5'd12:   rnd = 16'sd2048;
                    default: rnd = 16'sd0;
                endcase
            end
            wire signed [16:0] prnd = prod + rnd;     // MỘT phép cộng (dùng chung mọi shift)
            reg signed [16:0] sv;
            always @(*) begin
                case (sh)
                    5'd1:    sv = prnd >>> 1;
                    5'd2:    sv = prnd >>> 2;
                    5'd3:    sv = prnd >>> 3;
                    5'd4:    sv = prnd >>> 4;
                    5'd5:    sv = prnd >>> 5;
                    5'd6:    sv = prnd >>> 6;
                    5'd7:    sv = prnd >>> 7;
                    5'd8:    sv = prnd >>> 8;
                    5'd9:    sv = prnd >>> 9;
                    5'd10:   sv = prnd >>> 10;
                    5'd11:   sv = prnd >>> 11;
                    5'd12:   sv = prnd >>> 12;
                    default: sv = prnd;            // sh=0: không shift
                endcase
            end

            // Clamp INT8
            assign y_comb[i*8 +: 8] = (sv > 127)  ? 8'sd127 :
                                       (sv < -128) ? -8'sd128 :
                                       sv[7:0];
        end
    endgenerate

    // Output: combinational or pipelined
    generate
        if (PIPELINE) begin : gen_pipe
            reg [N_ELEM*8-1:0] y_reg;
            always @(posedge clk)
                y_reg <= y_comb;
            assign y_out = y_reg;
        end else begin : gen_comb
            assign y_out = y_comb;
        end
    endgenerate

endmodule
