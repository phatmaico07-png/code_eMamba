`timescale 1ns / 1ps
//============================================================================
// relu_unit.v — Vectorized ReLU: y[i] = max(0, x[i])
//
// Purely combinational. N_ELEM parallel INT8 elements.
// For signed INT8: if MSB=1 (negative) → output 0, else pass through.
//
// Resource: N_ELEM MUX2:1 = ~N_ELEM/2 LUT (trivial)
// Latency: 0 cycles (combinational)
//============================================================================

module relu_unit #(
    parameter N_ELEM = 40       // number of parallel elements
)(
    input  wire [N_ELEM*8-1:0]  x_in,
    output wire [N_ELEM*8-1:0]  y_out
);

    genvar i;
    generate
        for (i = 0; i < N_ELEM; i = i + 1) begin : gen_relu
            // Signed INT8: bit 7 = sign. Negative → 0, positive/zero → pass
            assign y_out[i*8 +: 8] = x_in[i*8 + 7] ? 8'd0 : x_in[i*8 +: 8];
        end
    endgenerate

endmodule
