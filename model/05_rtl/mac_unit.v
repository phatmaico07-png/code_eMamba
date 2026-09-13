`timescale 1ns / 1ps
//============================================================================
// mac_unit.v — INT8 Multiply-Accumulate Unit (FSMD style)
//
// Tách riêng:
//   Block 1: Datapath registers (sequential)
//   Block 2: Combinational logic (multiply, MUX, add, saturate)
//
// acc_out = acc_sel + (a × b)
//   clear_acc=1:  acc_sel = 0         (bắt đầu tích lũy mới)
//   use_acc_in=1: acc_sel = acc_in    (accumulator từ ngoài / bias pre-load)
//   else:         acc_sel = acc_out   (feedback nội bộ)
//============================================================================

module mac_unit #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 24
)(
    input  wire                        clk,
    input  wire                        rst_n,

    // Control
    input  wire                        en,
    input  wire                        clear_acc,

    // Data input
    input  wire signed [DATA_WIDTH-1:0]  a,
    input  wire signed [DATA_WIDTH-1:0]  b,
    input  wire signed [ACC_WIDTH-1:0]   acc_in,
    input  wire                          use_acc_in,

    // Data output
    output reg  signed [ACC_WIDTH-1:0]   acc_out,
    output reg                           valid_out
);

    // ══════════════════════════════════════════════════════════════
    // COMBINATIONAL LOGIC — multiply, MUX, add, saturate
    // ══════════════════════════════════════════════════════════════
    localparam PROD_WIDTH = 2 * DATA_WIDTH;
    localparam signed [ACC_WIDTH-1:0] SAT_MAX = (1 << (ACC_WIDTH - 1)) - 1;
    localparam signed [ACC_WIDTH-1:0] SAT_MIN = -(1 << (ACC_WIDTH - 1));

    // Multiply
    wire signed [PROD_WIDTH-1:0] product;
    assign product = a * b;

    // MUX: chọn accumulator source
    wire signed [ACC_WIDTH-1:0] acc_sel;
    assign acc_sel = clear_acc  ? {ACC_WIDTH{1'b0}} :
                     use_acc_in ? acc_in :
                                  acc_out;            // feedback

    // Add with overflow detection
    wire signed [ACC_WIDTH:0] sum_ext;  // 1 extra bit
    assign sum_ext = {{(ACC_WIDTH - PROD_WIDTH + 1){product[PROD_WIDTH-1]}}, product}
                   + {acc_sel[ACC_WIDTH-1], acc_sel};

    // Saturate
    wire overflow_pos = (sum_ext > SAT_MAX);
    wire overflow_neg = (sum_ext < SAT_MIN);

    wire signed [ACC_WIDTH-1:0] sum;
    assign sum = overflow_pos ? SAT_MAX :
                 overflow_neg ? SAT_MIN :
                                sum_ext[ACC_WIDTH-1:0];

    // ══════════════════════════════════════════════════════════════
    // SEQUENTIAL LOGIC — pipeline register
    // ══════════════════════════════════════════════════════════════
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_out   <= {ACC_WIDTH{1'b0}};
            valid_out <= 1'b0;
        end else if (en) begin
            acc_out   <= sum;
            valid_out <= 1'b1;
        end else begin
            valid_out <= 1'b0;
        end
    end

endmodule
