`timescale 1ns / 1ps
//============================================================================
// sync_fifo.v — FIFO dong bo 1 clock (distributed RAM), first-word-fall-through.
//   dout luon hien phan tu dau (head); rd_en day con tro doc. wr_en day vao neu
//   chua full. count = so phan tu dang co.
//============================================================================
module sync_fifo #(
    parameter WIDTH = 8,
    parameter DEPTH = 512
)(
    input  wire             clk,
    input  wire             rst_n,
    input  wire             wr_en,
    input  wire [WIDTH-1:0] din,
    output wire             full,
    input  wire             rd_en,
    output wire [WIDTH-1:0] dout,
    output wire             empty,
    output wire [$clog2(DEPTH):0] count
);
    localparam AW = $clog2(DEPTH);
    (* ram_style="distributed" *) reg [WIDTH-1:0] mem [0:DEPTH-1];
    reg [AW:0] wptr, rptr;                 // 1 bit thua de phan biet full/empty
    wire [AW-1:0] wa = wptr[AW-1:0];
    wire [AW-1:0] ra = rptr[AW-1:0];

    assign empty = (wptr == rptr);
    assign full  = (wa == ra) && (wptr[AW] != rptr[AW]);
    assign count = wptr - rptr;
    assign dout  = mem[ra];                // FWFT: head luon hien

    always @(posedge clk) if (wr_en && !full) mem[wa] <= din;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin wptr<=0; rptr<=0; end
        else begin
            if (wr_en && !full)  wptr <= wptr + 1'b1;
            if (rd_en && !empty) rptr <= rptr + 1'b1;
        end
    end
endmodule
