`timescale 1ns / 1ps
//============================================================================
// uart_tx.v — UART transmitter 8N1.
//   start (1 xung) + data -> phat noi tiep. busy=1 trong khi phat.
//   done = xung 1 chu ky khi phat xong 1 byte (sau stop bit).
//============================================================================
module uart_tx #(
    parameter CLKS_PER_BIT = 434
)(
    input  wire       clk,
    input  wire       rst_n,
    input  wire       start,
    input  wire [7:0] data,
    output reg        tx,            // serial out (idle = 1)
    output reg        busy,
    output reg        done           // 1-cycle pulse
);
    localparam S_IDLE=3'd0, S_START=3'd1, S_DATA=3'd2, S_STOP=3'd3;
    reg [2:0]  state;
    reg [15:0] cnt;
    reg [2:0]  bidx;
    reg [7:0]  sh;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state<=S_IDLE; tx<=1'b1; busy<=0; done<=0; cnt<=0; bidx<=0; sh<=0;
        end else begin
            done <= 1'b0;
            case (state)
                S_IDLE: begin
                    tx<=1'b1; busy<=1'b0;
                    if (start) begin sh<=data; busy<=1'b1; cnt<=0; state<=S_START; end
                end
                S_START: begin                          // start bit = 0
                    tx<=1'b0;
                    if (cnt==CLKS_PER_BIT-1) begin cnt<=0; bidx<=0; state<=S_DATA; end
                    else cnt<=cnt+1'b1;
                end
                S_DATA: begin
                    tx<=sh[bidx];                        // LSB first
                    if (cnt==CLKS_PER_BIT-1) begin
                        cnt<=0;
                        if (bidx==3'd7) state<=S_STOP; else bidx<=bidx+1'b1;
                    end else cnt<=cnt+1'b1;
                end
                S_STOP: begin                            // stop bit = 1
                    tx<=1'b1;
                    if (cnt==CLKS_PER_BIT-1) begin cnt<=0; done<=1'b1; busy<=1'b0; state<=S_IDLE; end
                    else cnt<=cnt+1'b1;
                end
                default: state<=S_IDLE;
            endcase
        end
    end
endmodule
