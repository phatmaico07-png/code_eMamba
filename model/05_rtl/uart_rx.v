`timescale 1ns / 1ps
//============================================================================
// uart_rx.v — UART receiver 8N1 (8 data, no parity, 1 stop).
//   CLKS_PER_BIT = f_clk / baud.  Vd: 50MHz / 115200 = 434.
//   Double-FF sync ngo vao; lay mau giua bit. valid = xung 1 chu ky khi xong byte.
//============================================================================
module uart_rx #(
    parameter CLKS_PER_BIT = 434
)(
    input  wire       clk,
    input  wire       rst_n,
    input  wire       rx,            // serial in (idle = 1)
    output reg  [7:0] data,
    output reg        valid          // 1-cycle pulse
);
    localparam S_IDLE=3'd0, S_START=3'd1, S_DATA=3'd2, S_STOP=3'd3, S_DONE=3'd4;
    reg [2:0]  state;
    reg [15:0] cnt;
    reg [2:0]  bidx;
    reg        rx_d, rx_s;           // sync 2-FF

    always @(posedge clk) begin rx_d <= rx; rx_s <= rx_d; end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state<=S_IDLE; valid<=0; cnt<=0; bidx<=0; data<=0;
        end else begin
            valid <= 1'b0;
            case (state)
                S_IDLE: begin
                    cnt<=0; bidx<=0;
                    if (!rx_s) state<=S_START;          // start bit (low)
                end
                S_START: begin                          // xac nhan giua start bit
                    if (cnt==(CLKS_PER_BIT>>1)) begin
                        if (!rx_s) begin cnt<=0; state<=S_DATA; end
                        else        state<=S_IDLE;       // nhieu, bo
                    end else cnt<=cnt+1'b1;
                end
                S_DATA: begin
                    if (cnt==CLKS_PER_BIT-1) begin
                        cnt<=0; data[bidx]<=rx_s;        // LSB first
                        if (bidx==3'd7) begin bidx<=0; state<=S_STOP; end
                        else bidx<=bidx+1'b1;
                    end else cnt<=cnt+1'b1;
                end
                S_STOP: begin
                    if (cnt==CLKS_PER_BIT-1) begin cnt<=0; valid<=1'b1; state<=S_DONE; end
                    else cnt<=cnt+1'b1;
                end
                S_DONE: state<=S_IDLE;
                default: state<=S_IDLE;
            endcase
        end
    end
endmodule
