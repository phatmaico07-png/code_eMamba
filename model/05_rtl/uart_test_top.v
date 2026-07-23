`timescale 1ns / 1ps
//============================================================================
// uart_test_top.v — MODEL TEST UART giong emamba_uart_top nhung BO core eMamba.
//   Nhan IN_B byte (320) -> tra OUT_B byte (57), out[i] = in[i] XOR 0xA5.
//   Dung de co lap & kiem duong UART/FIFO/FSM/RESYNC tren board (khong dinh core).
//   Cung clk/UART/8N1/CLKS_PER_BIT/RESYNC nhu emamba_uart_top.
//============================================================================
module uart_test_top #(
    parameter CLKS_PER_BIT = 434,
    parameter IN_B  = 320,
    parameter OUT_B = 57,
    parameter RXF_DEPTH = 512,
    parameter TXF_DEPTH = 64,
    parameter [7:0] XORK = 8'hA5     // out = in XOR XORK
)(
    input  wire clk,
    input  wire rst_n,
    input  wire uart_rx_pin,
    output wire uart_tx_pin
);
    // ── UART RX -> RX FIFO ──
    wire [7:0] rx_byte; wire rx_valid;
    uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_rx (
        .clk(clk), .rst_n(rst_n), .rx(uart_rx_pin), .data(rx_byte), .valid(rx_valid));
    wire [7:0] rxf_dout; wire rxf_empty, rxf_full; reg rxf_rd;
    sync_fifo #(.WIDTH(8), .DEPTH(RXF_DEPTH)) u_rxf (
        .clk(clk), .rst_n(rst_n), .wr_en(rx_valid), .din(rx_byte), .full(rxf_full),
        .rd_en(rxf_rd), .dout(rxf_dout), .empty(rxf_empty), .count());

    // ── TX FIFO -> drain -> UART TX ──
    reg txf_wr; reg [7:0] txf_din; wire txf_full;
    reg txf_rd; wire [7:0] txf_dout; wire txf_empty;
    sync_fifo #(.WIDTH(8), .DEPTH(TXF_DEPTH)) u_txf (
        .clk(clk), .rst_n(rst_n), .wr_en(txf_wr), .din(txf_din), .full(txf_full),
        .rd_en(txf_rd), .dout(txf_dout), .empty(txf_empty), .count());
    reg tx_start; reg [7:0] tx_byte; wire tx_busy, tx_done;
    uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_tx (
        .clk(clk), .rst_n(rst_n), .start(tx_start), .data(tx_byte),
        .tx(uart_tx_pin), .busy(tx_busy), .done(tx_done));
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin tx_start<=0; txf_rd<=0; tx_byte<=0; end
        else begin
            tx_start<=0; txf_rd<=0;
            if (!tx_busy && !tx_start && !txf_empty) begin
                tx_byte<=txf_dout; tx_start<=1; txf_rd<=1;
            end
        end
    end

    // ── FSM: gom IN_B -> day OUT_B (out=in XOR XORK), co RESYNC khung ──
    reg [7:0] in_buf [0:IN_B-1];
    reg [9:0] rcnt, tcnt;
    reg [19:0] idle_cnt;
    localparam [19:0] RESYNC_GAP = 20'd50000;
    localparam S_RX=2'd0, S_PUSH=2'd1;
    reg [1:0] st;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin st<=S_RX; rcnt<=0; tcnt<=0; rxf_rd<=0; txf_wr<=0; txf_din<=0; idle_cnt<=0; end
        else begin
            rxf_rd<=1'b0; txf_wr<=1'b0;
            case (st)
                S_RX: begin
                    if (rxf_empty) begin
                        if (idle_cnt < RESYNC_GAP) idle_cnt <= idle_cnt + 1'b1;
                        else rcnt <= 0;                         // RESYNC dau khung
                    end else begin
                        idle_cnt <= 0;
                        if (!rxf_rd) begin
                            in_buf[rcnt] <= rxf_dout; rxf_rd <= 1'b1;
                            if (rcnt==IN_B-1) begin rcnt<=0; tcnt<=0; st<=S_PUSH; end
                            else rcnt<=rcnt+1'b1;
                        end
                    end
                end
                S_PUSH: if (!txf_full) begin
                    txf_din <= in_buf[tcnt] ^ XORK;             // bien doi de chung minh logic chay
                    txf_wr  <= 1'b1;
                    if (tcnt==OUT_B-1) st<=S_RX; else tcnt<=tcnt+1'b1;
                end
                default: st<=S_RX;
            endcase
        end
    end
endmodule
