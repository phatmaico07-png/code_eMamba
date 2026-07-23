`timescale 1ns / 1ps
//============================================================================
// emamba_uart_top.v — Top giao tiep Raspberry Pi qua UART (KHONG dung PS cho data).
//
//   Pi --UART(rx)--> [RX FIFO] --> [gom 320B] --> emamba_top_synth (LOAD_MODE=0,
//   weight nung san) --> [57B] --> [TX FIFO] --> uart_tx --> Pi.
//
//   FIFO tach uart_rx/tx khoi FSM:
//    - RX FIFO: uart_rx ghi bat ky luc nao (ke ca khi core dang chay/dang TX) ->
//      KHONG rot byte; Pi co the stream lien tuc.
//    - TX FIFO + bo drain: FSM do 57B vao FIFO roi quay lai nhan ngay; bo drain
//      tu xa 57B ra uart_tx -> chong RX(frame ke)/TX(frame nay) full-duplex.
//
//   Byte order: in_buf[i]->core_in[i*8+:8] ; core_out[i*8+:8]->out_buf[i].
//   CLKS_PER_BIT = f_clk/baud = 50MHz/115200 = 434.
//============================================================================
module emamba_uart_top #(
    parameter CLKS_PER_BIT = 434,
    parameter IN_B  = 320,
    parameter OUT_B = 57,
    parameter RXF_DEPTH = 512,    // >= IN_B de hung 1 frame den khi core ban
    parameter TXF_DEPTH = 64      // >= OUT_B
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
        .clk(clk), .rst_n(rst_n),
        .wr_en(rx_valid), .din(rx_byte), .full(rxf_full),
        .rd_en(rxf_rd), .dout(rxf_dout), .empty(rxf_empty), .count());

    // ── TX FIFO -> bo drain -> UART TX ──
    reg  txf_wr; reg [7:0] txf_din; wire txf_full;
    reg  txf_rd; wire [7:0] txf_dout; wire txf_empty;
    sync_fifo #(.WIDTH(8), .DEPTH(TXF_DEPTH)) u_txf (
        .clk(clk), .rst_n(rst_n),
        .wr_en(txf_wr), .din(txf_din), .full(txf_full),
        .rd_en(txf_rd), .dout(txf_dout), .empty(txf_empty), .count());

    reg  tx_start; reg [7:0] tx_byte; wire tx_busy, tx_done;
    uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_tx (
        .clk(clk), .rst_n(rst_n), .start(tx_start), .data(tx_byte),
        .tx(uart_tx_pin), .busy(tx_busy), .done(tx_done));

    // bo drain TX FIFO: khi uart_tx ranh va FIFO co data -> day 1 byte
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin tx_start<=1'b0; txf_rd<=1'b0; tx_byte<=8'd0; end
        else begin
            tx_start<=1'b0; txf_rd<=1'b0;
            if (!tx_busy && !tx_start && !txf_empty) begin
                tx_byte <= txf_dout;        // head
                tx_start<= 1'b1;
                txf_rd  <= 1'b1;            // day con tro doc
            end
        end
    end

    // ── core ──
    reg [7:0] in_buf  [0:IN_B-1];
    reg [7:0] out_buf [0:OUT_B-1];
    reg [9:0] rcnt, tcnt;
    // RESYNC khung: RX ranh > RESYNC_GAP chu ky (khoang nghi giua 2 khung) -> reset rcnt=0.
    //   > 1 byte-time (10*CLKS_PER_BIT) de khong reset giua khung; << khoang nghi liên-khung.
    localparam [19:0] RESYNC_GAP = 20'd50000;   // ~1ms @50MHz (1 byte=4340cy; nghi lien-khung >100k)
    reg [19:0] idle_cnt;
    wire [IN_B*8-1:0] core_in;
    genvar gi;
    generate for (gi=0; gi<IN_B; gi=gi+1) assign core_in[gi*8 +: 8] = in_buf[gi]; endgenerate
    reg  core_iv; wire core_ir, core_ov; reg core_or;
    wire [OUT_B*8-1:0] core_out; wire core_busy;
    emamba_top_synth u_core (
        .clk(clk), .rst_n(rst_n),
        .in_valid(core_iv), .in_ready(core_ir), .in_data(core_in),
        .out_valid(core_ov), .out_ready(core_or), .out_data(core_out), .busy(core_busy));

    // ── FSM chinh ──
    localparam S_RX=3'd0, S_FEED=3'd1, S_RUN=3'd2, S_PUSH=3'd3;
    reg [2:0] st;
    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st<=S_RX; rcnt<=0; tcnt<=0; core_iv<=0; core_or<=0; rxf_rd<=0; txf_wr<=0; txf_din<=0; idle_cnt<=0;
        end else begin
            core_or<=1'b0; rxf_rd<=1'b0; txf_wr<=1'b0;
            case (st)
                S_RX: begin                          // drain RX FIFO vao in_buf (co RESYNC khung)
                    core_iv<=1'b0;
                    if (rxf_empty) begin
                        // RX ranh: dem khoang nghi; du lau (giua 2 khung) -> can lai dau khung
                        if (idle_cnt < RESYNC_GAP) idle_cnt <= idle_cnt + 1'b1;
                        else rcnt <= 0;
                    end else begin
                        idle_cnt <= 0;
                        if (!rxf_rd) begin
                            in_buf[rcnt] <= rxf_dout;
                            rxf_rd <= 1'b1;
                            if (rcnt==IN_B-1) begin rcnt<=0; st<=S_FEED; end
                            else rcnt<=rcnt+1'b1;
                        end
                    end
                end
                S_FEED: begin                        // xung in_valid
                    core_iv<=1'b1;
                    if (core_iv && core_ir) begin core_iv<=1'b0; st<=S_RUN; end
                end
                S_RUN: if (core_ov) begin            // chot output -> consume
                    for (i=0;i<OUT_B;i=i+1) out_buf[i] <= core_out[i*8 +: 8];
                    core_or<=1'b1; tcnt<=0; st<=S_PUSH;
                end
                S_PUSH: if (!txf_full) begin          // do 57B vao TX FIFO roi quay lai nhan
                    txf_din<=out_buf[tcnt]; txf_wr<=1'b1;
                    if (tcnt==OUT_B-1) st<=S_RX; else tcnt<=tcnt+1'b1;
                end
                default: st<=S_RX;
            endcase
        end
    end
endmodule
