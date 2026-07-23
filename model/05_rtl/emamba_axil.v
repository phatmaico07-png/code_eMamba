`timescale 1ns / 1ps
//============================================================================
// emamba_axil.v — AXI4-Lite wrapper cho eMamba accelerator (tích hợp SoC KV26)
//
// Bọc emamba_top_synth (lõi đã verify bit-exact) bằng 1 cổng AXI4-Lite slave.
// PS (ARM) điều khiển qua thanh ghi memory-mapped → KHÔNG cần DMA.
//
// Cách dùng (phần mềm PS):
//   1) Ghi 320 byte featuremap INT8 vào IN_BUF  (80 word 32-bit, offset 0x040)
//   2) Ghi CTRL = 1 (offset 0x000)  → start
//   3) Đợi STATUS bit0 (done) = 1   (poll offset 0x004)   [~2651 cycle ≈ 9µs @300MHz]
//   4) Đọc 57 byte pose INT8 từ OUT_BUF (15 word, offset 0x200)
//
// Bản đồ thanh ghi (byte offset, AXI-Lite 32-bit):
//   0x000  CTRL    W   bit0 = start (xung tự xoá)
//   0x004  STATUS  R   bit0 = done, bit1 = busy
//   0x040..0x17F   IN_BUF   RW  80 word = 320 byte input  (word i = in_data[i*32+:32])
//   0x200..0x23B   OUT_BUF  R   15 word = 57 byte output   (word i = out_data[i*32+:32])
//
// Top chỉ có cổng AXI-Lite (~100 bit) → implement được trên KV26 (không nổ 3023 I/O).
//============================================================================

module emamba_axil #(
    parameter integer C_S_AXI_DATA_WIDTH = 32,
    parameter integer C_S_AXI_ADDR_WIDTH = 12
)(
    // ── AXI4-Lite slave (nối tới PS qua AXI interconnect) ──
    input  wire                              S_AXI_ACLK,
    input  wire                              S_AXI_ARESETN,   // active-low
    input  wire [C_S_AXI_ADDR_WIDTH-1:0]     S_AXI_AWADDR,
    input  wire [2:0]                        S_AXI_AWPROT,
    input  wire                              S_AXI_AWVALID,
    output wire                              S_AXI_AWREADY,
    input  wire [C_S_AXI_DATA_WIDTH-1:0]     S_AXI_WDATA,
    input  wire [C_S_AXI_DATA_WIDTH/8-1:0]   S_AXI_WSTRB,
    input  wire                              S_AXI_WVALID,
    output wire                              S_AXI_WREADY,
    output wire [1:0]                        S_AXI_BRESP,
    output wire                              S_AXI_BVALID,
    input  wire                              S_AXI_BREADY,
    input  wire [C_S_AXI_ADDR_WIDTH-1:0]     S_AXI_ARADDR,
    input  wire [2:0]                        S_AXI_ARPROT,
    input  wire                              S_AXI_ARVALID,
    output wire                              S_AXI_ARREADY,
    output wire [C_S_AXI_DATA_WIDTH-1:0]     S_AXI_RDATA,
    output wire [1:0]                        S_AXI_RRESP,
    output wire                              S_AXI_RVALID,
    input  wire                              S_AXI_RREADY
);
    localparam integer ADDR_LSB = 2;                 // 32-bit word → bỏ 2 bit thấp
    localparam IN_WORDS  = 80;                        // 320 byte input
    localparam OUT_WORDS = 15;                        // 57 byte output (làm tròn lên)

    wire clk   = S_AXI_ACLK;
    wire rst_n = S_AXI_ARESETN;

    // ══════════ register file ══════════
    reg  [31:0] in_buf  [0:IN_WORDS-1];               // 80 word input
    reg  [479:0] out_cap;                             // 15 word output (456 bit + pad)
    reg         start_pulse;
    reg         done_r, busy_r;

    // ── ráp in_data 2560-bit từ in_buf ──
    wire [2559:0] core_in_data;
    genvar gi;
    generate
        for (gi = 0; gi < IN_WORDS; gi = gi + 1) begin : g_in
            assign core_in_data[gi*32 +: 32] = in_buf[gi];
        end
    endgenerate

    // ══════════ AXI4-Lite WRITE channel ══════════
    reg axi_awready, axi_wready, axi_bvalid;
    reg [C_S_AXI_ADDR_WIDTH-1:0] axi_awaddr;
    assign S_AXI_AWREADY = axi_awready;
    assign S_AXI_WREADY  = axi_wready;
    assign S_AXI_BRESP   = 2'b00;
    assign S_AXI_BVALID  = axi_bvalid;

    wire wr_en = axi_awready & S_AXI_AWVALID & axi_wready & S_AXI_WVALID;

    integer wi;
    always @(posedge clk) begin
        if (!rst_n) begin
            axi_awready <= 0; axi_wready <= 0; axi_bvalid <= 0; axi_awaddr <= 0;
            start_pulse <= 0;
            for (wi = 0; wi < IN_WORDS; wi = wi + 1) in_buf[wi] <= 32'd0;
        end else begin
            start_pulse <= 1'b0;                      // xung 1 cycle

            // address-ready (latch awaddr khi cả AW & W hợp lệ)
            if (!axi_awready && S_AXI_AWVALID && S_AXI_WVALID) begin
                axi_awready <= 1; axi_awaddr <= S_AXI_AWADDR;
            end else axi_awready <= 0;
            // data-ready
            if (!axi_wready && S_AXI_AWVALID && S_AXI_WVALID) axi_wready <= 1;
            else axi_wready <= 0;

            // ghi thanh ghi
            if (wr_en) begin
                if (axi_awaddr == 12'h000)
                    start_pulse <= S_AXI_WDATA[0];                 // CTRL 0x000 → start
                else if (axi_awaddr >= 12'h040 && axi_awaddr < (12'h040 + IN_WORDS*4))
                    in_buf[(axi_awaddr - 12'h040) >> ADDR_LSB] <= S_AXI_WDATA;  // IN_BUF
            end

            // write response
            if (wr_en) axi_bvalid <= 1;
            else if (S_AXI_BREADY && axi_bvalid) axi_bvalid <= 0;
        end
    end

    // ══════════ AXI4-Lite READ channel ══════════
    reg axi_arready, axi_rvalid;
    reg [C_S_AXI_ADDR_WIDTH-1:0] axi_araddr;
    reg [31:0] axi_rdata;
    assign S_AXI_ARREADY = axi_arready;
    assign S_AXI_RRESP   = 2'b00;
    assign S_AXI_RVALID  = axi_rvalid;
    assign S_AXI_RDATA   = axi_rdata;

    always @(posedge clk) begin
        if (!rst_n) begin
            axi_arready <= 0; axi_rvalid <= 0; axi_araddr <= 0; axi_rdata <= 0;
        end else begin
            if (!axi_arready && S_AXI_ARVALID) begin
                axi_arready <= 1; axi_araddr <= S_AXI_ARADDR;
            end else axi_arready <= 0;

            if (axi_arready && S_AXI_ARVALID && !axi_rvalid) begin
                axi_rvalid <= 1;
                // decode đọc
                if (axi_araddr == 12'h004)
                    axi_rdata <= {30'd0, busy_r, done_r};          // STATUS
                else if (axi_araddr >= 12'h040 && axi_araddr < 12'h040 + IN_WORDS*4)
                    axi_rdata <= in_buf[(axi_araddr - 12'h040) >> ADDR_LSB];  // IN readback
                else if (axi_araddr >= 12'h200 && axi_araddr < 12'h200 + OUT_WORDS*4)
                    axi_rdata <= out_cap[((axi_araddr - 12'h200) >> ADDR_LSB)*32 +: 32]; // OUT
                else
                    axi_rdata <= 32'd0;
            end else if (axi_rvalid && S_AXI_RREADY) begin
                axi_rvalid <= 0;
            end
        end
    end

    // ══════════ FSM điều khiển lõi ══════════
    localparam S_IDLE=2'd0, S_FEED=2'd1, S_WAIT=2'd2;
    reg [1:0] st;
    reg core_iv, core_or;
    wire core_ir, core_ov, core_busy;
    wire [455:0] core_out;

    always @(posedge clk) begin
        if (!rst_n) begin
            st <= S_IDLE; core_iv <= 0; core_or <= 0;
            done_r <= 0; busy_r <= 0; out_cap <= 0;
        end else begin
            case (st)
                S_IDLE: if (start_pulse) begin
                            core_iv <= 1; busy_r <= 1; done_r <= 0; st <= S_FEED;
                        end
                S_FEED: if (core_ir) begin               // lõi nhận frame
                            core_iv <= 0; core_or <= 1; st <= S_WAIT;
                        end
                S_WAIT: if (core_ov) begin               // lõi xuất pose
                            out_cap <= {24'd0, core_out};
                            core_or <= 0; busy_r <= 0; done_r <= 1; st <= S_IDLE;
                        end
            endcase
        end
    end

    // ══════════ lõi accelerator đã verify ══════════
    emamba_top_synth u_core (
        .clk(clk), .rst_n(rst_n),
        .in_valid(core_iv), .in_ready(core_ir), .in_data(core_in_data),
        .out_valid(core_ov), .out_ready(core_or), .out_data(core_out),
        .busy(core_busy)
    );

endmodule
