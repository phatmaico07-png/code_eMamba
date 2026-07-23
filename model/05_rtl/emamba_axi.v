`timescale 1ns / 1ps
//============================================================================
// emamba_axi.v — AXI4 SLAVE *** SINGLE-PORT STREAMING *** (KV26). DATA-ONLY.
//
//   KV26 SmartConnect chi decode tin cay 1 o dia chi -> dung SINGLE-PORT:
//   moi giao tiep qua DUY NHAT base 0x100. Ghi: lenh {cmd<<24|count} khi idle,
//   roi data. Doc: status hoac output (theo read_sel noi bo).
//   weight + shift NUNG SAN (LOAD_MODE=0) -> KHONG nap gi.
//
//   GHI (32-bit/word vao base 0x100):
//     IDLE: word = LENH {cmd[31:24], count[23:0]}
//       0x02 INPUT   : theo sau 80 word -> in_buf
//       0x05 START   : chay 1 frame; read_sel=STATUS
//       0x06 READOUT : read_sel=OUTPUT, rptr=0
//       0x07 RDSTAT  : read_sel=STATUS
//     dang INPUT: word=data -> in_buf[wptr], dem lui
//   DOC (base): STATUS {.., ld_done=1, busy, done} | OUTPUT out_cap[rptr] (rptr tu tang)
//
//   Loi: emamba_top_synth (LOAD_MODE=0 nung san, lean core).
//============================================================================
module emamba_axi #(
    parameter integer DW = 32,
    parameter integer AW = 16
)(
    input  wire           S_AXI_ACLK,
    input  wire           S_AXI_ARESETN,
    input  wire [AW-1:0]  S_AXI_AWADDR,
    input  wire [7:0]     S_AXI_AWLEN,
    input  wire [2:0]     S_AXI_AWSIZE,
    input  wire [1:0]     S_AXI_AWBURST,
    input  wire           S_AXI_AWVALID,
    output wire           S_AXI_AWREADY,
    input  wire [DW-1:0]  S_AXI_WDATA,
    input  wire [DW/8-1:0] S_AXI_WSTRB,
    input  wire           S_AXI_WLAST,
    input  wire           S_AXI_WVALID,
    output wire           S_AXI_WREADY,
    output wire [1:0]     S_AXI_BRESP,
    output wire           S_AXI_BVALID,
    input  wire           S_AXI_BREADY,
    input  wire [AW-1:0]  S_AXI_ARADDR,
    input  wire [7:0]     S_AXI_ARLEN,
    input  wire [2:0]     S_AXI_ARSIZE,
    input  wire [1:0]     S_AXI_ARBURST,
    input  wire           S_AXI_ARVALID,
    output wire           S_AXI_ARREADY,
    output wire [DW-1:0]  S_AXI_RDATA,
    output wire [1:0]     S_AXI_RRESP,
    output wire           S_AXI_RLAST,
    output wire           S_AXI_RVALID,
    input  wire           S_AXI_RREADY
);
    localparam IN_WORDS = 80, OUT_WORDS = 15;
    localparam [7:0] C_INPUT=8'h02, C_START=8'h05, C_READOUT=8'h06, C_RDSTAT=8'h07;
    localparam       M_IDLE=1'b0,  M_INPUT=1'b1;
    localparam       R_STAT=1'b0,  R_OUT=1'b1;

    wire clk = S_AXI_ACLK, rst_n = S_AXI_ARESETN;

    reg  [31:0]  in_buf [0:IN_WORDS-1];
    reg  [479:0] out_cap;
    reg          start_pulse, done_r, busy_r;
    reg          wmode;
    reg  [23:0]  wrem;
    reg  [6:0]   wptr;
    reg  [4:0]   rptr;
    reg          read_sel;
    reg          rptr_clr;

    wire [2559:0] core_in_data;
    genvar gi;
    generate for (gi = 0; gi < IN_WORDS; gi = gi + 1) begin : g_in
        assign core_in_data[gi*32 +: 32] = in_buf[gi];
    end endgenerate

    // ══════════ AXI WRITE — single-port stream (AWADDR bo qua) ══════════
    reg          aw_hs, bvalid_r;
    wire         w_ok = aw_hs && S_AXI_WVALID;

    assign S_AXI_AWREADY = !aw_hs && !bvalid_r;
    assign S_AXI_WREADY  = aw_hs;
    assign S_AXI_BRESP   = 2'b00;
    assign S_AXI_BVALID  = bvalid_r;

    integer wi;
    always @(posedge clk) begin
        if (!rst_n) begin
            aw_hs <= 0; bvalid_r <= 0;
            start_pulse <= 0; rptr_clr <= 0; read_sel <= R_STAT;
            wmode <= M_IDLE; wrem <= 0; wptr <= 0;
            for (wi = 0; wi < IN_WORDS; wi = wi + 1) in_buf[wi] <= 32'd0;
        end else begin
            start_pulse <= 0; rptr_clr <= 0;
            if (!aw_hs && !bvalid_r) begin
                if (S_AXI_AWVALID) aw_hs <= 1;
            end else if (w_ok) begin
                if (wmode == M_IDLE) begin
                    case (S_AXI_WDATA[31:24])
                        C_INPUT:  begin wmode<=M_INPUT; wrem<=S_AXI_WDATA[23:0]; wptr<=0; end
                        C_START:  begin start_pulse<=1; read_sel<=R_STAT; rptr_clr<=1'b1; end
                        C_READOUT:begin read_sel<=R_OUT;  rptr_clr<=1'b1; end
                        C_RDSTAT: read_sel<=R_STAT;
                        default: ;
                    endcase
                end else begin   // M_INPUT
                    in_buf[wptr] <= S_AXI_WDATA; wptr <= wptr + 1'b1;
                    wrem <= wrem - 1'b1;
                    if (wrem == 24'd1) wmode <= M_IDLE;
                end
                if (S_AXI_WLAST) begin aw_hs <= 0; bvalid_r <= 1; end
            end
            if (bvalid_r && S_AXI_BREADY) bvalid_r <= 0;
        end
    end

    // ══════════ AXI READ — status / output stream (ARADDR bo qua) ══════════
    reg          ar_hs, rvalid_r;
    reg  [31:0]  rdata_r;
    // bit0=done, bit1=busy, bit2=ld_done(=1 vi weight nung san)
    wire [31:0]  status = {21'd0, wptr, 1'b1, busy_r, done_r};

    assign S_AXI_ARREADY = !ar_hs;
    assign S_AXI_RVALID  = rvalid_r;
    assign S_AXI_RDATA   = rdata_r;
    assign S_AXI_RRESP   = 2'b00;
    assign S_AXI_RLAST   = rvalid_r;

    always @(*) begin
        case (read_sel)
            R_OUT:   rdata_r = out_cap[rptr*32 +: 32];
            default: rdata_r = status;
        endcase
    end

    always @(posedge clk) begin
        if (!rst_n) begin ar_hs <= 0; rvalid_r <= 0; rptr <= 0; end
        else begin
            if (rptr_clr) rptr <= 0;
            if (!ar_hs) begin
                if (S_AXI_ARVALID) begin ar_hs <= 1; rvalid_r <= 1; end
            end else if (rvalid_r && S_AXI_RREADY) begin
                rvalid_r <= 0; ar_hs <= 0;
                if (!rptr_clr && read_sel == R_OUT && rptr < OUT_WORDS-1) rptr <= rptr + 1'b1;
            end
        end
    end

    // ══════════ inference FSM ══════════
    localparam S_IDLE=2'd0, S_FEED=2'd1, S_WAIT=2'd2;
    reg [1:0] st; reg core_iv, core_or;
    wire core_ir, core_ov, core_busy; wire [455:0] core_out;
    always @(posedge clk) begin
        if (!rst_n) begin st<=S_IDLE; core_iv<=0; core_or<=0; done_r<=0; busy_r<=0; out_cap<=0; end
        else case (st)
            S_IDLE: if (start_pulse) begin core_iv<=1; busy_r<=1; done_r<=0; st<=S_FEED; end
            S_FEED: if (core_ir)     begin core_iv<=0; core_or<=1; st<=S_WAIT; end
            S_WAIT: if (core_ov)     begin out_cap<={24'd0,core_out}; core_or<=0; busy_r<=0; done_r<=1; st<=S_IDLE; end
        endcase
    end

    // Loi NUNG SAN (LOAD_MODE=0): khong wbus, khong shift port.
    emamba_top_synth u_core (
        .clk(clk), .rst_n(rst_n),
        .in_valid(core_iv), .in_ready(core_ir), .in_data(core_in_data),
        .out_valid(core_ov), .out_ready(core_or), .out_data(core_out),
        .busy(core_busy)
    );

endmodule
