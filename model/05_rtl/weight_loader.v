`timescale 1ns / 1ps
//============================================================================
// weight_loader.v — Nạp weight runtime: word blob (PS ghi) -> wbus byte-serial.
//
// v2 (board-fix):
//   1) Load-map ROM HARDCODE bằng case (bỏ $readmemh đường dẫn tuyệt đối —
//      loại rủi ro "sim ăn, silicon đói").
//   2) wr_ready KHÔNG BAO GIỜ kẹt bus: ở S_IDLE/S_DONE nhận-và-bỏ word thừa
//      (trước đây WREADY=0 vĩnh viễn -> 1 write lạc là treo cả AXI/board).
//   3) Cổng dbg: {state[1:0], seg[5:0], bcnt[15:0]} đọc qua AXI để chẩn đoán.
//
// Giao thức wbus GIỮ NGUYÊN BIT-EXACT: mỗi segment phát 1 seg_rst rồi stream
// nbytes byte; word 32-bit = 4 byte little-endian (byte thấp trước).
//============================================================================
module weight_loader #(
    parameter NSEG = 51
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        load_start,   // pulse 1 cycle: bắt đầu nạp (về segment 0)
    input  wire        wr_valid,     // PS ghi 1 word blob
    input  wire [31:0] wr_data,
    output wire        wr_ready,     // loader sẵn sàng nhận word kế
    output reg  [15:0] wbus,         // {byte[15:8],half[7],sel[6:2],seg_rst[1],en[0]}
    output reg         load_done,
    output wire [23:0] dbg           // {state,seg,bcnt} — đọc ở 0x008
);
    localparam S_IDLE = 2'd0, S_SEGRST = 2'd1, S_BYTE = 2'd2, S_DONE = 2'd3;
    reg [1:0]   state;
    reg [5:0]   seg;
    reg [15:0]  bcnt;     // byte đã phát trong segment
    reg [31:0]  wbuf;     // word đang xài
    reg [1:0]   widx;     // byte trong word (0..3)
    reg         wfull;    // wbuf còn byte chưa xài

    // ── load-map ROM (hardcode, sinh từ lm_sel/half/nbytes.hex — tổng 14771 byte) ──
    reg [4:0]  cur_sel;
    reg        cur_half;
    reg [15:0] cur_nb;
    always @* begin
        cur_sel = 5'd0; cur_half = 1'b0; cur_nb = 16'd0;
        case (seg)
            6'd0 : begin cur_sel=5'h00; cur_half=1'b0; cur_nb=16'd400 ; end
            6'd1 : begin cur_sel=5'h01; cur_half=1'b0; cur_nb=16'd40  ; end
            6'd2 : begin cur_sel=5'h02; cur_half=1'b0; cur_nb=16'd20  ; end
            6'd3 : begin cur_sel=5'h03; cur_half=1'b0; cur_nb=16'd40  ; end
            6'd4 : begin cur_sel=5'h04; cur_half=1'b0; cur_nb=16'd800 ; end
            6'd5 : begin cur_sel=5'h05; cur_half=1'b0; cur_nb=16'd80  ; end
            6'd6 : begin cur_sel=5'h06; cur_half=1'b0; cur_nb=16'd800 ; end
            6'd7 : begin cur_sel=5'h07; cur_half=1'b0; cur_nb=16'd80  ; end
            6'd8 : begin cur_sel=5'h08; cur_half=1'b0; cur_nb=16'd160 ; end
            6'd9 : begin cur_sel=5'h09; cur_half=1'b0; cur_nb=16'd80  ; end
            6'd10: begin cur_sel=5'h0a; cur_half=1'b0; cur_nb=16'd120 ; end
            6'd11: begin cur_sel=5'h0b; cur_half=1'b0; cur_nb=16'd80  ; end
            6'd12: begin cur_sel=5'h0c; cur_half=1'b0; cur_nb=16'd120 ; end
            6'd13: begin cur_sel=5'h0d; cur_half=1'b0; cur_nb=16'd6   ; end
            6'd14: begin cur_sel=5'h0e; cur_half=1'b0; cur_nb=16'd320 ; end
            6'd15: begin cur_sel=5'h0f; cur_half=1'b0; cur_nb=16'd320 ; end
            6'd16: begin cur_sel=5'h10; cur_half=1'b0; cur_nb=16'd800 ; end
            6'd17: begin cur_sel=5'h11; cur_half=1'b0; cur_nb=16'd40  ; end
            6'd18: begin cur_sel=5'h12; cur_half=1'b0; cur_nb=16'd320 ; end
            6'd19: begin cur_sel=5'h13; cur_half=1'b0; cur_nb=16'd80  ; end
            6'd20: begin cur_sel=5'h14; cur_half=1'b0; cur_nb=16'd256 ; end
            6'd21: begin cur_sel=5'h15; cur_half=1'b0; cur_nb=16'd256 ; end
            6'd22: begin cur_sel=5'h02; cur_half=1'b1; cur_nb=16'd20  ; end
            6'd23: begin cur_sel=5'h03; cur_half=1'b1; cur_nb=16'd40  ; end
            6'd24: begin cur_sel=5'h04; cur_half=1'b1; cur_nb=16'd800 ; end
            6'd25: begin cur_sel=5'h05; cur_half=1'b1; cur_nb=16'd80  ; end
            6'd26: begin cur_sel=5'h06; cur_half=1'b1; cur_nb=16'd800 ; end
            6'd27: begin cur_sel=5'h07; cur_half=1'b1; cur_nb=16'd80  ; end
            6'd28: begin cur_sel=5'h08; cur_half=1'b1; cur_nb=16'd160 ; end
            6'd29: begin cur_sel=5'h09; cur_half=1'b1; cur_nb=16'd80  ; end
            6'd30: begin cur_sel=5'h0a; cur_half=1'b1; cur_nb=16'd120 ; end
            6'd31: begin cur_sel=5'h0b; cur_half=1'b1; cur_nb=16'd80  ; end
            6'd32: begin cur_sel=5'h0c; cur_half=1'b1; cur_nb=16'd120 ; end
            6'd33: begin cur_sel=5'h0d; cur_half=1'b1; cur_nb=16'd6   ; end
            6'd34: begin cur_sel=5'h0e; cur_half=1'b1; cur_nb=16'd320 ; end
            6'd35: begin cur_sel=5'h0f; cur_half=1'b1; cur_nb=16'd320 ; end
            6'd36: begin cur_sel=5'h10; cur_half=1'b1; cur_nb=16'd800 ; end
            6'd37: begin cur_sel=5'h11; cur_half=1'b1; cur_nb=16'd40  ; end
            6'd38: begin cur_sel=5'h12; cur_half=1'b1; cur_nb=16'd320 ; end
            6'd39: begin cur_sel=5'h13; cur_half=1'b1; cur_nb=16'd80  ; end
            6'd40: begin cur_sel=5'h14; cur_half=1'b1; cur_nb=16'd256 ; end
            6'd41: begin cur_sel=5'h15; cur_half=1'b1; cur_nb=16'd256 ; end
            6'd42: begin cur_sel=5'h16; cur_half=1'b0; cur_nb=16'd25  ; end
            6'd43: begin cur_sel=5'h17; cur_half=1'b0; cur_nb=16'd48  ; end
            6'd44: begin cur_sel=5'h18; cur_half=1'b0; cur_nb=16'd48  ; end
            6'd45: begin cur_sel=5'h19; cur_half=1'b0; cur_nb=16'd1600; end
            6'd46: begin cur_sel=5'h1a; cur_half=1'b0; cur_nb=16'd160 ; end
            6'd47: begin cur_sel=5'h1b; cur_half=1'b0; cur_nb=16'd1600; end
            6'd48: begin cur_sel=5'h1c; cur_half=1'b0; cur_nb=16'd40  ; end
            6'd49: begin cur_sel=5'h1d; cur_half=1'b0; cur_nb=16'd1140; end
            6'd50: begin cur_sel=5'h1e; cur_half=1'b0; cur_nb=16'd114 ; end
            default: ;
        endcase
    end

    wire [7:0] cur_byte = wbuf[widx*8 +: 8];

    // KHÔNG BAO GIỜ chặn bus: chỉ backpressure khi đang tiêu word (S_BYTE,wfull)
    // hoặc 1 cycle seg_rst; ở IDLE/DONE nhận-và-bỏ.
    assign wr_ready = (state == S_BYTE) ? !wfull : (state != S_SEGRST);

    assign dbg = {state, seg, bcnt};

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; seg <= 0; bcnt <= 0; widx <= 0; wfull <= 1'b0;
            wbuf <= 32'd0; wbus <= 16'd0; load_done <= 1'b0;
        end else begin
            wbus <= 16'd0;                       // mặc định idle (en=0, seg_rst=0)
            case (state)
                S_IDLE: begin                    // word tới khi chưa start: bỏ qua
                    if (load_start) begin
                        seg <= 0; bcnt <= 0; widx <= 0; wfull <= 1'b0; load_done <= 1'b0;
                        state <= S_SEGRST;
                    end
                end
                S_SEGRST: begin
                    wbus  <= {8'd0, cur_half, cur_sel, 1'b1, 1'b0};   // seg_rst
                    state <= S_BYTE;
                end
                S_BYTE: begin
                    if (!wfull) begin
                        if (wr_valid) begin wbuf <= wr_data; widx <= 0; wfull <= 1'b1; end
                        // else: chờ word (wr_ready=1)
                    end else begin
                        wbus <= {cur_byte, cur_half, cur_sel, 1'b0, 1'b1};   // en + byte
                        bcnt <= bcnt + 1'b1;
                        widx <= widx + 1'b1;
                        if (widx == 2'd3) wfull <= 1'b0;                      // hết word
                        if (bcnt + 1'b1 == cur_nb) begin                     // hết segment
                            bcnt <= 0;
                            if (seg + 1'b1 == NSEG) begin load_done <= 1'b1; state <= S_DONE; end
                            else                        begin seg <= seg + 1'b1; state <= S_SEGRST; end
                            // word còn byte (widx<3) -> sau S_SEGRST quay lại S_BYTE xài tiếp
                        end
                    end
                end
                S_DONE: begin                    // word thừa: nhận-và-bỏ, không kẹt bus
                    load_done <= 1'b1;
                    if (load_start) begin
                        seg <= 0; bcnt <= 0; widx <= 0; wfull <= 1'b0; load_done <= 1'b0;
                        state <= S_SEGRST;
                    end
                end
            endcase
        end
    end
endmodule
