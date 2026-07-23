`timescale 1ns / 1ps
//============================================================================
// direct_lut.v — Bảng tra activation (SiLU), ĐỌC TUẦN TỰ qua BRAM 1 phần tử/chu kỳ.
//
// Trước: N_ELEM đọc tổ hợp SONG SONG -> với LOAD_MODE=1 (RAM runtime) Vivado trải
// bảng thành ~4096 FF + N_ELEM cây mux 512-lối = ~44k LUT/cái (xem F7/F8 Mux).
// Giờ: 1 BRAM (simple dual-port), đọc 1 phần tử/chu kỳ trong ~N_ELEM chu kỳ
// -> ~1 BRAM18 + vài chục LUT. BIT-EXACT (cùng bảng tra, chỉ đổi cách đọc).
//
// Handshake: pulse `start` (chốt x_in). Sau ~N_ELEM+2 chu kỳ -> `done` 1 chu kỳ;
// y_out giữ kết quả tới lần `start` kế. Cần clk/rst_n (trước là tổ hợp).
//
//   LOAD_MODE=0: $readmemh init BRAM (verified).
//   LOAD_MODE=1: nạp runtime qua write-bus (1 byte/entry), w_sel==MY_ID mới nghe.
//============================================================================
module direct_lut #(
    parameter N_ELEM      = 40,
    parameter SHARED      = 0,        // 1 = 512 entry (block0|block1)
    parameter LUT_FILE    = "none",
    parameter LUT_FILE_B1 = "none",
    parameter LOAD_MODE   = 0,
    parameter [4:0] MY_ID = 5'd0
)(
    input  wire                clk,
    input  wire                rst_n,
    input  wire                block_sel,
    input  wire                start,        // pulse: chốt x_in, bắt đầu tra
    input  wire [N_ELEM*8-1:0] x_in,
    output wire [N_ELEM*8-1:0] y_out,
    output reg                 done,
    // ── write-bus dùng chung (chỉ tác dụng khi LOAD_MODE=1) ──
    input  wire                w_en,
    input  wire                w_seg_rst,
    input  wire [4:0]          w_sel,
    input  wire                w_half,
    input  wire [7:0]          w_byte
);
    localparam AW    = SHARED ? 9 : 8;
    localparam DEPTH = SHARED ? 512 : 256;

    (* ram_style = "block" *)
    reg [7:0] lut [0:DEPTH-1];

    generate if (LOAD_MODE == 0) begin : gen_init
        initial begin
            if (SHARED) begin
                if (LUT_FILE    != "none") $readmemh(LUT_FILE,    lut, 0,   255);
                if (LUT_FILE_B1 != "none") $readmemh(LUT_FILE_B1, lut, 256, 511);
            end else if (LUT_FILE != "none") $readmemh(LUT_FILE, lut);
        end
    end endgenerate

    // ── cổng GHI (runtime load) ──
    wire sel_me = (w_sel == MY_ID);
    reg [AW-1:0] waddr;
    always @(posedge clk) begin
        if (LOAD_MODE && sel_me) begin
            if (w_seg_rst)   waddr <= (SHARED && w_half) ? 9'd256 : {AW{1'b0}};
            else if (w_en) begin lut[waddr] <= w_byte; waddr <= waddr + 1'b1; end
        end
    end

    // ── ĐỌC tuần tự (shift x ra, shift y vào -> không cần mux biến) ──
    reg [N_ELEM*8-1:0] x_sr;        // dịch 8/chu kỳ, luôn đọc x_sr[7:0]
    reg [N_ELEM*8-1:0] y_sr;        // dịch kết quả vào từ MSB
    reg [7:0]          rd;
    reg [6:0]          acnt, dcnt;  // đếm phát địa chỉ / nhận dữ liệu
    reg                state, rpend;

    localparam S_IDLE = 1'b0, S_READ = 1'b1;

    wire [7:0]    ub     = x_sr[7:0] + 8'd128;            // signed -> [0,255]
    wire [AW-1:0] a_next = SHARED ? {block_sel, ub} : ub[AW-1:0];

    always @(posedge clk) rd <= lut[a_next];             // cổng ĐỌC đồng bộ (BRAM)

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; done <= 0; acnt <= 0; dcnt <= 0; rpend <= 0;
            x_sr <= 0; y_sr <= 0;
        end else begin
            done  <= 1'b0;
            rpend <= 1'b0;
            case (state)
                S_IDLE: if (start) begin
                            x_sr  <= x_in;
                            acnt  <= 0; dcnt <= 0;
                            state <= S_READ;
                        end
                S_READ: begin
                    if (acnt < N_ELEM) begin            // phát địa chỉ phần tử acnt
                        x_sr  <= x_sr >> 8;
                        acnt  <= acnt + 1'b1;
                        rpend <= 1'b1;                   // rd hợp lệ chu kỳ sau
                    end
                    if (rpend) begin                    // nhận dữ liệu phần tử trước
                        y_sr <= {rd, y_sr[N_ELEM*8-1:8]};
                        dcnt <= dcnt + 1'b1;
                        if (dcnt + 1'b1 == N_ELEM) begin
                            done  <= 1'b1;
                            state <= S_IDLE;
                        end
                    end
                end
            endcase
        end
    end

    assign y_out = y_sr;
endmodule
