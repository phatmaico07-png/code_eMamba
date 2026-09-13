`timescale 1ns / 1ps
//===========================================================================
// range_norm_recip.v — RangeNorm reciprocal-multiply (FSMD), HARDCODE cau hinh eMamba-MARS
//
//   Co dinh: DIM=20, SHARED, N_CU=20, IDs. gamma_shift QUA PORT (board nap runtime).
//   N_CU=20 (song song), GAMMA_ID=2/BETA_ID=3, K_RECIP=23.
//   CHI giu cau hinh per-block: block_sel + nap gamma/beta (file sim hoac wbus board).
//
//   y[i] = clamp8( gshift( ((g[i]*q[i]+2^14)>>>15)+b[i], gamma_shift_in ) )   (shift configurable)
//   q[i] = sat16( sign(c)*(|c|*recip >> 8) ),  c = x[i]-mean
//   recip = ceil(2^23/range) [LUT/ROM 256x24],  range = max(x)-min(x) in [1,255]
//   mean  = (Sx*6554 + 2^16) >>> 17
//   FSMD 5 state: IDLE->STAT->MUL->POST->DONE, 4 cycle/token. Bit-exact divider.
//===========================================================================
module range_norm_recip #(
    parameter GAMMA_FILE    = "none",    // sim: init gamma block0 (INT8 hex)
    parameter BETA_FILE     = "none",    // sim: init beta  block0 (INT16 hex)
    parameter GAMMA_FILE_B1 = "none",    // sim: init gamma block1
    parameter BETA_FILE_B1  = "none",    // sim: init beta  block1
    parameter LOAD_MODE     = 0,         // 0 = readmemh (sim) ; 1 = nap runtime qua wbus (board)
    parameter RECIP_FILE    = "D:/DATN/eMamba_Accelerator/rtl/recip_lut.hex"
)(
    input  wire             clk,
    input  wire             rst_n,
    input  wire             block_sel,           // 0=block0, 1=block1 (chon gamma/beta)
    input  wire signed [3:0] gamma_shift_in,     // shift requant RUNTIME (board nap); <0 trai, >0 phai
    input  wire             in_valid,
    output wire             in_ready,
    input  wire [159:0]     in_data,             // 20 x INT8
    output wire             out_valid,
    input  wire             out_ready,
    output wire [159:0]     out_data,            // 20 x INT8
    input  wire [15:0]      wbus
);
    // ===================== HANG SO (HARDCODE) =====================
    localparam DIM      = 20;
    localparam K_RECIP  = 23;
    localparam RSH      = 8;              // q = (|c|*recip) >> 8
    localparam RW       = 24;            // recip 24-bit
    localparam DEPTH    = 2*DIM;          // SHARED: double-depth gamma/beta
    localparam CW       = 6;             // clog2(40)
    localparam [4:0] GAMMA_ID = 5'd2;
    localparam [4:0] BETA_ID  = 5'd3;
    wire signed [3:0] gs = gamma_shift_in;   // shift configurable (khong hardcode)

    // ===================== CONTROLLER (FSM) =====================
    // P-Fmax: tach S_STAT -> S_STAT (max/min/mean) + S_RECIP (rng + tra recip_rom)
    //   -> cat path dai x->recip_reg lam doi -> Fmax cao hon. +1 cyc/token (bit-exact).
    localparam S_IDLE=3'd0, S_STAT=3'd1, S_RECIP=3'd5, S_MUL=3'd2, S_POST=3'd3, S_DONE=3'd4;
    reg [2:0] state, nstate;
    always @(posedge clk or negedge rst_n)
        if (!rst_n) state <= S_IDLE; else state <= nstate;
    always @(*) begin
        nstate = state;
        case (state)
            S_IDLE: if (in_valid)  nstate = S_STAT;
            S_STAT:                nstate = S_RECIP;
            S_RECIP:               nstate = S_MUL;
            S_MUL :                nstate = S_POST;
            S_POST:                nstate = S_DONE;
            S_DONE: if (out_ready) nstate = S_IDLE;
            default:               nstate = S_IDLE;
        endcase
    end
    assign in_ready  = (state == S_IDLE);
    assign out_valid = (state == S_DONE);

    // ===================== RECIP ROM 256x24 -> LUTRAM distributed (0 BRAM, ~96 LUT) =====================
    (* rom_style = "distributed" *) reg [RW-1:0] recip_rom [0:255];
    initial $readmemh(RECIP_FILE, recip_rom);

    // ===================== GAMMA / BETA (double-depth, block_sel) =====================
    (* ram_style = "distributed" *) reg signed [7:0]  gamma_buf [0:DEPTH-1];
    (* ram_style = "distributed" *) reg signed [15:0] beta_buf  [0:DEPTH-1];
    initial begin
        if (LOAD_MODE==0 && GAMMA_FILE   !="none") $readmemh(GAMMA_FILE,    gamma_buf, 0,   DIM-1);
        if (LOAD_MODE==0 && GAMMA_FILE_B1!="none") $readmemh(GAMMA_FILE_B1, gamma_buf, DIM, 2*DIM-1);
        if (LOAD_MODE==0 && BETA_FILE    !="none") $readmemh(BETA_FILE,     beta_buf,  0,   DIM-1);
        if (LOAD_MODE==0 && BETA_FILE_B1 !="none") $readmemh(BETA_FILE_B1,  beta_buf,  DIM, 2*DIM-1);
    end
    // nap runtime qua wbus (board): g 1B (sel=GAMMA_ID), b 2B MSB-first (sel=BETA_ID)
    generate if (LOAD_MODE) begin : gen_wr
        wire       w_en   = wbus[0];
        wire       w_rst  = wbus[1];
        wire [4:0] w_sel  = wbus[6:2];
        wire       w_half = wbus[7];
        wire [7:0] w_byte = wbus[15:8];
        reg [CW-1:0] ga;
        always @(posedge clk) if (w_sel==GAMMA_ID) begin
            if (w_rst)     ga <= w_half ? DIM[CW-1:0] : {CW{1'b0}};
            else if (w_en) begin gamma_buf[ga] <= w_byte; ga <= ga + 1'b1; end
        end
        reg [CW-1:0] ba; reg bph; reg [7:0] bhi;
        always @(posedge clk) if (w_sel==BETA_ID) begin
            if (w_rst) begin ba <= w_half ? DIM[CW-1:0] : {CW{1'b0}}; bph <= 1'b0; end
            else if (w_en) begin
                if (!bph) begin bhi <= w_byte; bph <= 1'b1; end
                else      begin beta_buf[ba] <= {bhi, w_byte}; ba <= ba + 1'b1; bph <= 1'b0; end
            end
        end
    end endgenerate
    wire [CW-1:0] coff = block_sel ? DIM[CW-1:0] : {CW{1'b0}};  // offset block

    // ===================== DATAPATH =====================
    reg signed [7:0]  x   [0:DIM-1];
    reg signed [7:0]  yo  [0:DIM-1];
    reg signed [15:0] qn  [0:DIM-1];
    reg signed [15:0] sum;
    reg signed [7:0]  mean;
    reg [RW-1:0]      recip;
    reg signed [8:0]  mx_r, mn_r;   // P-Fmax: max/min chot o S_STAT, dung o S_RECIP

    genvar gi;
    generate for (gi=0; gi<DIM; gi=gi+1) begin: gout
        assign out_data[gi*8 +: 8] = yo[gi];
    end endgenerate

    function signed [7:0] clamp8(input signed [15:0] v);
        clamp8 = (v >  127) ?  8'sd127 : (v < -128) ? -8'sd128 : v[7:0];
    endfunction

    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sum<=0; mean<=0; recip<=0;
            for (i=0;i<DIM;i=i+1) begin x[i]<=0; yo[i]<=0; qn[i]<=0; end
        end else case (state)
            S_IDLE: if (in_valid) begin : latch
                reg signed [15:0] s; s = 0;
                for (i=0;i<DIM;i=i+1) begin
                    x[i] <= $signed(in_data[i*8 +: 8]);
                    s = s + $signed(in_data[i*8 +: 8]);
                end
                sum <= s;
            end
            S_STAT: begin : stat
                // max/min REDUCTION TREE (5 muc thay chuoi 20 noi tiep -> cat critical path)
                reg signed [8:0] xe [0:DIM-1];
                reg signed [8:0] a1 [0:9]; reg signed [8:0] n1 [0:9];
                reg signed [8:0] a2 [0:4]; reg signed [8:0] n2 [0:4];
                reg signed [8:0] a3 [0:2]; reg signed [8:0] n3 [0:2];
                reg signed [8:0] a4 [0:1]; reg signed [8:0] n4 [0:1];
                mean <= (sum * 17'sd6554 + 32'sd65536) >>> 17;
                for (i=0;i<DIM;i=i+1) xe[i] = {x[i][7], x[i]};
                for (i=0;i<10;i=i+1) begin                          // L1: 20->10
                    a1[i] = (xe[2*i] > xe[2*i+1]) ? xe[2*i] : xe[2*i+1];
                    n1[i] = (xe[2*i] < xe[2*i+1]) ? xe[2*i] : xe[2*i+1];
                end
                for (i=0;i<5;i=i+1) begin                           // L2: 10->5
                    a2[i] = (a1[2*i] > a1[2*i+1]) ? a1[2*i] : a1[2*i+1];
                    n2[i] = (n1[2*i] < n1[2*i+1]) ? n1[2*i] : n1[2*i+1];
                end
                a3[0]=(a2[0]>a2[1])?a2[0]:a2[1]; a3[1]=(a2[2]>a2[3])?a2[2]:a2[3]; a3[2]=a2[4]; // L3:5->3
                n3[0]=(n2[0]<n2[1])?n2[0]:n2[1]; n3[1]=(n2[2]<n2[3])?n2[2]:n2[3]; n3[2]=n2[4];
                a4[0]=(a3[0]>a3[1])?a3[0]:a3[1]; a4[1]=a3[2];       // L4: 3->2
                n4[0]=(n3[0]<n3[1])?n3[0]:n3[1]; n4[1]=n3[2];
                mx_r <= (a4[0]>a4[1])?a4[0]:a4[1];                  // L5: 2->1
                mn_r <= (n4[0]<n4[1])?n4[0]:n4[1];
            end
            S_RECIP: begin : reciplk
                reg [7:0] rng;
                rng   = (mx_r - mn_r > 0) ? (mx_r - mn_r) : 8'd1;
                recip <= recip_rom[rng];
            end
            S_MUL: for (i=0;i<DIM;i=i+1) begin : mul
                reg signed [8:0]  c; reg [8:0] cm;
                (* use_dsp = "yes" *) reg [RW+8:0] pr; reg [RW+8:0] qm; reg signed [RW+9:0] qs;
                c  = {x[i][7],x[i]} - {mean[7],mean};
                cm = c[8] ? -c : c;
                pr = cm * recip;
                qm = pr >> RSH;
                qs = c[8] ? -$signed({1'b0,qm}) : $signed({1'b0,qm});
                qn[i] <= (qs > 32767) ? 16'sd32767 : (qs < -32768) ? -16'sd32768 : qs[15:0];
            end
            S_POST: for (i=0;i<DIM;i=i+1) begin : post
                reg [CW-1:0] idx; (* use_dsp = "yes" *) reg signed [23:0] pr; reg signed [15:0] raw;
                idx = coff + i;
                pr  = $signed(gamma_buf[idx]) * qn[i];
                raw = ((pr + 24'sd16384) >>> 15) + beta_buf[idx];
                case (gs)                                     // gamma_shift configurable
                    -4'sd3: raw = raw <<< 3;
                    -4'sd2: raw = raw <<< 2;
                    -4'sd1: raw = raw <<< 1;
                     4'sd1: raw = (raw + 16'sd1) >>> 1;
                     4'sd2: raw = (raw + 16'sd2) >>> 2;
                     4'sd3: raw = (raw + 16'sd4) >>> 3;
                    default: raw = raw;                       // 0 = no shift
                endcase
                yo[i] <= clamp8(raw);
            end
            S_DONE: ;
        endcase
    end
endmodule
