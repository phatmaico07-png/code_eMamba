`timescale 1ns / 1ps
//============================================================================
// ssm_core.v — Selective SSM Recurrence Engine (eMamba paper §4.3 + §4.6)
//
// Computes per-token recurrence given pre-projected inputs:
//   Discretize: Ā = exp(Δ·A),  B̄ = Δ·B
//   State:      h_t = Ā·h_{t-1} + B̄·x_t      (INT24 compute, INT17 store)
//   Output:     y_t = Σ C·h_t + D·x_t          (INT32 acc → INT8)
//
// FSM: IDLE → ITER (N×5 cycles) → OUTPUT (2 cycles) → DONE
// Latency: N×5 + 2 = 42 cycles per token (N=8)
// Resource: 40 DSP (shared) + 40 PWL_Exp (11-seg) + ~5.5K FF (h state)
//============================================================================

module ssm_core #(
    parameter ED           = 40,
    parameter N            = 8,
    parameter Y_SHIFT      = 10,
    parameter H_SHIFT      = 7,        // h state truncation (A_bar_max/2^7 < 1 for stability)
    parameter DA_SHIFT     = 3,        // dA shift (calibrated from δ_s × A_s → EXP_INPUT_SCALE)
    parameter DB_SHIFT     = 3,        // B_bar shift (calibrated separately, can differ from DA_SHIFT)
    parameter DX_ALIGN     = 8,        // D×x alignment shift (calibrated from D_s × x_s / y_acc_s)
    parameter SHARED       = 0,           // 1 = shared mode
    parameter PER_CH_YSHIFT = 0,          // 1 = use per-channel Y_SHIFT from file
    parameter YSHIFT_FILE    = "none",    // ED-entry hex file, per-channel Y_SHIFT (block0)
    parameter YSHIFT_FILE_B1 = "none",    // block1 per-channel Y_SHIFT (SHARED=1 only)
    parameter A_FILE       = "none",
    parameter D_FILE       = "none",
    parameter EXP_BP_FILE  = "none",      // PWL Exp breakpoints (N_SEG+1, INT8)
    parameter EXP_SL_FILE  = "none",      // PWL Exp slopes (N_SEG, INT16)
    parameter EXP_IC_FILE  = "none",      // PWL Exp intercepts (N_SEG, INT16)
    parameter A_FILE_B1       = "none",   // block1 A_log (SHARED=1 only)
    parameter D_FILE_B1       = "none",   // block1 D_param (SHARED=1 only)
    parameter LOAD_MODE       = 0,        // 0=$readmemh (verified) ; 1=runtime byte-load
    parameter [4:0] A_ID      = 5'd0,     // A_log MY_ID (load-map sel)
    parameter [4:0] D_ID      = 5'd0,     // D_param MY_ID
    parameter [4:0] BP_ID     = 5'd0,     // exp bp / sl / ic MY_ID
    parameter [4:0] SL_ID     = 5'd0,
    parameter [4:0] IC_ID     = 5'd0
)(
    input  wire                     clk,
    input  wire                     rst_n,
    // Shared block select (0=block0, 1=block1); ignored when SHARED=0
    input  wire                     block_sel,
    // Runtime shift overrides (0 = use parameter default) — for SHARED per-block values
    input  wire [4:0]               y_shift_rt,
    input  wire [3:0]               da_shift_rt,    // 0 = use DA_SHIFT param
    input  wire [3:0]               db_shift_rt,    // 0 = use DB_SHIFT param
    input  wire [3:0]               dx_align_rt,    // 0 = use DX_ALIGN param
    input  wire                     token_start,
    input  wire                     seq_clear,
    input  wire [ED*8-1:0]         x_in,
    input  wire [ED*8-1:0]         delta_in,
    input  wire [N*8-1:0]          B_in,
    input  wire [N*8-1:0]          C_in,
    output reg  [ED*8-1:0]         y_out,
    output reg                      done,

    // write-bus runtime-load (bundled): [15:8]=byte [7]=half [6:2]=sel [1]=seg_rst [0]=en
    input  wire [15:0]              wbus
);

    // ══════════════════════════════════════════════════════════════
    // FSM
    // ══════════════════════════════════════════════════════════════
    localparam S_IDLE   = 3'b001;
    localparam S_ITER   = 3'b010;
    localparam S_OUTPUT = 3'b100;

    // ── Resolve runtime / parameter shifts ─────────────────────
    wire [3:0] eff_da_shift  = (da_shift_rt != 0) ? da_shift_rt  : DA_SHIFT[3:0];
    wire [3:0] eff_db_shift  = (db_shift_rt != 0) ? db_shift_rt  : DB_SHIFT[3:0];
    wire [3:0] eff_dx_align  = (dx_align_rt != 0) ? dx_align_rt  : DX_ALIGN[3:0];

    (* fsm_encoding = "one_hot" *)
    reg [2:0] state, next_state;

    localparam N_W = $clog2(N > 1 ? N : 2);
    reg [N_W-1:0] n_idx;
    reg [2:0]     step;
    wire          n_last    = (n_idx == N - 1);
    wire          step_last = (step == 3'd4);
    reg           out_step;

    // ══════════════════════════════════════════════════════════════
    // INPUT REGISTERS
    // ══════════════════════════════════════════════════════════════
    reg [ED*8-1:0] x_reg;
    reg [ED*8-1:0] delta_reg;
    reg [N*8-1:0]  B_reg;
    reg [N*8-1:0]  C_reg;

    // ══════════════════════════════════════════════════════════════
    // PARAMETER STORAGE
    // SHARED=1: double-depth arrays — [0..N-1]=block0, [N..2N-1]=block1
    // SHARED=0: single-depth (unused upper half optimized away)
    // ══════════════════════════════════════════════════════════════
    localparam A_DEPTH = SHARED ? 2*N  : N;
    localparam D_DEPTH = SHARED ? 2*ED : ED;

    reg [ED*8-1:0]   A_log   [0:A_DEPTH-1];
    reg signed [15:0] D_param [0:D_DEPTH-1];  // INT16 for h24+aligned (D needs more range)

    initial begin
        if (LOAD_MODE == 0 && A_FILE != "none") begin
            if (SHARED) $readmemh(A_FILE, A_log, 0, N-1);
            else        $readmemh(A_FILE, A_log);
        end
        if (LOAD_MODE == 0 && SHARED && A_FILE_B1 != "none")
            $readmemh(A_FILE_B1, A_log, N, 2*N-1);
        if (LOAD_MODE == 0 && D_FILE != "none") begin
            if (SHARED) $readmemh(D_FILE, D_param, 0, ED-1);
            else        $readmemh(D_FILE, D_param);
        end
        if (LOAD_MODE == 0 && SHARED && D_FILE_B1 != "none")
            $readmemh(D_FILE_B1, D_param, ED, 2*ED-1);
    end

    // ── Per-channel Y_SHIFT table ──────────────────────────────
    localparam YSHIFT_DEPTH = SHARED ? 2*ED : ED;
    reg [4:0] y_shift_ch [0:YSHIFT_DEPTH-1];
    integer ysi;
    initial begin
        // Default: fill with global Y_SHIFT
        for (ysi = 0; ysi < YSHIFT_DEPTH; ysi = ysi + 1)
            y_shift_ch[ysi] = Y_SHIFT[4:0];
        // Load per-channel shifts from file
        if (PER_CH_YSHIFT && YSHIFT_FILE != "none") begin
            if (SHARED) $readmemh(YSHIFT_FILE, y_shift_ch, 0, ED-1);
            else        $readmemh(YSHIFT_FILE, y_shift_ch);
        end
        if (PER_CH_YSHIFT && SHARED && YSHIFT_FILE_B1 != "none")
            $readmemh(YSHIFT_FILE_B1, y_shift_ch, ED, 2*ED-1);
    end

    // ══════════════════════════════════════════════════════════════
    // WRITE-BUS byte-serial — A_log (ED B/word, sel A_ID) + D_param (2B, sel D_ID)
    //   LOAD_MODE=1: nạp runtime; w_half=1 → nửa block1 (A:+N, D:+ED)
    // ══════════════════════════════════════════════════════════════
    localparam AW = $clog2(A_DEPTH > 1 ? A_DEPTH : 2);
    localparam DW = $clog2(D_DEPTH > 1 ? D_DEPTH : 2);
    generate if (LOAD_MODE) begin : gen_wr
        wire        w_en      = wbus[0];
        wire        w_seg_rst = wbus[1];
        wire [4:0]  w_sel     = wbus[6:2];
        wire        w_half    = wbus[7];
        wire [7:0]  w_byte    = wbus[15:8];
        // A_log : ED byte/word (320-bit), N word/half
        reg [AW-1:0] a_addr; reg [ED*8-1:0] a_acc; reg [$clog2(ED>1?ED:2)-1:0] a_bc;
        always @(posedge clk) if (w_sel == A_ID) begin
            if (w_seg_rst) begin a_addr <= w_half ? N[AW-1:0] : {AW{1'b0}}; a_bc <= 0; end
            else if (w_en) begin
                a_acc <= {a_acc[ED*8-9:0], w_byte};
                if (a_bc == ED-1) begin A_log[a_addr] <= {a_acc[ED*8-9:0], w_byte}; a_addr <= a_addr + 1'b1; a_bc <= 0; end
                else a_bc <= a_bc + 1'b1;
            end
        end
        // D_param : 2 byte/word (INT16 MSB-first), ED word/half
        reg [DW-1:0] d_addr; reg d_bc; reg [7:0] d_hi;
        always @(posedge clk) if (w_sel == D_ID) begin
            if (w_seg_rst) begin d_addr <= w_half ? ED[DW-1:0] : {DW{1'b0}}; d_bc <= 0; end
            else if (w_en) begin
                if (d_bc == 0) begin d_hi <= w_byte; d_bc <= 1; end
                else begin D_param[d_addr] <= {d_hi, w_byte}; d_addr <= d_addr + 1'b1; d_bc <= 0; end
            end
        end
    end endgenerate

    // ══════════════════════════════════════════════════════════════
    // HIDDEN STATE h[ED][N] — packed as N columns of ED×17 bits
    // ══════════════════════════════════════════════════════════════
    reg signed [16:0] h_col [0:ED-1];   // current column h[:, n_idx]
    // Full state: N columns stored separately
    reg [ED*17-1:0] h_store [0:N-1];    // packed: h_store[n][e*17 +: 17]

    // ══════════════════════════════════════════════════════════════
    // INTERMEDIATE REGISTERS
    // ══════════════════════════════════════════════════════════════
    reg signed [7:0]  dA_reg    [0:ED-1];
    reg        [7:0]  A_bar_reg [0:ED-1];  // UNSIGNED: exp output [0,128], exp(0)=128 exact
    reg signed [7:0]  B_bar_reg [0:ED-1];
    reg signed [23:0] term1_reg [0:ED-1];
    reg signed [23:0] term2_reg [0:ED-1];
    reg signed [31:0] y_acc     [0:ED-1];  // 32-bit: đủ cho y_max thực tế (~1.7M)

    // ── DEBUG (chi mo phong): h24 = trang thai an INT24 day du = clamp24(term1+term2).
    //    Dung cho waveform. synthesis translate_off -> KHONG anh huong tong hop.
    // synthesis translate_off
    reg signed [23:0] h24_dbg [0:ED-1];
    integer dbg_i; reg signed [24:0] dbg_s;
    always @(*) begin
        for (dbg_i = 0; dbg_i < ED; dbg_i = dbg_i + 1) begin
            dbg_s = $signed(term1_reg[dbg_i]) + $signed(term2_reg[dbg_i]);
            if      (dbg_s >  25'sd8388607) h24_dbg[dbg_i] =  24'sd8388607;
            else if (dbg_s < -25'sd8388608) h24_dbg[dbg_i] = -24'sd8388608;
            else                            h24_dbg[dbg_i] = dbg_s[23:0];
        end
    end
    // synthesis translate_on

    // ══════════════════════════════════════════════════════════════
    // PWL EXP — combinational, ED parallel
    // y = (slope_int16 × x_int8 + ic_int16) >> FRAC
    // Same pattern as linear layer: weight×input + bias >> shift
    // ══════════════════════════════════════════════════════════════
    wire [ED*8-1:0] exp_in;
    wire [ED*8-1:0] exp_out;

    genvar ge;
    generate
        for (ge = 0; ge < ED; ge = ge + 1) begin : gen_exp_pack
            assign exp_in[ge*8 +: 8] = dA_reg[ge];
        end
    endgenerate

    piecewise_lut #(
        .N_SEG(11), .FRAC(7), .MODE(1), .HI_CONST(127),  // FIX: N_SEG=11 khop exp data; HI_CONST=127 khop int_reference clip(y,0,127) (=128 lech 1 LSB)
        .N_ELEM(ED), .PIPELINE(0),
        .BP_FILE(EXP_BP_FILE), .SL_FILE(EXP_SL_FILE), .IC_FILE(EXP_IC_FILE),
        .LOAD_MODE(LOAD_MODE), .MY_ID_BP(BP_ID), .MY_ID_SL(SL_ID), .MY_ID_IC(IC_ID)
    ) u_exp (
        .clk(clk), .x_in(exp_in), .y_out(exp_out),
        .w_en(wbus[0]), .w_seg_rst(wbus[1]), .w_sel(wbus[6:2]), .w_half(wbus[7]), .w_byte(wbus[15:8])
    );

    // ══════════════════════════════════════════════════════════════
    // CURRENT n-th SIGNALS
    // ══════════════════════════════════════════════════════════════
    wire signed [7:0] B_n = $signed(B_reg[n_idx*8 +: 8]);
    wire signed [7:0] C_n = $signed(C_reg[n_idx*8 +: 8]);

    // ══════════════════════════════════════════════════════════════
    // BLOCK 1: STATE REGISTER
    // ══════════════════════════════════════════════════════════════
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) state <= S_IDLE;
        else        state <= next_state;
    end

    // ══════════════════════════════════════════════════════════════
    // BLOCK 2: NEXT STATE LOGIC
    // ══════════════════════════════════════════════════════════════
    always @(*) begin
        next_state = state;
        case (state)
            S_IDLE:   if (token_start)       next_state = S_ITER;
            S_ITER:   if (n_last && step_last) next_state = S_OUTPUT;
            S_OUTPUT: if (out_step)          next_state = S_IDLE;
            default:                         next_state = S_IDLE;
        endcase
    end

    // ══════════════════════════════════════════════════════════════
    // BLOCK 3: DATAPATH
    // ══════════════════════════════════════════════════════════════
    integer e, ni;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            n_idx    <= 0;
            step     <= 0;
            out_step <= 0;
            done     <= 0;
            y_out    <= 0;
            x_reg    <= 0;
            delta_reg <= 0;
            B_reg    <= 0;
            C_reg    <= 0;
            for (e = 0; e < ED; e = e + 1) begin
                dA_reg[e]    <= 0;
                A_bar_reg[e] <= 0;
                B_bar_reg[e] <= 0;
                term1_reg[e] <= 0;
                term2_reg[e] <= 0;
                y_acc[e]     <= 0;
                h_col[e]     <= 0;
            end
            for (ni = 0; ni < N; ni = ni + 1)
                h_store[ni] <= 0;
        end else begin
            done <= 1'b0;

            // ── Seq clear ────────────────────────────────────
            if (seq_clear) begin
                for (ni = 0; ni < N; ni = ni + 1)
                    h_store[ni] <= 0;
            end

            case (state)

                // ── IDLE ─────────────────────────────────────
                S_IDLE: begin
                    n_idx    <= 0;
                    step     <= 0;
                    out_step <= 0;
                    if (token_start) begin
                        x_reg     <= x_in;
                        delta_reg <= delta_in;
                        B_reg     <= B_in;
                        C_reg     <= C_in;
                        for (e = 0; e < ED; e = e + 1)
                            y_acc[e] <= 0;
                        // Load h column 0
                        for (e = 0; e < ED; e = e + 1)
                            h_col[e] <= $signed(h_store[0][e*17 +: 17]);
                    end
                end

                // ── ITER: 5 steps × N ────────────────────────
                S_ITER: begin
                    case (step)

                        // Step 0: dA = (delta × A_log[:, n]) >>> DA_SHIFT (0=no shift)
                        3'd0: begin : step0_blk
                            reg [N_W:0] a_idx;  // extra bit for 2*N range
                            a_idx = block_sel ? (N[N_W:0] + n_idx) : {1'b0, n_idx};
                            for (e = 0; e < ED; e = e + 1) begin : step0_loop
                                reg signed [15:0] p0;
                                reg signed [15:0] s0;
                                p0 = $signed(delta_reg[e*8 +: 8])
                                   * $signed(A_log[a_idx][e*8 +: 8]);
                                s0 = (eff_da_shift > 0) ? ((p0 + (1 <<< (eff_da_shift-1))) >>> eff_da_shift) : p0;
                                // clamp INT8
                                if (s0 > 127)       dA_reg[e] <= 8'sd127;
                                else if (s0 < -128) dA_reg[e] <= -8'sd128;
                                else                dA_reg[e] <= s0[7:0];
                            end
                            step <= 3'd1;
                        end

                        // Step 1: B_bar + A_bar(combinational from dA_reg)
                        3'd1: begin
                            for (e = 0; e < ED; e = e + 1) begin : step1_loop
                                reg signed [15:0] p1;
                                reg signed [15:0] s1;
                                p1 = $signed(delta_reg[e*8 +: 8]) * B_n;
                                s1 = (eff_db_shift > 0) ? ((p1 + (1 <<< (eff_db_shift-1))) >>> eff_db_shift) : p1;
                                if (s1 > 127)       B_bar_reg[e] <= 8'sd127;
                                else if (s1 < -128) B_bar_reg[e] <= -8'sd128;
                                else                B_bar_reg[e] <= s1[7:0];
                                // A_bar from PWL Exp — UNSIGNED [0,128]
                                A_bar_reg[e] <= exp_out[e*8 +: 8];
                            end
                            step <= 3'd2;
                        end

                        // Step 2: term1 = A_bar × h → INT24 (no shift here)
                        // Matches Python SPLIT_H_SHIFT=False: shift AFTER sum
                        // FIX: A_bar is UNSIGNED [0,128] × h is SIGNED INT17
                        3'd2: begin
                            for (e = 0; e < ED; e = e + 1) begin : step2_loop
                                reg signed [25:0] wp;
                                wp = $signed({1'b0, A_bar_reg[e]}) * h_col[e]; // UINT8→INT9 × INT17 = INT26
                                // Clamp to INT24 (no H_SHIFT yet!)
                                if (wp > 8388607)        term1_reg[e] <= 24'sd8388607;
                                else if (wp < -8388608)  term1_reg[e] <= -24'sd8388608;
                                else                     term1_reg[e] <= wp[23:0];
                            end
                            step <= 3'd3;
                        end

                        // Step 3: term2 = B_bar × x_t → INT16 (stays INT24 reg, sign-extended)
                        3'd3: begin
                            for (e = 0; e < ED; e = e + 1) begin : step3_loop
                                reg signed [15:0] p3;
                                p3 = $signed(B_bar_reg[e])
                                   * $signed(x_reg[e*8 +: 8]);
                                // sign-extend INT16 → INT24
                                term2_reg[e] <= {{8{p3[15]}}, p3};
                            end
                            step <= 3'd4;
                        end

                        // Step 4: y dùng h24, shift sau lưu h17 (paper §4.6)
                        3'd4: begin
                            for (e = 0; e < ED; e = e + 1) begin : step4_loop
                                reg signed [24:0] hc;
                                reg signed [23:0] h24;
                                reg signed [24:0] h_shifted;
                                reg signed [16:0] h17;
                                reg signed [31:0] cy;

                                hc = $signed(term1_reg[e]) + $signed(term2_reg[e]);
                                if (hc > 8388607)        h24 = 24'sd8388607;
                                else if (hc < -8388608)  h24 = -24'sd8388608;
                                else                     h24 = hc[23:0];

                                cy = C_n * h24;
                                y_acc[e] <= y_acc[e] + cy;

                                h_shifted = (h24 + (1 <<< (H_SHIFT-1))) >>> H_SHIFT;
                                if (h_shifted > 65535)       h17 = 17'sd65535;
                                else if (h_shifted < -65536) h17 = -17'sd65536;
                                else                         h17 = h_shifted[16:0];

                                h_col[e] <= h17;
                            end

                            // Save current h_col back to h_store
                            // (will be written with new values from above)
                            // Actually need to save AFTER h_col update takes effect
                            // So save in next step or use combinational

                            if (n_last) begin
                                // Save last column and move to OUTPUT
                                step <= 0;
                            end else begin
                                n_idx <= n_idx + 1;
                                step  <= 0;
                            end
                        end

                        default: step <= 0;
                    endcase

                    // Save h_col to h_store at step transitions
                    // h_col was updated in step 4 (non-blocking),
                    // so save PREVIOUS column at step 0 of next n
                    if (step == 3'd0 && n_idx > 0) begin
                        for (e = 0; e < ED; e = e + 1)
                            h_store[n_idx - 1][e*17 +: 17] <= h_col[e];
                    end

                    // Load next h column at start of new n (step 0)
                    if (step == 3'd0) begin
                        for (e = 0; e < ED; e = e + 1)
                            h_col[e] <= $signed(h_store[n_idx][e*17 +: 17]);
                    end
                end

                // ── OUTPUT ───────────────────────────────────
                S_OUTPUT: begin
                    // Save last h_col on entry
                    if (!out_step) begin
                        // Save final column
                        for (e = 0; e < ED; e = e + 1)
                            h_store[N-1][e*17 +: 17] <= h_col[e];

                        // D(INT16) × x(INT8) >>> (H_SHIFT+1) to align with C×h24
                        for (e = 0; e < ED; e = e + 1) begin : out_dx_loop
                            reg signed [23:0] dx;       // INT16 × INT8 = INT24
                            reg signed [23:0] dx_aligned;
                            reg [$clog2(D_DEPTH)-1:0] d_idx;
                            d_idx = (block_sel ? ED : 0) + e;
                            dx = D_param[d_idx]
                               * $signed(x_reg[e*8 +: 8]);
                            dx_aligned = (eff_dx_align > 0) ? ((dx + (1 <<< (eff_dx_align-1))) >>> eff_dx_align) : dx;
                            y_acc[e] <= y_acc[e] + {{8{dx_aligned[23]}}, dx_aligned};
                        end
                        out_step <= 1;
                    end else begin
                        // Requant — per-channel Y_SHIFT or global
                        for (e = 0; e < ED; e = e + 1) begin : out_rq_loop
                            reg signed [31:0] sv;
                            reg [4:0]         ys_eff;
                            reg [$clog2(YSHIFT_DEPTH)-1:0] ys_idx;
                            ys_idx = (block_sel ? ED : 0) + e;
                            ys_eff = PER_CH_YSHIFT ? y_shift_ch[ys_idx] :
                                     (y_shift_rt != 0) ? y_shift_rt : Y_SHIFT[4:0];
                            sv = (y_acc[e] + (1 <<< (ys_eff-1))) >>> ys_eff;
                            if (sv > 127)       y_out[e*8 +: 8] <= 8'sd127;
                            else if (sv < -128) y_out[e*8 +: 8] <= -8'sd128;
                            else                y_out[e*8 +: 8] <= sv[7:0];
                        end
                        done <= 1'b1;
                    end
                end

            endcase
        end
    end

endmodule
