//===================================================================
// TrapezoidPipeline Top Module
// 5-stage Pipeline Architecture for Matrix Multiplication
//===================================================================

module trapezoid_pipeline #(
    parameter M = 1,
    parameter K = 32,
    parameter N = 256,
    parameter PE_NUM = 4,
    parameter BF16_WIDTH = 16,
    parameter INDEX_WIDTH = 16,
    parameter VALUE_QUEUE_DEPTH = 64,
    parameter MAX_VALUES = 128
)(
    input  logic                    clk,
    input  logic                    rst_n,
    
    // Input interface
    input  logic                    valid_in,
    input  logic                    is_hbm,
    input  logic [INDEX_WIDTH-1:0]  start_index,
    
    // Matrix A inputs (dense vector)
    input  logic [K*BF16_WIDTH-1:0] matrix_a,
    
    // Matrix B inputs (sparse CSR format)
    input  logic [MAX_VALUES*BF16_WIDTH-1:0] values_b,
    input  logic [MAX_VALUES*INDEX_WIDTH-1:0] col_indices,
    input  logic [MAX_VALUES*INDEX_WIDTH-1:0] row_ptr,
    input  logic [7:0]               num_values_b,
    
    // Output interface
    output logic                    valid_out,
    output logic [M*N*BF16_WIDTH-1:0] c_matrix_out,
    
    // Status
    output logic                    pipeline_active
);

    // Internal signals for each stage
    logic stage1_valid, stage2_valid, stage3_valid, stage4_valid, stage5_valid;
    
    // Stage 1 -> Stage 2
    logic [MAX_VALUES*BF16_WIDTH-1:0] stage1_values_a, stage1_values_b;
    logic [MAX_VALUES*INDEX_WIDTH-1:0] stage1_offset_a, stage1_offset_b;
    logic [MAX_VALUES-1:0] stage1_masks_a, stage1_masks_b;
    logic [7:0] stage1_len_a, stage1_len_b;
    
    // Stage 2 -> Stage 3
    logic [PE_NUM-1:0][MAX_VALUES*INDEX_WIDTH-1:0] stage2_index_a, stage2_index_b;
    logic [MAX_VALUES*BF16_WIDTH-1:0] stage2_values_a, stage2_values_b;
    
    // Stage 3 -> Stage 4
    logic [PE_NUM-1:0][VALUE_QUEUE_DEPTH*BF16_WIDTH-1:0] stage3_queue_a, stage3_queue_b;
    logic [PE_NUM-1:0][VALUE_QUEUE_DEPTH*INDEX_WIDTH-1:0] stage3_index_queue;
    logic [PE_NUM-1:0][7:0] stage3_queue_sizes;
    
    // Stage 4 -> Stage 5
    logic [PE_NUM-1:0]      stage4_mul_valid;
    logic [PE_NUM-1:0][BF16_WIDTH-1:0] stage4_mul_results;
    logic [PE_NUM-1:0][INDEX_WIDTH-1:0] stage4_mul_indices;
    
    // Output C matrix storage
    logic [M*N-1:0][BF16_WIDTH-1:0] c_matrix_reg;

    //===================================================================
    // Stage 1: Input Processing
    //===================================================================
    
    stage1_input_processor #(
        .M(M), .K(K), .N(N),
        .BF16_WIDTH(BF16_WIDTH),
        .INDEX_WIDTH(INDEX_WIDTH),
        .MAX_VALUES(MAX_VALUES)
    ) stage1_inst (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .is_hbm(is_hbm),
        .matrix_a(matrix_a),
        .values_b(values_b),
        .col_indices(col_indices),
        .row_ptr(row_ptr),
        .num_values_b(num_values_b),
        
        .valid_out(stage1_valid),
        .values_a_out(stage1_values_a),
        .values_b_out(stage1_values_b),
        .offset_a_out(stage1_offset_a),
        .offset_b_out(stage1_offset_b),
        .masks_a_out(stage1_masks_a),
        .masks_b_out(stage1_masks_b),
        .len_a_out(stage1_len_a),
        .len_b_out(stage1_len_b)
    );

    //===================================================================
    // Stage 2: MFIU Pipeline
    //===================================================================
    
    mfiu_pipeline #(
        .BF16_WIDTH(BF16_WIDTH),
        .INDEX_WIDTH(INDEX_WIDTH),
        .MAX_VALUES(MAX_VALUES),
        .PE_NUM(PE_NUM)
    ) mfiu_inst (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(stage1_valid),
        .masks_a_in(stage1_masks_a),
        .masks_b_in(stage1_masks_b),
        .offset_a_in(stage1_offset_a),
        .offset_b_in(stage1_offset_b),
        .len_a_in(stage1_len_a),
        .len_b_in(stage1_len_b),
        
        .valid_out(stage2_valid),
        .index_a_out(stage2_index_a),
        .index_b_out(stage2_index_b)
    );

    //===================================================================
    // Stage 3: Queue Management
    //===================================================================
    
    queue_manager #(
        .PE_NUM(PE_NUM),
        .BF16_WIDTH(BF16_WIDTH),
        .INDEX_WIDTH(INDEX_WIDTH),
        .VALUE_QUEUE_DEPTH(VALUE_QUEUE_DEPTH),
        .MAX_VALUES(MAX_VALUES)
    ) queue_mgr_inst (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(stage2_valid),
        .index_a_in(stage2_index_a),
        .index_b_in(stage2_index_b),
        .values_a_in(stage1_values_a),  // From queue
        .values_b_in(stage1_values_b),  // From queue
        .start_index(start_index),
        
        .valid_out(stage3_valid),
        .queue_a_out(stage3_queue_a),
        .queue_b_out(stage3_queue_b),
        .index_queue_out(stage3_index_queue),
        .queue_sizes_out(stage3_queue_sizes)
    );

    //===================================================================
    // Stage 4: Multiply Units (PE_NUM parallel multipliers)
    //===================================================================
    
    genvar i;
    generate
        for (i = 0; i < PE_NUM; i++) begin : mul_units
            multiply_unit #(
                .BF16_WIDTH(BF16_WIDTH),
                .INDEX_WIDTH(INDEX_WIDTH)
            ) mul_inst (
                .clk(clk),
                .rst_n(rst_n),
                .valid_in(stage3_valid && (stage3_queue_sizes[i] > 0)),
                .value_a_in(stage3_queue_a[i][BF16_WIDTH-1:0]),
                .value_b_in(stage3_queue_b[i][BF16_WIDTH-1:0]),
                .index_in(stage3_index_queue[i][INDEX_WIDTH-1:0]),
                
                .valid_out(stage4_mul_valid[i]),
                .result_out(stage4_mul_results[i]),
                .index_out(stage4_mul_indices[i])
            );
        end
    endgenerate

    //===================================================================
    // Stage 5: Add Tree
    //===================================================================
    
    add_tree #(
        .PE_NUM(PE_NUM),
        .BF16_WIDTH(BF16_WIDTH),
        .INDEX_WIDTH(INDEX_WIDTH),
        .M(M), .N(N)
    ) add_tree_inst (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(|stage4_mul_valid),
        .mul_valid_in(stage4_mul_valid),
        .mul_results_in(stage4_mul_results),
        .mul_indices_in(stage4_mul_indices),
        
        .valid_out(stage5_valid),
        .c_matrix_update(c_matrix_reg)
    );

    //===================================================================
    // Output Logic
    //===================================================================
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            c_matrix_out <= '0;
        end else begin
            valid_out <= stage5_valid;
            if (stage5_valid) begin
                // Flatten c_matrix_reg to output
                for (int j = 0; j < M*N; j++) begin
                    c_matrix_out[j*BF16_WIDTH +: BF16_WIDTH] <= c_matrix_reg[j];
                end
            end
        end
    end

    // Pipeline active status
    assign pipeline_active = stage1_valid || stage2_valid || stage3_valid || 
                             stage4_valid || stage5_valid;

endmodule

//===================================================================
// Stage 1: Input Processor
//===================================================================

module stage1_input_processor #(
    parameter M = 1,
    parameter K = 32,
    parameter N = 256,
    parameter BF16_WIDTH = 16,
    parameter INDEX_WIDTH = 16,
    parameter MAX_VALUES = 128
)(
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    valid_in,
    input  logic                    is_hbm,
    input  logic [K*BF16_WIDTH-1:0] matrix_a,
    input  logic [MAX_VALUES*BF16_WIDTH-1:0] values_b,
    input  logic [MAX_VALUES*INDEX_WIDTH-1:0] col_indices,
    input  logic [MAX_VALUES*INDEX_WIDTH-1:0] row_ptr,
    input  logic [7:0]               num_values_b,
    
    output logic                    valid_out,
    output logic [MAX_VALUES*BF16_WIDTH-1:0] values_a_out,
    output logic [MAX_VALUES*BF16_WIDTH-1:0] values_b_out,
    output logic [MAX_VALUES*INDEX_WIDTH-1:0] offset_a_out,
    output logic [MAX_VALUES*INDEX_WIDTH-1:0] offset_b_out,
    output logic [MAX_VALUES-1:0]    masks_a_out,
    output logic [MAX_VALUES-1:0]    masks_b_out,
    output logic [7:0]               len_a_out,
    output logic [7:0]               len_b_out
);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            values_a_out <= '0;
            values_b_out <= '0;
            offset_a_out <= '0;
            offset_b_out <= '0;
            masks_a_out <= '0;
            masks_b_out <= '0;
            len_a_out <= '0;
            len_b_out <= '0;
        end else begin
            valid_out <= valid_in;
            
            if (valid_in) begin
                // Process Matrix A (dense to sparse conversion)
                len_a_out <= K;  // All K elements are non-zero
                for (int i = 0; i < K; i++) begin
                    values_a_out[i*BF16_WIDTH +: BF16_WIDTH] <= matrix_a[i*BF16_WIDTH +: BF16_WIDTH];
                    offset_a_out[i*INDEX_WIDTH +: INDEX_WIDTH] <= i;
                    masks_a_out[i] <= (matrix_a[i*BF16_WIDTH +: BF16_WIDTH] != 0);
                end
                
                // Process Matrix B
                len_b_out <= num_values_b;
                values_b_out <= values_b;
                offset_b_out <= col_indices;
                for (int i = 0; i < MAX_VALUES; i++) begin
                    masks_b_out[i] <= (i < num_values_b) ? 
                                      (values_b[i*BF16_WIDTH +: BF16_WIDTH] != 0) : 1'b0;
                end
            end
        end
    end

endmodule

//===================================================================
// MFIU Pipeline 
//===================================================================

module mfiu_pipeline #(
    parameter BF16_WIDTH = 16,
    parameter INDEX_WIDTH = 16,
    parameter MAX_VALUES = 128,
    parameter PE_NUM = 4
)(
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    valid_in,
    input  logic [MAX_VALUES-1:0]   masks_a_in,
    input  logic [MAX_VALUES-1:0]   masks_b_in,
    input  logic [MAX_VALUES*INDEX_WIDTH-1:0] offset_a_in,
    input  logic [MAX_VALUES*INDEX_WIDTH-1:0] offset_b_in,
    input  logic [7:0]              len_a_in,
    input  logic [7:0]              len_b_in,
    
    output logic                    valid_out,
    output logic [PE_NUM-1:0][MAX_VALUES*INDEX_WIDTH-1:0] index_a_out,
    output logic [PE_NUM-1:0][MAX_VALUES*INDEX_WIDTH-1:0] index_b_out
);

    // 5-stage MFIU pipeline
    logic [4:0] valid_reg;
    logic [MAX_VALUES-1:0] masks_a_reg [4:0];
    logic [MAX_VALUES-1:0] masks_b_reg [4:0];
    logic [MAX_VALUES*INDEX_WIDTH-1:0] offset_a_reg [4:0];
    logic [MAX_VALUES*INDEX_WIDTH-1:0] offset_b_reg [4:0];
    logic [7:0] len_a_reg [4:0];
    logic [7:0] len_b_reg [4:0];
    
    // Stage intermediate results
    logic [MAX_VALUES-1:0] ab_mask;
    logic [MAX_VALUES-1:0] bit_seq;
    logic [MAX_VALUES-1:0] prefix_sum;
    logic [PE_NUM-1:0][MAX_VALUES-1:0] ec_idx_vec;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_reg <= 5'b0;
            for (int i = 0; i < 5; i++) begin
                masks_a_reg[i] <= '0;
                masks_b_reg[i] <= '0;
                offset_a_reg[i] <= '0;
                offset_b_reg[i] <= '0;
                len_a_reg[i] <= '0;
                len_b_reg[i] <= '0;
            end
        end else begin
            // Pipeline stages
            valid_reg[0] <= valid_in;
            masks_a_reg[0] <= masks_a_in;
            masks_b_reg[0] <= masks_b_in;
            offset_a_reg[0] <= offset_a_in;
            offset_b_reg[0] <= offset_b_in;
            len_a_reg[0] <= len_a_in;
            len_b_reg[0] <= len_b_in;
            
            for (int i = 1; i < 5; i++) begin
                valid_reg[i] <= valid_reg[i-1];
                masks_a_reg[i] <= masks_a_reg[i-1];
                masks_b_reg[i] <= masks_b_reg[i-1];
                offset_a_reg[i] <= offset_a_reg[i-1];
                offset_b_reg[i] <= offset_b_reg[i-1];
                len_a_reg[i] <= len_a_reg[i-1];
                len_b_reg[i] <= len_b_reg[i-1];
            end
        end
    end

    // Stage 2: Compute AB mask
    always_comb begin
        ab_mask = masks_a_reg[1] & masks_b_reg[1];
    end

    // Stage 3: Compute prefix sum
    always_comb begin
        bit_seq = ab_mask;
        prefix_sum[0] = bit_seq[0];
        for (int i = 1; i < MAX_VALUES; i++) begin
            prefix_sum[i] = prefix_sum[i-1] + bit_seq[i];
        end
    end

    // Stage 4: Compute EC indices
    always_comb begin
        for (int pe = 0; pe < PE_NUM; pe++) begin
            for (int i = 0; i < MAX_VALUES; i++) begin
                ec_idx_vec[pe][i] = bit_seq[i] ? prefix_sum[i] : 1'b0;
            end
        end
    end

    // Stage 5: Generate output indices
    always_comb begin
        valid_out = valid_reg[4];
        for (int pe = 0; pe < PE_NUM; pe++) begin
            for (int i = 0; i < MAX_VALUES; i++) begin
                if (ec_idx_vec[pe][i] != 0 && ((ec_idx_vec[pe][i] - 1) % PE_NUM) == pe) begin
                    index_a_out[pe][i*INDEX_WIDTH +: INDEX_WIDTH] = 
                        offset_a_reg[4][i*INDEX_WIDTH +: INDEX_WIDTH];
                    index_b_out[pe][i*INDEX_WIDTH +: INDEX_WIDTH] = 
                        offset_b_reg[4][i*INDEX_WIDTH +: INDEX_WIDTH];
                end else begin
                    index_a_out[pe][i*INDEX_WIDTH +: INDEX_WIDTH] = '0;
                    index_b_out[pe][i*INDEX_WIDTH +: INDEX_WIDTH] = '0;
                end
            end
        end
    end

endmodule

//===================================================================
// Queue Manager
//===================================================================

module queue_manager #(
    parameter PE_NUM = 4,
    parameter BF16_WIDTH = 16,
    parameter INDEX_WIDTH = 16,
    parameter VALUE_QUEUE_DEPTH = 64,
    parameter MAX_VALUES = 128
)(
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    valid_in,
    input  logic [PE_NUM-1:0][MAX_VALUES*INDEX_WIDTH-1:0] index_a_in,
    input  logic [PE_NUM-1:0][MAX_VALUES*INDEX_WIDTH-1:0] index_b_in,
    input  logic [MAX_VALUES*BF16_WIDTH-1:0] values_a_in,
    input  logic [MAX_VALUES*BF16_WIDTH-1:0] values_b_in,
    input  logic [INDEX_WIDTH-1:0]  start_index,
    
    output logic                    valid_out,
    output logic [PE_NUM-1:0][VALUE_QUEUE_DEPTH*BF16_WIDTH-1:0] queue_a_out,
    output logic [PE_NUM-1:0][VALUE_QUEUE_DEPTH*BF16_WIDTH-1:0] queue_b_out,
    output logic [PE_NUM-1:0][VALUE_QUEUE_DEPTH*INDEX_WIDTH-1:0] index_queue_out,
    output logic [PE_NUM-1:0][7:0]  queue_sizes_out
);

    // Queue storage
    logic [PE_NUM-1:0][VALUE_QUEUE_DEPTH-1:0][BF16_WIDTH-1:0] queue_a_mem;
    logic [PE_NUM-1:0][VALUE_QUEUE_DEPTH-1:0][BF16_WIDTH-1:0] queue_b_mem;
    logic [PE_NUM-1:0][VALUE_QUEUE_DEPTH-1:0][INDEX_WIDTH-1:0] index_queue_mem;
    
    // Queue pointers
    logic [PE_NUM-1:0][7:0] queue_write_ptr, queue_read_ptr, queue_size;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            queue_write_ptr <= '0;
            queue_read_ptr <= '0;
            queue_size <= '0;
            valid_out <= 1'b0;
        end else begin
            valid_out <= |queue_size;  // Valid if any queue has data
            
            if (valid_in) begin
                // Distribute values to queues based on indices
                for (int pe = 0; pe < PE_NUM; pe++) begin
                    for (int i = 0; i < MAX_VALUES; i++) begin
                        if (index_a_in[pe][i*INDEX_WIDTH +: INDEX_WIDTH] != 0) begin
                            // Add to queue
                            if (queue_size[pe] < VALUE_QUEUE_DEPTH) begin
                                queue_a_mem[pe][queue_write_ptr[pe]] <= 
                                    values_a_in[i*BF16_WIDTH +: BF16_WIDTH];
                                queue_b_mem[pe][queue_write_ptr[pe]] <= 
                                    values_b_in[i*BF16_WIDTH +: BF16_WIDTH];
                                index_queue_mem[pe][queue_write_ptr[pe]] <= 
                                    start_index + i;
                                queue_write_ptr[pe] <= queue_write_ptr[pe] + 1;
                                queue_size[pe] <= queue_size[pe] + 1;
                            end
                        end
                    end
                end
            end
            
            // Pop from queues when consumed
            for (int pe = 0; pe < PE_NUM; pe++) begin
                if (queue_size[pe] > 0) begin
                    queue_read_ptr[pe] <= queue_read_ptr[pe] + 1;
                    queue_size[pe] <= queue_size[pe] - 1;
                end
            end
        end
    end

    // Output assignments
    always_comb begin
        for (int pe = 0; pe < PE_NUM; pe++) begin
            queue_sizes_out[pe] = queue_size[pe];
            for (int i = 0; i < VALUE_QUEUE_DEPTH; i++) begin
                queue_a_out[pe][i*BF16_WIDTH +: BF16_WIDTH] = queue_a_mem[pe][i];
                queue_b_out[pe][i*BF16_WIDTH +: BF16_WIDTH] = queue_b_mem[pe][i];
                index_queue_out[pe][i*INDEX_WIDTH +: INDEX_WIDTH] = index_queue_mem[pe][i];
            end
        end
    end

endmodule

//===================================================================
// Multiply Unit
//===================================================================

module multiply_unit #(
    parameter BF16_WIDTH = 16,
    parameter INDEX_WIDTH = 16
)(
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    valid_in,
    input  logic [BF16_WIDTH-1:0]   value_a_in,
    input  logic [BF16_WIDTH-1:0]   value_b_in,
    input  logic [INDEX_WIDTH-1:0]  index_in,
    
    output logic                    valid_out,
    output logic [BF16_WIDTH-1:0]   result_out,
    output logic [INDEX_WIDTH-1:0]  index_out
);

    // BF16 multiplication pipeline (5 stages)
    logic [4:0] valid_pipe;
    logic [BF16_WIDTH-1:0] a_pipe [4:0];
    logic [BF16_WIDTH-1:0] b_pipe [4:0];
    logic [INDEX_WIDTH-1:0] index_pipe [4:0];
    logic [BF16_WIDTH-1:0] result_pipe [4:0];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_pipe <= 5'b0;
            for (int i = 0; i < 5; i++) begin
                a_pipe[i] <= '0;
                b_pipe[i] <= '0;
                index_pipe[i] <= '0;
                result_pipe[i] <= '0;
            end
        end else begin
            // Stage 1: Input
            valid_pipe[0] <= valid_in;
            a_pipe[0] <= value_a_in;
            b_pipe[0] <= value_b_in;
            index_pipe[0] <= index_in;
            
            // Stages 2-5: Pipeline
            for (int i = 1; i < 5; i++) begin
                valid_pipe[i] <= valid_pipe[i-1];
                a_pipe[i] <= a_pipe[i-1];
                b_pipe[i] <= b_pipe[i-1];
                index_pipe[i] <= index_pipe[i-1];
            end
            
            // BF16 multiplication (simplified)
            result_pipe[0] <= a_pipe[0] * b_pipe[0];  // Simplified for synthesis
            for (int i = 1; i < 5; i++) begin
                result_pipe[i] <= result_pipe[i-1];
            end
        end
    end

    assign valid_out = valid_pipe[4];
    assign result_out = result_pipe[4];
    assign index_out = index_pipe[4];

endmodule

//===================================================================
// Add Tree
//===================================================================

module add_tree #(
    parameter PE_NUM = 4,
    parameter BF16_WIDTH = 16,
    parameter INDEX_WIDTH = 16,
    parameter M = 1,
    parameter N = 256
)(
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    valid_in,
    input  logic [PE_NUM-1:0]       mul_valid_in,
    input  logic [PE_NUM-1:0][BF16_WIDTH-1:0] mul_results_in,
    input  logic [PE_NUM-1:0][INDEX_WIDTH-1:0] mul_indices_in,
    
    output logic                    valid_out,
    output logic [M*N-1:0][BF16_WIDTH-1:0] c_matrix_update
);

    localparam TREE_LEVELS = $clog2(PE_NUM);
    
    // Tree storage for each level
    logic [TREE_LEVELS:0] level_valid;
    logic [TREE_LEVELS:0][PE_NUM-1:0][BF16_WIDTH-1:0] level_values;
    logic [TREE_LEVELS:0][PE_NUM-1:0][INDEX_WIDTH-1:0] level_indices;
    
    // C matrix accumulator
    logic [M*N-1:0][BF16_WIDTH-1:0] c_matrix_reg;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            level_valid <= '0;
            level_values <= '0;
            level_indices <= '0;
            c_matrix_reg <= '0;
            valid_out <= 1'b0;
        end else begin
            // Level 0: Input from multipliers
            level_valid[0] <= valid_in;
            if (valid_in) begin
                for (int pe = 0; pe < PE_NUM; pe++) begin
                    if (mul_valid_in[pe]) begin
                        level_values[0][pe] <= mul_results_in[pe];
                        level_indices[0][pe] <= mul_indices_in[pe];
                    end
                end
            end
            
            // Tree levels: Add pairs
            for (int level = 1; level <= TREE_LEVELS; level++) begin
                level_valid[level] <= level_valid[level-1];
                if (level_valid[level-1]) begin
                    for (int pe = 0; pe < (PE_NUM >> level); pe++) begin
                        int pe1 = pe * 2;
                        int pe2 = pe * 2 + 1;
                        if (level_indices[level-1][pe1] == level_indices[level-1][pe2] && 
                            level_indices[level-1][pe1] != 0) begin
                            // Same index, add values
                            level_values[level][pe] <= 
                                level_values[level-1][pe1] + level_values[level-1][pe2];
                            level_indices[level][pe] <= level_indices[level-1][pe1];
                        end else if (level_indices[level-1][pe1] != 0) begin
                            level_values[level][pe] <= level_values[level-1][pe1];
                            level_indices[level][pe] <= level_indices[level-1][pe1];
                        end else if (level_indices[level-1][pe2] != 0) begin
                            level_values[level][pe] <= level_values[level-1][pe2];
                            level_indices[level][pe] <= level_indices[level-1][pe2];
                        end
                    end
                end
            end
            
            // Final level: Update C matrix
            valid_out <= level_valid[TREE_LEVELS];
            if (level_valid[TREE_LEVELS]) begin
                for (int pe = 0; pe < (PE_NUM >> TREE_LEVELS); pe++) begin
                    if (level_indices[TREE_LEVELS][pe] != 0 && 
                        level_indices[TREE_LEVELS][pe] < M*N) begin
                        c_matrix_reg[level_indices[TREE_LEVELS][pe]] <= 
                            c_matrix_reg[level_indices[TREE_LEVELS][pe]] + 
                            level_values[TREE_LEVELS][pe];
                    end
                end
            end
        end
    end

    assign c_matrix_update = c_matrix_reg;

endmodule

//===================================================================
// Testbench for Design Compiler
//===================================================================

module trapezoid_tb;
    parameter M = 1;
    parameter K = 32;
    parameter N = 256;
    parameter PE_NUM = 4;
    parameter BF16_WIDTH = 16;
    parameter INDEX_WIDTH = 16;
    parameter MAX_VALUES = 128;

    logic clk, rst_n;
    logic valid_in, is_hbm, valid_out, pipeline_active;
    logic [INDEX_WIDTH-1:0] start_index;
    logic [K*BF16_WIDTH-1:0] matrix_a;
    logic [MAX_VALUES*BF16_WIDTH-1:0] values_b;
    logic [MAX_VALUES*INDEX_WIDTH-1:0] col_indices, row_ptr;
    logic [7:0] num_values_b;
    logic [M*N*BF16_WIDTH-1:0] c_matrix_out;

    trapezoid_pipeline #(
        .M(M), .K(K), .N(N), .PE_NUM(PE_NUM),
        .BF16_WIDTH(BF16_WIDTH), .INDEX_WIDTH(INDEX_WIDTH),
        .MAX_VALUES(MAX_VALUES)
    ) dut (.*);

    // Clock generation
    always #5 clk = ~clk;

    initial begin
        clk = 0;
        rst_n = 0;
        valid_in = 0;
        is_hbm = 1;
        start_index = 0;
        matrix_a = '0;
        values_b = '0;
        col_indices = '0;
        row_ptr = '0;
        num_values_b = 0;

        #20 rst_n = 1;
        
        // Test case
        #10;
        valid_in = 1;
        matrix_a = {32{16'h3F80}}; // 1.0 in BF16
        values_b[15:0] = 16'h4000;  // 2.0 in BF16
        col_indices[15:0] = 16'h0001;
        num_values_b = 1;
        
        #10 valid_in = 0;
        
        // Wait for pipeline to complete
        wait(valid_out);
        #100;
        
        $finish;
    end

    // Monitor
    always @(posedge clk) begin
        if (valid_out) begin
            $display("Time %0t: Output valid, C[0] = %h", $time, c_matrix_out[15:0]);
        end
    end

endmodule