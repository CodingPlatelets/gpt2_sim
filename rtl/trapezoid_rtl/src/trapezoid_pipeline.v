// Trapezoid Pipeline - Main Module
// 8x128 PE Array for Sparse Matrix Multiplication
module trapezoid_pipeline #(
    parameter M = 8,
    parameter K = 128,
    parameter N = 128,
    parameter PE_NUM = 4
)(
    input wire clk,
    input wire rst_n,
    
    // Input interface
    input wire valid_in,
    input wire [K*16-1:0] A_values,           // Dense vector A (1xK)
    input wire [16*32-1:0] B_values,          // Sparse matrix B values
    input wire [16*16-1:0] B_col_indices,    // Column indices for B
    input wire [17*16-1:0] B_row_ptr,        // Row pointers for B
    input wire [15:0] start_index,
    input wire is_hbm,
    
    // Output interface
    output reg valid_out,
    output reg [M*N*16-1:0] C_values,        // Result matrix C
    output reg [31:0] cycle_count
);

    // Internal pipeline stage signals
    // Stage 1: Input processing
    reg stage1_valid;
    reg [K*16-1:0] stage1_A_values;
    reg [16*32-1:0] stage1_B_values;
    reg [31:0] stage1_A_offset [0:M-1];
    reg [31:0] stage1_B_offset [0:N-1];
    reg [31:0] stage1_A_masks [0:M-1];
    reg [31:0] stage1_B_masks [0:N-1];

    // Stage 2: MFIU processing
    reg stage2_valid;
    reg [15:0] stage2_indices_A [0:M*K-1];
    reg [15:0] stage2_indices_B [0:N*K-1];
    reg [K*16-1:0] stage2_values_A;
    reg [16*32-1:0] stage2_values_B;

    // Stage 3: Input queue distribution
    reg stage3_valid;
    reg [15:0] stage3_mul_queue_a [0:PE_NUM-1][0:31];
    reg [15:0] stage3_mul_queue_b [0:PE_NUM-1][0:31];
    reg [15:0] stage3_sft_index_queue [0:PE_NUM-1][0:31];
    reg [4:0] stage3_queue_depth [0:PE_NUM-1];

    // Stage 4: Multiply units
    reg stage4_valid;
    wire [PE_NUM-1:0] mul_valid_out;
    wire [15:0] mul_results [0:PE_NUM-1];
    wire [15:0] mul_indices [0:PE_NUM-1];

    // Stage 5: Add tree
    reg stage5_valid;
    wire add_tree_valid;
    wire [15:0] add_tree_index;
    wire [15:0] add_tree_value;
    wire add_tree_update_valid;

    // Multiply units array
    genvar i;
    generate
        for (i = 0; i < PE_NUM; i = i + 1) begin : multiply_units
            multiply_unit mul_inst (
                .clk(clk),
                .rst_n(rst_n),
                .input_valid(stage3_valid && stage3_queue_depth[i] > 0),
                .input1(stage3_mul_queue_a[i][0]),
                .input2(stage3_mul_queue_b[i][0]),
                .sft_index(stage3_sft_index_queue[i][0]),
                .valid_out(mul_valid_out[i]),
                .result(mul_results[i]),
                .index_out(mul_indices[i])
            );
        end
    endgenerate

    // Add tree for final accumulation
    add_tree #(
        .PE_NUM(PE_NUM),
        .M(M),
        .N(N),
        .TREE_LEVELS(2)
    ) add_tree_inst (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(stage4_valid),
        .input_map_queue({PE_NUM*32{1'b0}}), // Placeholder
        .input_indices({mul_indices[3], mul_indices[2], mul_indices[1], mul_indices[0]}),
        .input_values({mul_results[3], mul_results[2], mul_results[1], mul_results[0]}),
        .input_valid_mask(mul_valid_out),
        .valid_out(add_tree_valid),
        .output_index(add_tree_index),
        .output_value(add_tree_value),
        .output_update_valid(add_tree_update_valid)
    );

    // CSR format processing for sparse matrix B
    reg [15:0] csr_values [0:K*N-1];
    reg [15:0] csr_col_indices [0:K*N-1];
    reg [16:0] csr_row_ptr [0:N];
    
    // C matrix storage (flattened)
    reg [15:0] c_matrix [0:M*N-1];

    // MFIU-like processing: bit mask generation and sparse indexing
    function [31:0] generate_bitmask;
        input [15:0] start_idx;
        input [4:0] count;
        integer j;
        begin
            generate_bitmask = 0;
            for (j = 0; j < count && j < 32; j = j + 1) begin
                generate_bitmask[j] = 1;
            end
        end
    endfunction

    // Index mapping for sparse operations
    function [15:0] map_sparse_index;
        input [15:0] row;
        input [15:0] col;
        begin
            map_sparse_index = row * N + col;
        end
    endfunction

    integer k, m, n;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset all pipeline stages
            stage1_valid <= 0;
            stage2_valid <= 0;
            stage3_valid <= 0;
            stage4_valid <= 0;
            stage5_valid <= 0;
            valid_out <= 0;
            cycle_count <= 0;
            
            // Initialize C matrix to zero
            for (k = 0; k < M*N; k = k + 1) begin
                c_matrix[k] <= 16'h0000;
            end
            
            // Reset queue depths
            for (k = 0; k < PE_NUM; k = k + 1) begin
                stage3_queue_depth[k] <= 0;
            end
            
        end else begin
            cycle_count <= cycle_count + 1;
            
            // Stage 5: Add tree output processing
            stage5_valid <= add_tree_valid;
            valid_out <= stage5_valid;
            
            if (add_tree_update_valid && add_tree_index < M*N) begin
                // Update C matrix with accumulated result
                c_matrix[add_tree_index] <= add_tree_value;
            end
            
            // Output C matrix (flattened)
            if (stage5_valid) begin
                for (k = 0; k < M*N; k = k + 1) begin
                    C_values[k*16 +: 16] <= c_matrix[k];
                end
            end

            // Stage 4: Multiply units processing
            stage4_valid <= stage3_valid;
            
            // Advance queues when multiply units consume data
            if (stage3_valid) begin
                for (k = 0; k < PE_NUM; k = k + 1) begin
                    if (stage3_queue_depth[k] > 0 && mul_valid_out[k]) begin
                        // Shift queue elements
                        for (m = 0; m < 31; m = m + 1) begin
                            stage3_mul_queue_a[k][m] <= stage3_mul_queue_a[k][m+1];
                            stage3_mul_queue_b[k][m] <= stage3_mul_queue_b[k][m+1];
                            stage3_sft_index_queue[k][m] <= stage3_sft_index_queue[k][m+1];
                        end
                        stage3_queue_depth[k] <= stage3_queue_depth[k] - 1;
                    end
                end
            end

            // Stage 3: Distribute to multiply unit queues
            stage3_valid <= stage2_valid;
            
            if (stage2_valid) begin
                // Round-robin distribution to PE units
                reg [15:0] distribution_counter;
                distribution_counter = 0;
                
                // Distribute A and B values to PE queues
                for (k = 0; k < K && distribution_counter < PE_NUM * 32; k = k + 1) begin
                    if (stage2_values_A[k*16 +: 16] != 0) begin
                        reg [2:0] pe_idx;
                        pe_idx = distribution_counter % PE_NUM;
                        
                        if (stage3_queue_depth[pe_idx] < 32) begin
                            stage3_mul_queue_a[pe_idx][stage3_queue_depth[pe_idx]] <= 
                                stage2_values_A[k*16 +: 16];
                            stage3_sft_index_queue[pe_idx][stage3_queue_depth[pe_idx]] <= 
                                map_sparse_index(k / N, k % N);
                            stage3_queue_depth[pe_idx] <= stage3_queue_depth[pe_idx] + 1;
                        end
                        distribution_counter = distribution_counter + 1;
                    end
                end
                
                // Similar distribution for B values
                distribution_counter = 0;
                for (k = 0; k < 16 && distribution_counter < PE_NUM * 32; k = k + 1) begin
                    if (stage2_values_B[k*32 +: 16] != 0) begin
                        reg [2:0] pe_idx;
                        pe_idx = distribution_counter % PE_NUM;
                        
                        if (stage3_queue_depth[pe_idx] < 32) begin
                            stage3_mul_queue_b[pe_idx][stage3_queue_depth[pe_idx]] <= 
                                stage2_values_B[k*32 +: 16];
                            stage3_queue_depth[pe_idx] <= stage3_queue_depth[pe_idx] + 1;
                        end
                        distribution_counter = distribution_counter + 1;
                    end
                end
            end

            // Stage 2: MFIU processing (simplified)
            stage2_valid <= stage1_valid;
            
            if (stage1_valid) begin
                // Copy values for processing
                stage2_values_A <= stage1_A_values;
                stage2_values_B <= stage1_B_values;
                
                // Generate indices based on masks and offsets
                for (k = 0; k < M; k = k + 1) begin
                    for (m = 0; m < K; m = m + 1) begin
                        if (stage1_A_masks[k][m]) begin
                            stage2_indices_A[k*K + m] <= stage1_A_offset[k] + m;
                        end else begin
                            stage2_indices_A[k*K + m] <= 16'hFFFF; // Invalid index
                        end
                    end
                end
                
                for (k = 0; k < N; k = k + 1) begin
                    for (m = 0; m < K; m = m + 1) begin
                        if (stage1_B_masks[k][m]) begin
                            stage2_indices_B[k*K + m] <= stage1_B_offset[k] + m;
                        end else begin
                            stage2_indices_B[k*K + m] <= 16'hFFFF; // Invalid index
                        end
                    end
                end
            end

            // Stage 1: Input processing and CSR conversion
            stage1_valid <= valid_in;
            
            if (valid_in) begin
                stage1_A_values <= A_values;
                stage1_B_values <= B_values;
                
                if (is_hbm) begin
                    // HBM mode: process CSR format directly
                    // Extract CSR data from inputs
                    for (k = 0; k < 16; k = k + 1) begin
                        csr_values[k] <= B_values[k*32 +: 16];
                        csr_col_indices[k] <= B_col_indices[k*16 +: 16];
                    end
                    for (k = 0; k < 17; k = k + 1) begin
                        csr_row_ptr[k] <= B_row_ptr[k*16 +: 16];
                    end
                    
                    // Generate masks and offsets from CSR
                    for (k = 0; k < M; k = k + 1) begin
                        stage1_A_masks[k] <= generate_bitmask(0, K);
                        stage1_A_offset[k] <= k * K;
                    end
                    
                    for (k = 0; k < N; k = k + 1) begin
                        reg [16:0] row_start, row_end;
                        row_start = csr_row_ptr[k];
                        row_end = csr_row_ptr[k+1];
                        stage1_B_masks[k] <= generate_bitmask(row_start, row_end - row_start);
                        stage1_B_offset[k] <= row_start;
                    end
                    
                end else begin
                    // Dense mode: process dense matrices
                    for (k = 0; k < M; k = k + 1) begin
                        stage1_A_masks[k] <= generate_bitmask(0, K);
                        stage1_A_offset[k] <= k * K;
                    end
                    
                    for (k = 0; k < N; k = k + 1) begin
                        stage1_B_masks[k] <= generate_bitmask(0, K);
                        stage1_B_offset[k] <= k * K;
                    end
                end
            end
        end
    end

endmodule 