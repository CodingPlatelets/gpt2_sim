// Add Tree for Trapezoid Pipeline
// Tree structure for parallel addition with multiple PEs
module add_tree #(
    parameter PE_NUM = 4,
    parameter M = 8,
    parameter N = 128,
    parameter TREE_LEVELS = 2  // log2(PE_NUM)
)(
    input wire clk,
    input wire rst_n,
    input wire valid_in,
    input wire [PE_NUM*32-1:0] input_map_queue, // Flattened input: [index][value] pairs
    input wire [PE_NUM*16-1:0] input_indices,
    input wire [PE_NUM*16-1:0] input_values,
    input wire [PE_NUM-1:0] input_valid_mask,
    output reg valid_out,
    output reg [15:0] output_index,
    output reg [15:0] output_value,
    output reg output_update_valid
);

    // Internal tree structure
    // Level 0: PE_NUM inputs -> PE_NUM/2 outputs
    // Level 1: PE_NUM/2 inputs -> PE_NUM/4 outputs
    // etc.
    
    genvar i, j;
    integer k;

    // Stage registers for each tree level
    reg [TREE_LEVELS:0] stage_valid;
    
    // Level 0 (input level) - direct from PEs
    reg [15:0] level0_indices [0:PE_NUM-1];
    reg [15:0] level0_values [0:PE_NUM-1];
    reg [PE_NUM-1:0] level0_valid_mask;
    
    // Level 1 - first reduction
    reg [15:0] level1_indices [0:PE_NUM/2-1];
    reg [15:0] level1_values [0:PE_NUM/2-1];
    reg [PE_NUM/2-1:0] level1_valid_mask;
    wire [PE_NUM/2-1:0] level1_add_valid;
    wire [15:0] level1_add_results [0:PE_NUM/2-1];
    
    // Level 2 - second reduction (if PE_NUM > 2)
    generate
        if (PE_NUM > 2) begin : level2_gen
            reg [15:0] level2_indices [0:PE_NUM/4-1];
            reg [15:0] level2_values [0:PE_NUM/4-1];
            reg [PE_NUM/4-1:0] level2_valid_mask;
            wire [PE_NUM/4-1:0] level2_add_valid;
            wire [15:0] level2_add_results [0:PE_NUM/4-1];
        end
    endgenerate

    // Add units for each tree level
    generate
        for (i = 0; i < PE_NUM/2; i = i + 1) begin : level1_adders
            add_unit level1_add_inst (
                .clk(clk),
                .rst_n(rst_n),
                .input_valid(stage_valid[0] && level0_valid_mask[2*i] && level0_valid_mask[2*i+1] &&
                             (level0_indices[2*i] == level0_indices[2*i+1])),
                .input1(level0_values[2*i]),
                .input2(level0_values[2*i+1]),
                .sft_index(level0_indices[2*i]),
                .valid_out(level1_add_valid[i]),
                .result(level1_add_results[i]),
                .index_out() // Not used in tree
            );
        end
        
        if (PE_NUM > 2) begin : level2_adders_gen
            for (i = 0; i < PE_NUM/4; i = i + 1) begin : level2_adders
                add_unit level2_add_inst (
                    .clk(clk),
                    .rst_n(rst_n),
                    .input_valid(stage_valid[1] && level1_valid_mask[2*i] && level1_valid_mask[2*i+1] &&
                                 (level1_indices[2*i] == level1_indices[2*i+1])),
                    .input1(level1_values[2*i]),
                    .input2(level1_values[2*i+1]),
                    .sft_index(level1_indices[2*i]),
                    .valid_out(level2_gen.level2_add_valid[i]),
                    .result(level2_gen.level2_add_results[i]),
                    .index_out() // Not used in tree
                );
            end
        end
    endgenerate

    // Evict index detection - find indices that appear only once
    reg [PE_NUM-1:0] evict_mask;
    reg [15:0] evict_indices [0:PE_NUM-1];
    reg [15:0] evict_values [0:PE_NUM-1];
    reg [4:0] evict_count;

    // C values memory for accumulation
    reg [15:0] c_values [0:M*N-1];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset all stages
            for (k = 0; k <= TREE_LEVELS; k = k + 1) begin
                stage_valid[k] <= 0;
            end
            
            // Reset level valid masks
            level0_valid_mask <= 0;
            level1_valid_mask <= 0;
            if (PE_NUM > 2) begin
                level2_gen.level2_valid_mask <= 0;
            end
            
            valid_out <= 0;
            output_update_valid <= 0;
            evict_count <= 0;
            
            // Initialize C values to zero
            for (k = 0; k < M*N; k = k + 1) begin
                c_values[k] <= 16'h0000;
            end
            
        end else begin
            // Pipeline stages
            
            // Stage progression
            for (k = TREE_LEVELS; k > 0; k = k - 1) begin
                stage_valid[k] <= stage_valid[k-1];
            end
            stage_valid[0] <= valid_in;
            
            // Output stage
            valid_out <= stage_valid[TREE_LEVELS];
            
            // Level 0: Input processing
            if (valid_in) begin
                // Load inputs
                for (k = 0; k < PE_NUM; k = k + 1) begin
                    level0_indices[k] <= input_indices[k*16 +: 16];
                    level0_values[k] <= input_values[k*16 +: 16];
                end
                level0_valid_mask <= input_valid_mask;
                
                // Detect evict indices (single occurrences)
                evict_count <= 0;
                for (k = 0; k < PE_NUM; k = k + 1) begin
                    evict_mask[k] <= 0;
                    if (input_valid_mask[k]) begin
                        // Check if this index appears only once
                        reg is_single;
                        integer m;
                        is_single = 1;
                        for (m = 0; m < PE_NUM; m = m + 1) begin
                            if (m != k && input_valid_mask[m] && 
                                input_indices[k*16 +: 16] == input_indices[m*16 +: 16]) begin
                                is_single = 0;
                            end
                        end
                        if (is_single) begin
                            evict_mask[k] <= 1;
                            evict_indices[evict_count] <= input_indices[k*16 +: 16];
                            evict_values[evict_count] <= input_values[k*16 +: 16];
                            evict_count <= evict_count + 1;
                        end
                    end
                end
            end
            
            // Level 1: First reduction
            if (stage_valid[0]) begin
                for (k = 0; k < PE_NUM/2; k = k + 1) begin
                    if (level0_valid_mask[2*k] && level0_valid_mask[2*k+1] &&
                        (level0_indices[2*k] == level0_indices[2*k+1])) begin
                        // Same index, will be added
                        level1_valid_mask[k] <= 1;
                        level1_indices[k] <= level0_indices[2*k];
                        // Value will come from adder
                    end else if (level0_valid_mask[2*k] && !level0_valid_mask[2*k+1]) begin
                        // Only first input valid
                        level1_valid_mask[k] <= 1;
                        level1_indices[k] <= level0_indices[2*k];
                        level1_values[k] <= level0_values[2*k];
                    end else if (!level0_valid_mask[2*k] && level0_valid_mask[2*k+1]) begin
                        // Only second input valid
                        level1_valid_mask[k] <= 1;
                        level1_indices[k] <= level0_indices[2*k+1];
                        level1_values[k] <= level0_values[2*k+1];
                    end else begin
                        level1_valid_mask[k] <= 0;
                    end
                end
            end
            
            // Update level 1 values from adders
            for (k = 0; k < PE_NUM/2; k = k + 1) begin
                if (level1_add_valid[k]) begin
                    level1_values[k] <= level1_add_results[k];
                end
            end
            
            // Level 2 and beyond (if needed)
            if (PE_NUM > 2 && stage_valid[1]) begin
                for (k = 0; k < PE_NUM/4; k = k + 1) begin
                    if (level1_valid_mask[2*k] && level1_valid_mask[2*k+1] &&
                        (level1_indices[2*k] == level1_indices[2*k+1])) begin
                        level2_gen.level2_valid_mask[k] <= 1;
                        level2_gen.level2_indices[k] <= level1_indices[2*k];
                    end else if (level1_valid_mask[2*k] && !level1_valid_mask[2*k+1]) begin
                        level2_gen.level2_valid_mask[k] <= 1;
                        level2_gen.level2_indices[k] <= level1_indices[2*k];
                        level2_gen.level2_values[k] <= level1_values[2*k];
                    end else if (!level1_valid_mask[2*k] && level1_valid_mask[2*k+1]) begin
                        level2_gen.level2_valid_mask[k] <= 1;
                        level2_gen.level2_indices[k] <= level1_indices[2*k+1];
                        level2_gen.level2_values[k] <= level1_values[2*k+1];
                    end else begin
                        level2_gen.level2_valid_mask[k] <= 0;
                    end
                end
                
                // Update level 2 values from adders
                for (k = 0; k < PE_NUM/4; k = k + 1) begin
                    if (level2_gen.level2_add_valid[k]) begin
                        level2_gen.level2_values[k] <= level2_gen.level2_add_results[k];
                    end
                end
            end
            
            // Handle evicted values (accumulate to C memory)
            output_update_valid <= 0;
            if (stage_valid[0] && evict_count > 0) begin
                // Process first evicted value
                reg [15:0] evict_idx;
                reg [7:0] m_idx, n_idx;
                evict_idx = evict_indices[0];
                if (evict_idx != 16'hFFFF && evict_idx < M*N) begin  // Valid index
                    m_idx = evict_idx % M;
                    n_idx = evict_idx / M;
                    // Accumulate using BF16 addition
                    // For simplicity, direct assignment here (should use BF16 add)
                    c_values[m_idx * N + n_idx] <= evict_values[0];
                    output_update_valid <= 1;
                    output_index <= evict_idx;
                    output_value <= evict_values[0];
                end
            end
            
            // Final output processing
            if (stage_valid[TREE_LEVELS]) begin
                // Output the final result from top of tree
                if (PE_NUM == 2) begin
                    if (level1_valid_mask[0]) begin
                        output_index <= level1_indices[0];
                        output_value <= level1_values[0];
                    end
                end else if (PE_NUM == 4) begin
                    if (level2_gen.level2_valid_mask[0]) begin
                        output_index <= level2_gen.level2_indices[0];
                        output_value <= level2_gen.level2_values[0];
                    end
                end
            end
        end
    end

endmodule 