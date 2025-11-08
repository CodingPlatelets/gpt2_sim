// Simplified Add Tree for Trapezoid Pipeline
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

    // Simplified internal signals
    integer i;

    // Stage registers for pipeline
    reg [TREE_LEVELS:0] stage_valid;
    
    // Level 0 (input level) - 4 inputs
    reg [15:0] level0_indices [0:3];
    reg [15:0] level0_values [0:3];
    reg [3:0] level0_valid_mask;
    
    // Level 1 - first reduction (4->2)
    reg [15:0] level1_indices [0:1];
    reg [15:0] level1_values [0:1];
    reg [1:0] level1_valid_mask;
    wire [1:0] level1_add_valid;
    wire [15:0] level1_add_results [0:1];
    
    // Level 2 - final reduction (2->1)
    reg [15:0] level2_index;
    reg [15:0] level2_value;
    reg level2_valid_mask;
    wire level2_add_valid;
    wire [15:0] level2_add_result;

    // Add units for tree levels
    // Level 1 adders (pairs 0,1 and 2,3)
    add_unit level1_add0 (
        .clk(clk),
        .rst_n(rst_n),
        .input_valid(stage_valid[0] && level0_valid_mask[0] && level0_valid_mask[1] &&
                     (level0_indices[0] == level0_indices[1])),
        .input1(level0_values[0]),
        .input2(level0_values[1]),
        .sft_index(level0_indices[0]),
        .valid_out(level1_add_valid[0]),
        .result(level1_add_results[0]),
        .index_out() // Not used
    );
    
    add_unit level1_add1 (
        .clk(clk),
        .rst_n(rst_n),
        .input_valid(stage_valid[0] && level0_valid_mask[2] && level0_valid_mask[3] &&
                     (level0_indices[2] == level0_indices[3])),
        .input1(level0_values[2]),
        .input2(level0_values[3]),
        .sft_index(level0_indices[2]),
        .valid_out(level1_add_valid[1]),
        .result(level1_add_results[1]),
        .index_out() // Not used
    );
    
    // Level 2 adder (final)
    add_unit level2_add (
        .clk(clk),
        .rst_n(rst_n),
        .input_valid(stage_valid[1] && level1_valid_mask[0] && level1_valid_mask[1] &&
                     (level1_indices[0] == level1_indices[1])),
        .input1(level1_values[0]),
        .input2(level1_values[1]),
        .sft_index(level1_indices[0]),
        .valid_out(level2_add_valid),
        .result(level2_add_result),
        .index_out() // Not used
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset all stages
            stage_valid <= 0;
            level0_valid_mask <= 0;
            level1_valid_mask <= 0;
            level2_valid_mask <= 0;
            valid_out <= 0;
            output_update_valid <= 0;
            output_index <= 0;
            output_value <= 0;
            
            // Reset arrays
            for (i = 0; i < 4; i = i + 1) begin
                level0_indices[i] <= 0;
                level0_values[i] <= 0;
            end
            for (i = 0; i < 2; i = i + 1) begin
                level1_indices[i] <= 0;
                level1_values[i] <= 0;
            end
            level2_index <= 0;
            level2_value <= 0;
            
        end else begin
            // Pipeline progression
            stage_valid[2] <= stage_valid[1];
            stage_valid[1] <= stage_valid[0];
            stage_valid[0] <= valid_in;
            
            // Output stage
            valid_out <= stage_valid[2];
            
            // Level 0: Input processing
            if (valid_in) begin
                // Load inputs from packed arrays
                level0_indices[0] <= input_indices[15:0];
                level0_indices[1] <= input_indices[31:16];
                level0_indices[2] <= input_indices[47:32];
                level0_indices[3] <= input_indices[63:48];
                
                level0_values[0] <= input_values[15:0];
                level0_values[1] <= input_values[31:16];
                level0_values[2] <= input_values[47:32];
                level0_values[3] <= input_values[63:48];
                
                level0_valid_mask <= input_valid_mask;
            end
            
            // Level 1: First reduction
            if (stage_valid[0]) begin
                // Process pair 0,1
                if (level0_valid_mask[0] && level0_valid_mask[1] &&
                    (level0_indices[0] == level0_indices[1])) begin
                    // Same index, will be added by add unit
                    level1_valid_mask[0] <= 1;
                    level1_indices[0] <= level0_indices[0];
                    // Value will come from adder
                end else if (level0_valid_mask[0] && !level0_valid_mask[1]) begin
                    // Only first input valid
                    level1_valid_mask[0] <= 1;
                    level1_indices[0] <= level0_indices[0];
                    level1_values[0] <= level0_values[0];
                end else if (!level0_valid_mask[0] && level0_valid_mask[1]) begin
                    // Only second input valid
                    level1_valid_mask[0] <= 1;
                    level1_indices[0] <= level0_indices[1];
                    level1_values[0] <= level0_values[1];
                end else begin
                    level1_valid_mask[0] <= 0;
                end
                
                // Process pair 2,3
                if (level0_valid_mask[2] && level0_valid_mask[3] &&
                    (level0_indices[2] == level0_indices[3])) begin
                    level1_valid_mask[1] <= 1;
                    level1_indices[1] <= level0_indices[2];
                end else if (level0_valid_mask[2] && !level0_valid_mask[3]) begin
                    level1_valid_mask[1] <= 1;
                    level1_indices[1] <= level0_indices[2];
                    level1_values[1] <= level0_values[2];
                end else if (!level0_valid_mask[2] && level0_valid_mask[3]) begin
                    level1_valid_mask[1] <= 1;
                    level1_indices[1] <= level0_indices[3];
                    level1_values[1] <= level0_values[3];
                end else begin
                    level1_valid_mask[1] <= 0;
                end
            end
            
            // Update level 1 values from adders
            if (level1_add_valid[0]) begin
                level1_values[0] <= level1_add_results[0];
            end
            if (level1_add_valid[1]) begin
                level1_values[1] <= level1_add_results[1];
            end
            
            // Level 2: Final reduction
            if (stage_valid[1]) begin
                if (level1_valid_mask[0] && level1_valid_mask[1] &&
                    (level1_indices[0] == level1_indices[1])) begin
                    level2_valid_mask <= 1;
                    level2_index <= level1_indices[0];
                end else if (level1_valid_mask[0] && !level1_valid_mask[1]) begin
                    level2_valid_mask <= 1;
                    level2_index <= level1_indices[0];
                    level2_value <= level1_values[0];
                end else if (!level1_valid_mask[0] && level1_valid_mask[1]) begin
                    level2_valid_mask <= 1;
                    level2_index <= level1_indices[1];
                    level2_value <= level1_values[1];
                end else begin
                    level2_valid_mask <= 0;
                end
            end
            
            // Update level 2 value from adder
            if (level2_add_valid) begin
                level2_value <= level2_add_result;
            end
            
            // Output
            output_update_valid <= 0;
            if (stage_valid[2] && level2_valid_mask) begin
                output_index <= level2_index;
                output_value <= level2_value;
                output_update_valid <= 1;
            end
        end
    end

endmodule 