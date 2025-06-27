// Compute Units: Multiply and Add Units for Trapezoid Pipeline
module multiply_unit (
    input wire clk,
    input wire rst_n,
    input wire input_valid,
    input wire [15:0] input1,
    input wire [15:0] input2,
    input wire [15:0] sft_index,
    output wire valid_out,
    output wire [15:0] result,
    output wire [15:0] index_out
);

    // Index queue for maintaining input-output correspondence
    reg [15:0] index_queue [0:7]; // Depth 8 for 5-stage pipeline + margin
    reg [2:0] index_queue_head, index_queue_tail;
    reg [3:0] index_queue_count;

    wire multiply_valid_out;
    wire [15:0] multiply_result;

    // BF16 multiply pipeline instance
    bf16_multiply_pipeline bf16_mul_inst (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(input_valid),
        .bf16_a(input1),
        .bf16_b(input2),
        .valid_out(multiply_valid_out),
        .result(multiply_result)
    );

    // Index queue management
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            index_queue_head <= 0;
            index_queue_tail <= 0;
            index_queue_count <= 0;
        end else begin
            // Enqueue when input is valid
            if (input_valid && index_queue_count < 8) begin
                index_queue[index_queue_tail] <= sft_index;
                index_queue_tail <= index_queue_tail + 1;
                index_queue_count <= index_queue_count + 1;
            end
            
            // Dequeue when output is valid
            if (multiply_valid_out && index_queue_count > 0) begin
                index_queue_head <= index_queue_head + 1;
                index_queue_count <= index_queue_count - 1;
            end
        end
    end

    assign valid_out = multiply_valid_out && (index_queue_count > 0);
    assign result = multiply_result;
    assign index_out = index_queue[index_queue_head];

endmodule

module add_unit (
    input wire clk,
    input wire rst_n,
    input wire input_valid,
    input wire [15:0] input1,
    input wire [15:0] input2,
    input wire [15:0] sft_index,
    output wire valid_out,
    output wire [15:0] result,
    output wire [15:0] index_out
);

    // Index queue for maintaining input-output correspondence
    reg [15:0] index_queue [0:7]; // Depth 8 for 5-stage pipeline + margin
    reg [2:0] index_queue_head, index_queue_tail;
    reg [3:0] index_queue_count;

    wire add_valid_out;
    wire [15:0] add_result;

    // BF16 add pipeline instance
    bf16_add_pipeline bf16_add_inst (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(input_valid),
        .bf16_a(input1),
        .bf16_b(input2),
        .valid_out(add_valid_out),
        .result(add_result)
    );

    // Index queue management
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            index_queue_head <= 0;
            index_queue_tail <= 0;
            index_queue_count <= 0;
        end else begin
            // Enqueue when input is valid
            if (input_valid && index_queue_count < 8) begin
                index_queue[index_queue_tail] <= sft_index;
                index_queue_tail <= index_queue_tail + 1;
                index_queue_count <= index_queue_count + 1;
            end
            
            // Dequeue when output is valid
            if (add_valid_out && index_queue_count > 0) begin
                index_queue_head <= index_queue_head + 1;
                index_queue_count <= index_queue_count - 1;
            end
        end
    end

    assign valid_out = add_valid_out && (index_queue_count > 0);
    assign result = add_result;
    assign index_out = index_queue[index_queue_head];

endmodule

// MAC Unit combining multiply and add operations
module mac_unit (
    input wire clk,
    input wire rst_n,
    input wire input_valid,
    input wire [15:0] input1,
    input wire [15:0] input2,
    input wire [15:0] initial_acc,
    input wire set_initial_acc,
    output reg valid_out,
    output reg [15:0] result
);

    // Internal signals
    wire multiply_valid;
    wire [15:0] multiply_result;
    wire [15:0] multiply_index;
    
    wire add_valid;
    wire [15:0] add_result;
    wire [15:0] add_index;

    // Accumulator queue and multiply result queue
    reg [15:0] acc_queue [0:15];
    reg [3:0] acc_queue_head, acc_queue_tail;
    reg [4:0] acc_queue_count;
    
    reg [15:0] multiply_queue [0:15];
    reg [3:0] multiply_queue_head, multiply_queue_tail;
    reg [4:0] multiply_queue_count;

    // Multiply unit instance
    multiply_unit mul_inst (
        .clk(clk),
        .rst_n(rst_n),
        .input_valid(input_valid),
        .input1(input1),
        .input2(input2),
        .sft_index(16'h0000),
        .valid_out(multiply_valid),
        .result(multiply_result),
        .index_out(multiply_index)
    );

    // Add unit instance
    add_unit add_inst (
        .clk(clk),
        .rst_n(rst_n),
        .input_valid(multiply_queue_count > 0 && acc_queue_count > 0),
        .input1(multiply_queue[multiply_queue_head]),
        .input2(acc_queue[acc_queue_head]),
        .sft_index(16'h0000),
        .valid_out(add_valid),
        .result(add_result),
        .index_out(add_index)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_queue_head <= 0;
            acc_queue_tail <= 0;
            acc_queue_count <= 0;
            multiply_queue_head <= 0;
            multiply_queue_tail <= 0;
            multiply_queue_count <= 0;
            valid_out <= 0;
            result <= 0;
        end else begin
            // Set initial accumulator value
            if (set_initial_acc && acc_queue_count < 16) begin
                acc_queue[acc_queue_tail] <= initial_acc;
                acc_queue_tail <= acc_queue_tail + 1;
                acc_queue_count <= acc_queue_count + 1;
            end

            // Handle multiply result
            if (multiply_valid && multiply_queue_count < 16) begin
                multiply_queue[multiply_queue_tail] <= multiply_result;
                multiply_queue_tail <= multiply_queue_tail + 1;
                multiply_queue_count <= multiply_queue_count + 1;
            end

            // Handle add operation
            if (multiply_queue_count > 0 && acc_queue_count > 0) begin
                // Consume from both queues
                multiply_queue_head <= multiply_queue_head + 1;
                multiply_queue_count <= multiply_queue_count - 1;
                acc_queue_head <= acc_queue_head + 1;
                acc_queue_count <= acc_queue_count - 1;
            end

            // Handle add result
            valid_out <= add_valid;
            if (add_valid) begin
                result <= add_result;
                // Add result back to accumulator queue
                if (acc_queue_count < 16) begin
                    acc_queue[acc_queue_tail] <= add_result;
                    acc_queue_tail <= acc_queue_tail + 1;
                    acc_queue_count <= acc_queue_count + 1;
                end
            end
        end
    end

endmodule 