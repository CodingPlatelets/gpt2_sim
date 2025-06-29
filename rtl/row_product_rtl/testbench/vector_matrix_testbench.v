// Vector Matrix Row Product HBM Multi-Batch Testbench
// 测试顶层向量矩阵乘法器模块

`timescale 1ns / 1ps

module vector_matrix_testbench;

// 参数定义
parameter NUM_PEROWS = 4;          // 减少规模便于测试
parameter PES_PER_ROW = 8;         // 减少规模便于测试
parameter VECTOR_SIZE = 64;        // 减少规模便于测试
parameter MAX_BATCH_SIZE = 4;      // 减少规模便于测试
parameter MAX_BLOCKS = 16;         // 减少规模便于测试
parameter BLOCK_SIZE = 32;         // 减少规模便于测试
parameter ADDR_WIDTH = 6;          // log2(64)
parameter PEROW_ADDR_WIDTH = 2;    // log2(4)
parameter BATCH_ADDR_WIDTH = 2;    // log2(4)
parameter BLOCK_ADDR_WIDTH = 4;    // log2(16)
parameter ELEMENT_ADDR_WIDTH = 5;  // log2(32)

// 时钟和复位
reg clk;
reg rst_n;

// 被测试模块的信号
reg a_vector_config_valid;
reg [BATCH_ADDR_WIDTH-1:0] a_vector_batch_id;
reg [ADDR_WIDTH-1:0] a_vector_addr;
reg [31:0] a_vector_data;
wire a_vector_config_ready;

reg b_matrix_config_valid;
reg [BLOCK_ADDR_WIDTH-1:0] b_matrix_block_id;
reg [15:0] b_matrix_value;
reg [ADDR_WIDTH-1:0] b_matrix_col_index;
reg [ADDR_WIDTH-1:0] b_matrix_row_index;
reg [ELEMENT_ADDR_WIDTH-1:0] b_matrix_element_index;
reg b_matrix_block_complete;
wire b_matrix_config_ready;

reg start_computation;
reg [BATCH_ADDR_WIDTH:0] batch_size;
reg [BLOCK_ADDR_WIDTH:0] num_blocks;
wire computation_done;
wire [31:0] total_clock_cycles;

wire result_valid;
wire [BATCH_ADDR_WIDTH-1:0] result_batch_id;
wire [ADDR_WIDTH-1:0] result_addr;
wire [31:0] result_data;
reg result_ready;

wire system_busy;
wire [31:0] total_tasks_completed;

// 实例化被测试模块
vector_matrix_row_product_hbm_multi_batch #(
    .NUM_PEROWS(NUM_PEROWS),
    .PES_PER_ROW(PES_PER_ROW),
    .VECTOR_SIZE(VECTOR_SIZE),
    .MAX_BATCH_SIZE(MAX_BATCH_SIZE),
    .MAX_BLOCKS(MAX_BLOCKS),
    .BLOCK_SIZE(BLOCK_SIZE),
    .ADDR_WIDTH(ADDR_WIDTH),
    .PEROW_ADDR_WIDTH(PEROW_ADDR_WIDTH),
    .BATCH_ADDR_WIDTH(BATCH_ADDR_WIDTH),
    .BLOCK_ADDR_WIDTH(BLOCK_ADDR_WIDTH),
    .ELEMENT_ADDR_WIDTH(ELEMENT_ADDR_WIDTH)
) dut (
    .clk(clk),
    .rst_n(rst_n),
    .a_vector_config_valid(a_vector_config_valid),
    .a_vector_batch_id(a_vector_batch_id),
    .a_vector_addr(a_vector_addr),
    .a_vector_data(a_vector_data),
    .a_vector_config_ready(a_vector_config_ready),
    .b_matrix_config_valid(b_matrix_config_valid),
    .b_matrix_block_id(b_matrix_block_id),
    .b_matrix_value(b_matrix_value),
    .b_matrix_col_index(b_matrix_col_index),
    .b_matrix_row_index(b_matrix_row_index),
    .b_matrix_element_index(b_matrix_element_index),
    .b_matrix_block_complete(b_matrix_block_complete),
    .b_matrix_config_ready(b_matrix_config_ready),
    .start_computation(start_computation),
    .batch_size(batch_size),
    .num_blocks(num_blocks),
    .computation_done(computation_done),
    .total_clock_cycles(total_clock_cycles),
    .result_valid(result_valid),
    .result_batch_id(result_batch_id),
    .result_addr(result_addr),
    .result_data(result_data),
    .result_ready(result_ready),
    .system_busy(system_busy),
    .total_tasks_completed(total_tasks_completed)
);

// 时钟生成
always #5 clk = ~clk;

// 测试任务
task load_a_vector;
    input [BATCH_ADDR_WIDTH-1:0] batch_id;
    input [ADDR_WIDTH-1:0] addr;
    input [31:0] data;
    begin
        a_vector_config_valid = 1'b1;
        a_vector_batch_id = batch_id;
        a_vector_addr = addr;
        a_vector_data = data;
        
        wait(a_vector_config_ready);
        @(posedge clk);
        a_vector_config_valid = 1'b0;
        @(posedge clk);
    end
endtask

task load_b_matrix_element;
    input [BLOCK_ADDR_WIDTH-1:0] block_id;
    input [15:0] value;
    input [ADDR_WIDTH-1:0] col_index;
    input [ADDR_WIDTH-1:0] row_index;
    input [ELEMENT_ADDR_WIDTH-1:0] element_index;
    input block_complete;
    begin
        b_matrix_config_valid = 1'b1;
        b_matrix_block_id = block_id;
        b_matrix_value = value;
        b_matrix_col_index = col_index;
        b_matrix_row_index = row_index;
        b_matrix_element_index = element_index;
        b_matrix_block_complete = block_complete;
        
        wait(b_matrix_config_ready);
        @(posedge clk);
        b_matrix_config_valid = 1'b0;
        b_matrix_block_complete = 1'b0;
        @(posedge clk);
    end
endtask

// 测试序列
initial begin
    $display("=== 向量矩阵乘法器测试开始 ===");
    
    // 初始化信号
    clk = 0;
    rst_n = 0;
    a_vector_config_valid = 0;
    a_vector_batch_id = 0;
    a_vector_addr = 0;
    a_vector_data = 0;
    b_matrix_config_valid = 0;
    b_matrix_block_id = 0;
    b_matrix_value = 0;
    b_matrix_col_index = 0;
    b_matrix_row_index = 0;
    b_matrix_element_index = 0;
    b_matrix_block_complete = 0;
    start_computation = 0;
    batch_size = 1;
    num_blocks = 1;
    result_ready = 1;
    
    // 复位
    #20 rst_n = 1;
    #20;
    
    $display("1. 加载输入向量A (批次0)");
    // 加载简单的测试向量A: [1.0, 2.0, 3.0, 4.0, ...]
    for (integer i = 0; i < 8; i = i + 1) begin
        load_a_vector(0, i, $realtobits(1.0 * (i + 1)));
    end
    
    $display("2. 加载稀疏矩阵B (块0)");
    // 加载简单的稀疏矩阵B块
    // 行0: 位置[0,1] = 1.0, 位置[0,2] = 2.0
    // 行1: 位置[1,0] = 3.0, 位置[1,3] = 4.0
    load_b_matrix_element(0, 16'h3f80, 1, 0, 0, 0);  // B[0,1] = 1.0 (BF16: 3f80)
    load_b_matrix_element(0, 16'h4000, 2, 0, 1, 0);  // B[0,2] = 2.0 (BF16: 4000)
    load_b_matrix_element(0, 16'h4040, 0, 1, 2, 0);  // B[1,0] = 3.0 (BF16: 4040)
    load_b_matrix_element(0, 16'h4080, 3, 1, 3, 1);  // B[1,3] = 4.0 (BF16: 4080), 最后一个元素
    
    $display("3. 开始计算");
    start_computation = 1;
    @(posedge clk);
    start_computation = 0;
    
    // 等待计算完成
    $display("4. 等待计算完成...");
    wait(computation_done);
    
    $display("5. 计算完成!");
    $display("   总时钟周期: %d", total_clock_cycles);
    $display("   总完成任务数: %d", total_tasks_completed);
    
    // 等待一些时钟周期
    #100;
    
    $display("=== 测试完成 ===");
    $finish;
end

// 监控信号
initial begin
    $monitor("时间=%t, 状态=busy:%b, 完成:%b, 周期:%d, 任务:%d", 
             $time, system_busy, computation_done, total_clock_cycles, total_tasks_completed);
end

// 生成VCD文件用于波形查看
initial begin
    $dumpfile("vector_matrix_test.vcd");
    $dumpvars(0, vector_matrix_testbench);
end

endmodule 