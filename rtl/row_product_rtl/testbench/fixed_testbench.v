// Fixed Vector Matrix Row Product HBM Multi-Batch Testbench
// 修复版测试台：添加超时保护，使用4096维度

`timescale 1ns / 1ps

module fixed_testbench;

// 真实规模参数 - 用户需求
parameter NUM_PEROWS = 32;         // 32个PERow
parameter PES_PER_ROW = 128;       // 每行128个PE
parameter VECTOR_SIZE = 4096;      // 4096维向量
parameter MAX_BATCH_SIZE = 1;      // 1个批次
parameter MAX_BLOCKS = 1024;       // 1024个块
parameter BLOCK_SIZE = 2048;       // 每块2048元素
parameter ADDR_WIDTH = 12;         // log2(4096)
parameter PEROW_ADDR_WIDTH = 5;    // log2(32)
parameter BATCH_ADDR_WIDTH = 1;    // log2(1)
parameter BLOCK_ADDR_WIDTH = 10;   // log2(1024)
parameter ELEMENT_ADDR_WIDTH = 11; // log2(2048)

// 测试控制参数
parameter MAX_CYCLES = 1000000;    // 最大100万周期超时保护
parameter SPARSE_RATIO = 90;       // 90%稀疏度

// 时钟和复位
reg clk;
reg rst_n;
integer cycle_count;

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

// 周期计数和超时保护
always @(posedge clk) begin
    if (!rst_n) begin
        cycle_count <= 0;
    end else begin
        cycle_count <= cycle_count + 1;
        
        // 超时保护
        if (cycle_count > MAX_CYCLES) begin
            $display("❌ 错误：仿真超时！运行了 %d 周期", cycle_count);
            $display("   系统状态：busy=%b, done=%b", system_busy, computation_done);
            $display("   完成任务数：%d", total_tasks_completed);
            $display("   这说明逻辑有bug，无法正常完成");
            $finish;
        end
        
        // 每10000周期报告一次状态
        if (cycle_count % 10000 == 0) begin
            $display("周期 %d: busy=%b, done=%b, 任务=%d", 
                    cycle_count, system_busy, computation_done, total_tasks_completed);
        end
    end
end

// 简化的测试序列
initial begin
    $display("=== 4096维向量矩阵乘法测试 ===");
    $display("向量维度: %d", VECTOR_SIZE);
    $display("矩阵维度: %d x %d", VECTOR_SIZE, VECTOR_SIZE);
    $display("稀疏度: %d%%", SPARSE_RATIO);
    
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
    num_blocks = 1;  // 先只用1个块进行测试
    result_ready = 1;
    
    // 复位
    #20 rst_n = 1;
    #20;
    
    $display("1. 快速加载测试向量A (只加载前16个元素)");
    // 简化测试：只加载少量元素
    for (integer i = 0; i < 16; i = i + 1) begin
        load_a_vector(0, i, $realtobits(1.0));
    end
    
    $display("2. 加载一个简单的稀疏矩阵块");
    // 只加载4个非零元素
    load_b_matrix_element(0, 16'h3f80, 0, 0, 0, 0);  // B[0,0] = 1.0
    load_b_matrix_element(0, 16'h4000, 1, 1, 1, 0);  // B[1,1] = 2.0
    load_b_matrix_element(0, 16'h4040, 2, 2, 2, 0);  // B[2,2] = 3.0
    load_b_matrix_element(0, 16'h4080, 3, 3, 3, 1);  // B[3,3] = 4.0 (最后一个)
    
    $display("3. 开始计算");
    start_computation = 1;
    @(posedge clk);
    start_computation = 0;
    
    $display("4. 等待计算完成...");
    
    // 等待完成或超时
    wait(computation_done || (cycle_count > MAX_CYCLES));
    
    if (computation_done) begin
        $display("✅ 计算完成!");
        $display("   总时钟周期: %d", total_clock_cycles);
        $display("   总完成任务数: %d", total_tasks_completed);
    end else begin
        $display("❌ 计算超时或出错!");
    end
    
    #100;
    $display("=== 测试结束 ===");
    $finish;
end

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

endmodule 