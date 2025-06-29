// 4096维真实规模测试台 - GPT2向量矩阵乘法
`timescale 1ns / 1ps

module vector_matrix_testbench_4096;

// 真实规模参数 - 4096维GPT2应用
parameter NUM_PEROWS = 32;         // 32个PERow
parameter PES_PER_ROW = 128;       // 每行128个PE
parameter VECTOR_SIZE = 4096;      // 4096维向量
parameter MAX_BATCH_SIZE = 64;     // 最大64批次
parameter MAX_BLOCKS = 1024;       // 最大1024个块
parameter BLOCK_SIZE = 2048;       // 每块2048个元素
parameter ADDR_WIDTH = 12;         // log2(4096)
parameter PEROW_ADDR_WIDTH = 5;    // log2(32)
parameter BATCH_ADDR_WIDTH = 6;    // log2(64)
parameter BLOCK_ADDR_WIDTH = 10;   // log2(1024)
parameter ELEMENT_ADDR_WIDTH = 11; // log2(2048)

// 测试控制
parameter MAX_CYCLES = 1000000;    // 100万周期超时（真实应用需要更多时间）
parameter SPARSITY_PERCENT = 90;   // 90%稀疏度

reg clk, rst_n;
integer cycle_count;

// 系统信号
reg start_computation;
reg [BATCH_ADDR_WIDTH:0] batch_size;
reg [BLOCK_ADDR_WIDTH:0] num_blocks;
wire computation_done;
wire system_busy;
wire [31:0] total_tasks_completed;

// A向量配置接口
reg a_vector_config_valid;
reg [BATCH_ADDR_WIDTH-1:0] a_vector_batch_id;
reg [ADDR_WIDTH-1:0] a_vector_addr;
reg [31:0] a_vector_data;
wire a_vector_config_ready;

// B矩阵配置接口
reg b_matrix_config_valid;
reg [BLOCK_ADDR_WIDTH-1:0] b_matrix_block_id;
reg [15:0] b_matrix_value;
reg [ADDR_WIDTH-1:0] b_matrix_col_index;
reg [ADDR_WIDTH-1:0] b_matrix_row_index;
reg [ELEMENT_ADDR_WIDTH-1:0] b_matrix_element_index;
reg b_matrix_block_complete;
wire b_matrix_config_ready;

// 结果输出接口
wire [31:0] total_clock_cycles;
wire result_valid;
wire [BATCH_ADDR_WIDTH-1:0] result_batch_id;
wire [ADDR_WIDTH-1:0] result_addr;
wire [31:0] result_data;
reg result_ready;

// 循环变量声明（标准Verilog要求在模块级声明）
integer elements_in_block;
integer random_row;
integer random_col;

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

// 时钟生成 - 100MHz
always #5 clk = ~clk;

// 性能监控
always @(posedge clk) begin
    if (!rst_n) begin
        cycle_count <= 0;
    end else begin
        cycle_count <= cycle_count + 1;
        
        // 每1000周期输出一次状态
        if (cycle_count % 1000 == 0) begin
            $display("周期 %d: busy=%b, done=%b, 任务=%d, 状态=%d", 
                    cycle_count, system_busy, computation_done, 
                    total_tasks_completed, dut.main_state);
        end
        
        // 超时保护
        if (cycle_count > MAX_CYCLES) begin
            $display("❌ 4096维测试：仿真超时在 %d 周期", cycle_count);
            $display("   总任务完成: %d", total_tasks_completed);
            $display("   主状态: %d", dut.main_state);
            $display("   当前块ID: %d / %d", dut.current_block_id, num_blocks);
            $finish;
        end
    end
end

// 随机数生成器（用于稀疏矩阵）
integer seed = 12345;
function [15:0] random_bf16_value;
    input integer dummy; // Verilog要求函数至少有一个输入
    begin
        // 生成随机BF16值（简化版本）
        random_bf16_value = ($random(seed) % 1000) + 16'h3f80; // 基于1.0的随机值
    end
endfunction

// 4096维测试序列
initial begin
    $display("=== 4096维GPT2向量矩阵乘法测试 ===");
    $display("配置: %d PERows × %d PEs = %d 并行单元", NUM_PEROWS, PES_PER_ROW, NUM_PEROWS * PES_PER_ROW);
    $display("向量规模: %d维", VECTOR_SIZE);
    $display("稀疏度: %d%%", SPARSITY_PERCENT);
    
    // 初始化
    clk = 0; rst_n = 0;
    a_vector_config_valid = 0;
    b_matrix_config_valid = 0;
    start_computation = 0;
    batch_size = 1;
    num_blocks = 100;  // 使用100个块进行测试
    result_ready = 1;
    
    // 复位
    #20 rst_n = 1; #50;
    
    $display("1. 加载4096维向量A (全1向量)...");
    for (integer i = 0; i < VECTOR_SIZE; i = i + 1) begin
        a_vector_config_valid = 1;
        a_vector_batch_id = 0;
        a_vector_addr = i;
        a_vector_data = $realtobits(1.0);
        @(posedge clk);
        if (i % 1000 == 999) begin
            $display("   已加载 %d / %d 向量元素", i+1, VECTOR_SIZE);
        end
    end
    a_vector_config_valid = 0;
    @(posedge clk);
    
    $display("2. 加载稀疏矩阵块 (90%稀疏度)...");
    for (integer block_id = 0; block_id < num_blocks; block_id = block_id + 1) begin
        elements_in_block = 0;
        
        // 每个块大约有10%的非零元素
        for (integer elem = 0; elem < BLOCK_SIZE / 10; elem = elem + 1) begin
            // 随机选择行列位置
            random_row = ($random(seed) % VECTOR_SIZE);
            random_col = ($random(seed) % VECTOR_SIZE);
            
            b_matrix_config_valid = 1;
            b_matrix_block_id = block_id;
            b_matrix_value = random_bf16_value(0);
            b_matrix_col_index = random_col;
            b_matrix_row_index = random_row;
            b_matrix_element_index = elements_in_block;
            b_matrix_block_complete = (elem == (BLOCK_SIZE / 10 - 1));
            
            @(posedge clk);
            elements_in_block = elements_in_block + 1;
        end
        
        b_matrix_config_valid = 0;
        b_matrix_block_complete = 0;
        @(posedge clk);
        
        if ((block_id + 1) % 10 == 0) begin
            $display("   已加载 %d / %d 矩阵块", block_id + 1, num_blocks);
        end
    end
    
    $display("3. 开始4096维向量矩阵乘法计算...");
    start_computation = 1;
    @(posedge clk);
    start_computation = 0;
    
    $display("4. 等待计算完成...");
    $display("   预计需要数万到数十万周期");
    $display("   使用Ctrl+C可中断测试");
    
    // 等待完成
    wait(computation_done || (cycle_count > MAX_CYCLES));
    
    if (computation_done) begin
        $display("🎉 4096维计算完成!");
        $display("   总周期数: %d", cycle_count);
        $display("   总任务数: %d", total_tasks_completed);
        $display("   平均性能: %.2f MAC运算/周期", $itor(total_tasks_completed) / $itor(cycle_count));
        $display("   系统吞吐量: %.2f GOPS @ 100MHz", 
                ($itor(total_tasks_completed) / $itor(cycle_count)) * 100.0 / 1000.0);
    end
    
    #1000;
    $finish;
end

endmodule 