// HBM控制器专门调试测试台
`timescale 1ns / 1ps

module hbm_debug_testbench;

// 最小参数
parameter MAX_BLOCKS = 1;
parameter BLOCK_SIZE = 4;
parameter ADDR_WIDTH = 2;
parameter BLOCK_ADDR_WIDTH = 1;
parameter ELEMENT_ADDR_WIDTH = 2;
parameter MAX_CYCLES = 1000;

reg clk, rst_n;
integer cycle_count;

// HBM控制器信号
reg config_valid;
reg [BLOCK_ADDR_WIDTH-1:0] config_block_id;
reg [15:0] config_value;
reg [ADDR_WIDTH-1:0] config_col_index;
reg [ADDR_WIDTH-1:0] config_row_index;
reg [ELEMENT_ADDR_WIDTH-1:0] config_element_index;
reg config_block_complete;
wire config_ready;

reg block_request_valid;
reg [BLOCK_ADDR_WIDTH-1:0] block_request_id;
wire block_request_ready;

wire block_data_valid;
wire [BLOCK_ADDR_WIDTH-1:0] block_data_id;
wire [15:0] block_data_value;
wire [ADDR_WIDTH-1:0] block_data_col_index;
wire [ADDR_WIDTH-1:0] block_data_row_start;
wire [ELEMENT_ADDR_WIDTH-1:0] block_data_element_index;
wire block_data_last;
reg block_data_ready;

wire [BLOCK_ADDR_WIDTH:0] total_blocks_stored;
wire hbm_busy;

// 实例化HBM控制器
hbm_controller #(
    .MAX_BLOCKS(MAX_BLOCKS),
    .BLOCK_SIZE(BLOCK_SIZE),
    .ADDR_WIDTH(ADDR_WIDTH),
    .BLOCK_ADDR_WIDTH(BLOCK_ADDR_WIDTH),
    .ELEMENT_ADDR_WIDTH(ELEMENT_ADDR_WIDTH)
) hbm_ctrl (
    .clk(clk),
    .rst_n(rst_n),
    .config_valid(config_valid),
    .config_block_id(config_block_id),
    .config_value(config_value),
    .config_col_index(config_col_index),
    .config_row_index(config_row_index),
    .config_element_index(config_element_index),
    .config_block_complete(config_block_complete),
    .config_ready(config_ready),
    .block_request_valid(block_request_valid),
    .block_request_id(block_request_id),
    .block_request_ready(block_request_ready),
    .block_data_valid(block_data_valid),
    .block_data_id(block_data_id),
    .block_data_value(block_data_value),
    .block_data_col_index(block_data_col_index),
    .block_data_row_start(block_data_row_start),
    .block_data_element_index(block_data_element_index),
    .block_data_last(block_data_last),
    .block_data_ready(block_data_ready),
    .total_blocks_stored(total_blocks_stored),
    .hbm_busy(hbm_busy)
);

// 时钟生成
always #5 clk = ~clk;

// 详细状态监控
always @(posedge clk) begin
    if (!rst_n) begin
        cycle_count <= 0;
    end else begin
        cycle_count <= cycle_count + 1;
        
        // 每个周期都输出详细状态
        $display("周期 %d:", cycle_count);
        $display("  配置状态: %d, 输出状态: %d", 
                hbm_ctrl.config_state, hbm_ctrl.output_state);
        $display("  存储块数: %d", total_blocks_stored);
        $display("  块0有效: %b, 大小: %d", 
                hbm_ctrl.block_valid[0], hbm_ctrl.block_size[0]);
        $display("  请求: valid=%b, ready=%b, id=%d", 
                block_request_valid, block_request_ready, block_request_id);
        $display("  数据: valid=%b, ready=%b, value=%h", 
                block_data_valid, block_data_ready, block_data_value);
        $display("  输出块: %d, 元素: %d, 大小: %d", 
                hbm_ctrl.current_output_block, 
                hbm_ctrl.current_output_element,
                hbm_ctrl.output_block_size);
        $display("---");
        
        if (cycle_count > MAX_CYCLES) begin
            $display("❌ HBM调试超时");
            $finish;
        end
    end
end

// 测试序列
initial begin
    $display("=== HBM控制器调试测试 ===");
    
    // 初始化
    clk = 0; rst_n = 0;
    config_valid = 0;
    config_block_id = 0;
    config_value = 0;
    config_col_index = 0;
    config_row_index = 0;
    config_element_index = 0;
    config_block_complete = 0;
    block_request_valid = 0;
    block_request_id = 0;
    block_data_ready = 1;
    
    // 复位
    #20 rst_n = 1; #20;
    
    $display("1. 配置一个矩阵块");
    // 加载一个元素到块0
    config_valid = 1;
    config_block_id = 0;
    config_value = 16'h3f80;  // 1.0 in BF16
    config_col_index = 0;
    config_row_index = 0;
    config_element_index = 0;
    config_block_complete = 1;  // 立即完成
    
    wait(config_ready);
    @(posedge clk);
    config_valid = 0;
    config_block_complete = 0;
    @(posedge clk);
    
    $display("2. 等待配置完成");
    #100;  // 等待几个周期
    
    $display("3. 请求块数据");
    block_request_valid = 1;
    block_request_id = 0;
    
    wait(block_request_ready);
    @(posedge clk);
    block_request_valid = 0;
    @(posedge clk);
    
    $display("4. 等待数据输出");
    #200;  // 等待数据输出
    
    if (block_data_valid) begin
        $display("✅ HBM数据输出成功!");
        $display("   数据值: %h", block_data_value);
        $display("   列索引: %d", block_data_col_index);
    end else begin
        $display("❌ HBM数据输出失败!");
    end
    
    #100;
    $finish;
end

endmodule 