// Fixed HBM Controller Module
// 修复了配置和输出逻辑问题

module hbm_controller_fixed #(
    parameter MAX_BLOCKS = 1024,
    parameter BLOCK_SIZE = 2048,
    parameter ADDR_WIDTH = 12,
    parameter BLOCK_ADDR_WIDTH = 10,
    parameter ELEMENT_ADDR_WIDTH = 11,
    parameter MAX_ROWS_PER_BLOCK = 256,
    parameter ROW_PTR_WIDTH = 8
)(
    input wire clk,
    input wire rst_n,
    
    // 配置接口
    input wire config_valid,
    input wire [BLOCK_ADDR_WIDTH-1:0] config_block_id,
    input wire [15:0] config_value,
    input wire [ADDR_WIDTH-1:0] config_col_index,
    input wire [ADDR_WIDTH-1:0] config_row_index,
    input wire [ELEMENT_ADDR_WIDTH-1:0] config_element_index,
    input wire config_block_complete,
    output wire config_ready,
    
    // 块请求接口
    input wire block_request_valid,
    input wire [BLOCK_ADDR_WIDTH-1:0] block_request_id,
    output wire block_request_ready,
    
    // 块数据输出接口
    output wire block_data_valid,
    output wire [BLOCK_ADDR_WIDTH-1:0] block_data_id,
    output wire [15:0] block_data_value,
    output wire [ADDR_WIDTH-1:0] block_data_col_index,
    output wire [ADDR_WIDTH-1:0] block_data_row_start,
    output wire [ELEMENT_ADDR_WIDTH-1:0] block_data_element_index,
    output wire block_data_last,
    input wire block_data_ready,
    
    // 状态输出
    output wire [BLOCK_ADDR_WIDTH:0] total_blocks_stored,
    output wire hbm_busy
);

// 块存储结构
reg [15:0] block_values [0:MAX_BLOCKS-1][0:BLOCK_SIZE-1];
reg [ADDR_WIDTH-1:0] block_col_indices [0:MAX_BLOCKS-1][0:BLOCK_SIZE-1];
reg [ADDR_WIDTH-1:0] block_row_start [0:MAX_BLOCKS-1];
reg [ELEMENT_ADDR_WIDTH:0] block_size [0:MAX_BLOCKS-1];
reg block_valid [0:MAX_BLOCKS-1];

// 配置状态机
parameter [2:0] CONFIG_IDLE = 3'b000,
                CONFIG_LOADING = 3'b001,
                CONFIG_COMPLETE = 3'b010;

reg [2:0] config_state;
reg [BLOCK_ADDR_WIDTH:0] num_blocks_stored;

// 输出状态机
parameter [2:0] OUTPUT_IDLE = 3'b000,
                OUTPUT_READING = 3'b001,
                OUTPUT_SENDING = 3'b010;

reg [2:0] output_state;
reg [BLOCK_ADDR_WIDTH-1:0] current_output_block;
reg [ELEMENT_ADDR_WIDTH:0] current_output_element;
reg [ELEMENT_ADDR_WIDTH:0] output_block_size;

integer i;

// 修复后的配置状态机逻辑
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        config_state <= CONFIG_IDLE;
        num_blocks_stored <= 0;
        
        // 初始化所有块
        for (i = 0; i < MAX_BLOCKS; i = i + 1) begin
            block_valid[i] <= 1'b0;
            block_size[i] <= 0;
            block_row_start[i] <= 0;
        end
    end else begin
        case (config_state)
            CONFIG_IDLE: begin
                if (config_valid) begin
                    config_state <= CONFIG_LOADING;
                    
                    // 立即存储数据 - 修复关键问题！
                    block_values[config_block_id][config_element_index] <= config_value;
                    block_col_indices[config_block_id][config_element_index] <= config_col_index;
                    
                    // 初始化块信息
                    if (!block_valid[config_block_id]) begin
                        block_valid[config_block_id] <= 1'b1;
                        block_row_start[config_block_id] <= config_row_index;
                        block_size[config_block_id] <= 1; // 至少有一个元素
                    end else begin
                        // 更新块大小
                        if (config_element_index + 1 > block_size[config_block_id]) begin
                            block_size[config_block_id] <= config_element_index + 1;
                        end
                    end
                end
            end
            
            CONFIG_LOADING: begin
                if (config_valid) begin
                    // 继续存储数据
                    block_values[config_block_id][config_element_index] <= config_value;
                    block_col_indices[config_block_id][config_element_index] <= config_col_index;
                    
                    // 更新块大小
                    if (config_element_index + 1 > block_size[config_block_id]) begin
                        block_size[config_block_id] <= config_element_index + 1;
                    end
                end
                
                if (config_block_complete) begin
                    config_state <= CONFIG_COMPLETE;
                    num_blocks_stored <= num_blocks_stored + 1;
                end else if (!config_valid) begin
                    // 如果没有更多配置数据，回到IDLE
                    config_state <= CONFIG_IDLE;
                end
            end
            
            CONFIG_COMPLETE: begin
                config_state <= CONFIG_IDLE;
            end
        endcase
    end
end

// 修复后的输出状态机逻辑
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        output_state <= OUTPUT_IDLE;
        current_output_block <= 0;
        current_output_element <= 0;
        output_block_size <= 0;
    end else begin
        case (output_state)
            OUTPUT_IDLE: begin
                if (block_request_valid && block_request_ready) begin
                    current_output_block <= block_request_id;
                    current_output_element <= 0;
                    output_block_size <= block_size[block_request_id];
                    
                    // 简化条件检查 - 只要块有效就输出
                    if (block_valid[block_request_id]) begin
                        output_state <= OUTPUT_READING;
                    end else begin
                        // 如果块无效，立即回到IDLE
                        output_state <= OUTPUT_IDLE;
                    end
                end
            end
            
            OUTPUT_READING: begin
                // 添加一个周期延迟以确保数据稳定
                output_state <= OUTPUT_SENDING;
            end
            
            OUTPUT_SENDING: begin
                if (block_data_ready) begin
                    if (current_output_element < output_block_size - 1) begin
                        current_output_element <= current_output_element + 1;
                        output_state <= OUTPUT_READING;
                    end else begin
                        output_state <= OUTPUT_IDLE;
                    end
                end
                // 注意：如果block_data_ready为0，保持在SENDING状态等待
            end
        endcase
    end
end

// 输出信号赋值
assign config_ready = (config_state == CONFIG_IDLE) || (config_state == CONFIG_LOADING);
assign block_request_ready = (output_state == OUTPUT_IDLE);

assign block_data_valid = (output_state == OUTPUT_SENDING);
assign block_data_id = current_output_block;
assign block_data_value = block_values[current_output_block][current_output_element];
assign block_data_col_index = block_col_indices[current_output_block][current_output_element];
assign block_data_row_start = block_row_start[current_output_block];
assign block_data_element_index = current_output_element;
assign block_data_last = (current_output_element == output_block_size - 1);

assign total_blocks_stored = num_blocks_stored;
assign hbm_busy = (config_state != CONFIG_IDLE) || (output_state != OUTPUT_IDLE);

endmodule 