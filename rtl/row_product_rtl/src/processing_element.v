// Processing Element (PE) Module
// 实现三级流水线：输入缓存 -> 乘法 -> 累加
// 对应Python中的ProcessingElement类

module processing_element #(
    parameter PE_ID = 0,
    parameter PEROW_ID = 0,
    parameter VECTOR_SIZE = 4096,
    parameter ADDR_WIDTH = 12  // log2(4096)
)(
    input wire clk,
    input wire rst_n,
    
    // 任务输入接口
    input wire task_valid,
    input wire [15:0] a_value,      // BF16格式
    input wire [15:0] b_value,      // BF16格式
    input wire [ADDR_WIDTH-1:0] col_index,
    output wire task_ready,
    
    // 结果向量接口 (连接到外部BRAM)
    output wire result_write_enable,
    output wire [ADDR_WIDTH-1:0] result_write_addr,
    output wire [15:0] result_write_data,
    input wire [15:0] result_read_data,
    output wire result_read_enable,
    output wire [ADDR_WIDTH-1:0] result_read_addr,
    
    // 状态输出
    output wire pe_busy,
    output wire [31:0] tasks_completed
);

// 流水线阶段寄存器
reg [47:0] stage1_data;  // {a_value[15:0], b_value[15:0], col_index[15:0]}
reg stage1_valid;

reg [47:0] stage2_data;
reg stage2_valid;

reg [31:0] stage3_data;  // {mul_result[15:0], col_index[15:0]}
reg stage3_valid;

// 任务完成计数器
reg [31:0] task_counter;

// BF16乘法器实例
wire [15:0] mul_result;
wire mul_valid_out;

bf16_multiply_pipeline bf16_mul_inst (
    .clk(clk),
    .rst_n(rst_n),
    .bf16_a(stage2_data[47:32]),
    .bf16_b(stage2_data[31:16]),
    .valid_in(stage2_valid),
    .bf16_out(mul_result),
    .valid_out(mul_valid_out)
);

// BF16加法器实例
wire [15:0] add_result;
wire add_valid_out;

bf16_add_pipeline bf16_add_inst (
    .clk(clk),
    .rst_n(rst_n),
    .bf16_a(result_read_data),
    .bf16_b(stage3_data[31:16]),
    .valid_in(stage3_valid),
    .bf16_out(add_result),
    .valid_out(add_valid_out)
);

// 状态机定义 (使用参数定义状态)
parameter [2:0] IDLE = 3'b000,
                STAGE1 = 3'b001,
                STAGE2 = 3'b010,
                STAGE3 = 3'b011,
                WRITE_RESULT = 3'b100;

reg [2:0] current_state, next_state;

// 状态转移逻辑
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        current_state <= IDLE;
        stage1_data <= 48'b0;
        stage1_valid <= 1'b0;
        stage2_data <= 48'b0;
        stage2_valid <= 1'b0;
        stage3_data <= 32'b0;
        stage3_valid <= 1'b0;
        task_counter <= 32'b0;
    end else begin
        current_state <= next_state;
        
        // 流水线数据传递
        if (task_valid && task_ready) begin
            stage1_data <= {a_value, b_value, {{(16-ADDR_WIDTH){1'b0}}, col_index}};
            stage1_valid <= 1'b1;
        end else if (stage1_valid && !stage2_valid) begin
            stage2_data <= stage1_data;
            stage2_valid <= stage1_valid;
            stage1_valid <= 1'b0;
        end
        
        if (mul_valid_out && !stage3_valid) begin
            stage3_data <= {mul_result, stage2_data[15:0]};
            stage3_valid <= 1'b1;
            stage2_valid <= 1'b0;
        end
        
        if (add_valid_out) begin
            task_counter <= task_counter + 1;
            stage3_valid <= 1'b0;
        end
    end
end

// 状态机组合逻辑
always @(*) begin
    next_state = current_state;
    
    case (current_state)
        IDLE: begin
            if (task_valid) begin
                next_state = STAGE1;
            end
        end
        
        STAGE1: begin
            if (stage1_valid && !stage2_valid) begin
                next_state = STAGE2;
            end
        end
        
        STAGE2: begin
            if (mul_valid_out) begin
                next_state = STAGE3;
            end
        end
        
        STAGE3: begin
            if (add_valid_out) begin
                next_state = WRITE_RESULT;
            end
        end
        
        WRITE_RESULT: begin
            next_state = IDLE;
        end
        
        default: next_state = IDLE;
    endcase
end

// 输出信号
assign task_ready = (current_state == IDLE) || (!stage1_valid && !stage2_valid);

assign result_read_enable = (current_state == STAGE3) && stage3_valid;
assign result_read_addr = stage3_data[ADDR_WIDTH-1:0];

assign result_write_enable = add_valid_out;
assign result_write_addr = stage3_data[ADDR_WIDTH-1:0];
assign result_write_data = add_result;

assign pe_busy = (current_state != IDLE) || stage1_valid || stage2_valid || stage3_valid;
assign tasks_completed = task_counter;

endmodule 