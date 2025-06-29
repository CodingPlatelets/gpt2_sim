// Fixed Processing Element Row Module
// 修复了任务调度和BRAM访问问题

module processing_element_row #(
    parameter PEROW_ID = 0,
    parameter NUM_PES = 128,
    parameter VECTOR_SIZE = 4096,
    parameter ADDR_WIDTH = 12,
    parameter PE_ADDR_WIDTH = 7,  // log2(128)
    parameter TASK_QUEUE_DEPTH = 1024,
    parameter TASK_QUEUE_ADDR_WIDTH = 10  // log2(1024)
)(
    input wire clk,
    input wire rst_n,
    
    // 单个任务输入接口
    input wire task_valid,
    input wire [15:0] a_value,
    input wire [15:0] b_value,
    input wire [ADDR_WIDTH-1:0] col_index,
    output wire task_ready,
    
    // 批量任务输入接口
    input wire batch_task_valid,
    input wire [15:0] batch_a_value,
    input wire [15:0] batch_b_value,
    input wire [ADDR_WIDTH-1:0] batch_col_index,
    input wire [15:0] batch_task_count,
    output wire batch_task_ready,
    
    // 状态输出
    output wire perow_busy,
    output wire [31:0] total_tasks_completed,
    output wire [TASK_QUEUE_ADDR_WIDTH:0] task_queue_count,
    
    // 结果向量输出接口 (给外部访问)
    output wire result_vector_valid,
    output wire [ADDR_WIDTH-1:0] result_vector_addr,
    output wire [15:0] result_vector_data,
    input wire result_vector_read_ack
);

// 任务队列存储
reg [47:0] task_queue [0:TASK_QUEUE_DEPTH-1];
reg [TASK_QUEUE_ADDR_WIDTH:0] queue_write_ptr;
reg [TASK_QUEUE_ADDR_WIDTH:0] queue_read_ptr;
reg [TASK_QUEUE_ADDR_WIDTH:0] queue_count;

// PE调度状态
reg [PE_ADDR_WIDTH-1:0] pe_scheduler_ptr;
reg task_dispatch_valid;
reg [15:0] task_dispatch_a, task_dispatch_b;
reg [ADDR_WIDTH-1:0] task_dispatch_col;
reg scheduler_busy;  // ✅ 新增：防止调度器指针乱跳

// PE接口信号
wire [NUM_PES-1:0] pe_task_ready;
wire [NUM_PES-1:0] pe_busy;
wire [31:0] pe_tasks_completed [0:NUM_PES-1];

// PE的结果向量访问信号
wire [NUM_PES-1:0] pe_result_write_enable;
wire [ADDR_WIDTH-1:0] pe_result_write_addr [0:NUM_PES-1];
wire [15:0] pe_result_write_data [0:NUM_PES-1];
wire [NUM_PES-1:0] pe_result_read_enable;
wire [ADDR_WIDTH-1:0] pe_result_read_addr [0:NUM_PES-1];
wire [15:0] pe_result_read_data [0:NUM_PES-1];

// BRAM控制信号
wire bram_write_enable;
wire [ADDR_WIDTH-1:0] bram_write_addr;
wire [15:0] bram_write_data;
wire bram_read_enable;
wire [ADDR_WIDTH-1:0] bram_read_addr;
wire [15:0] bram_read_data;

// 结果向量双端口BRAM
dual_port_bram #(
    .DATA_WIDTH(16),
    .ADDR_WIDTH(ADDR_WIDTH),
    .DEPTH(VECTOR_SIZE)
) result_bram_inst (
    .clk(clk),
    .rst_n(rst_n),
    .write_enable(bram_write_enable),
    .write_addr(bram_write_addr),
    .write_data(bram_write_data),
    .read_enable(bram_read_enable),
    .read_addr(bram_read_addr),
    .read_data(bram_read_data)
);

// ✅ 修复后的BRAM仲裁器 - 支持多个PE并发访问
reg [PE_ADDR_WIDTH-1:0] bram_write_arbiter_ptr;
reg [PE_ADDR_WIDTH-1:0] bram_read_arbiter_ptr;
reg bram_write_grant;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        bram_write_arbiter_ptr <= 0;
        bram_read_arbiter_ptr <= 0;
        bram_write_grant <= 1'b0;
    end else begin
        // 写仲裁：轮询找到需要写的PE
        if (!bram_write_grant) begin
            if (pe_result_write_enable[bram_write_arbiter_ptr]) begin
                bram_write_grant <= 1'b1;
            end else begin
                bram_write_arbiter_ptr <= (bram_write_arbiter_ptr + 1) % NUM_PES;
            end
        end else begin
            bram_write_grant <= 1'b0;
            bram_write_arbiter_ptr <= (bram_write_arbiter_ptr + 1) % NUM_PES;
        end
        
        // 读仲裁：每周期轮询
        bram_read_arbiter_ptr <= (bram_read_arbiter_ptr + 1) % NUM_PES;
    end
end

// BRAM信号连接
assign bram_write_enable = bram_write_grant && pe_result_write_enable[bram_write_arbiter_ptr];
assign bram_write_addr = pe_result_write_addr[bram_write_arbiter_ptr];
assign bram_write_data = pe_result_write_data[bram_write_arbiter_ptr];

assign bram_read_enable = pe_result_read_enable[bram_read_arbiter_ptr];
assign bram_read_addr = pe_result_read_addr[bram_read_arbiter_ptr];

// ✅ 修复：为每个PE分配正确的读取数据
genvar i;
generate
    for (i = 0; i < NUM_PES; i = i + 1) begin : pe_read_data_gen
        assign pe_result_read_data[i] = (bram_read_arbiter_ptr == i) ? bram_read_data : 16'b0;
    end
endgenerate

// PE实例化
generate
    for (i = 0; i < NUM_PES; i = i + 1) begin : pe_gen
        processing_element #(
            .PE_ID(i),
            .PEROW_ID(PEROW_ID),
            .VECTOR_SIZE(VECTOR_SIZE),
            .ADDR_WIDTH(ADDR_WIDTH)
        ) pe_inst (
            .clk(clk),
            .rst_n(rst_n),
            .task_valid(task_dispatch_valid && (pe_scheduler_ptr == i)),
            .a_value(task_dispatch_a),
            .b_value(task_dispatch_b),
            .col_index(task_dispatch_col),
            .task_ready(pe_task_ready[i]),
            .result_write_enable(pe_result_write_enable[i]),
            .result_write_addr(pe_result_write_addr[i]),
            .result_write_data(pe_result_write_data[i]),
            .result_read_data(pe_result_read_data[i]),
            .result_read_enable(pe_result_read_enable[i]),
            .result_read_addr(pe_result_read_addr[i]),
            .pe_busy(pe_busy[i]),
            .tasks_completed(pe_tasks_completed[i])
        );
    end
endgenerate

// ✅ 修复后的任务队列管理
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        queue_write_ptr <= 0;
        queue_read_ptr <= 0;
        queue_count <= 0;
        pe_scheduler_ptr <= 0;
        task_dispatch_valid <= 1'b0;
        scheduler_busy <= 1'b0;
    end else begin
        // 任务入队
        if (task_valid && task_ready) begin
            task_queue[queue_write_ptr[TASK_QUEUE_ADDR_WIDTH-1:0]] <= {a_value, b_value, {{(16-ADDR_WIDTH){1'b0}}, col_index}};
            queue_write_ptr <= queue_write_ptr + 1;
            queue_count <= queue_count + 1;
        end
        
        // 批量任务入队
        if (batch_task_valid && batch_task_ready) begin
            task_queue[queue_write_ptr[TASK_QUEUE_ADDR_WIDTH-1:0]] <= {batch_a_value, batch_b_value, {{(16-ADDR_WIDTH){1'b0}}, batch_col_index}};
            queue_write_ptr <= queue_write_ptr + 1;
            queue_count <= queue_count + 1;
        end
        
        // ✅ 修复后的任务分发逻辑
        if (queue_count > 0 && !scheduler_busy) begin
            // 开始调度，标记忙碌
            scheduler_busy <= 1'b1;
            task_dispatch_a <= task_queue[queue_read_ptr[TASK_QUEUE_ADDR_WIDTH-1:0]][47:32];
            task_dispatch_b <= task_queue[queue_read_ptr[TASK_QUEUE_ADDR_WIDTH-1:0]][31:16];
            task_dispatch_col <= task_queue[queue_read_ptr[TASK_QUEUE_ADDR_WIDTH-1:0]][ADDR_WIDTH-1:0];
            task_dispatch_valid <= 1'b1;
        end else if (scheduler_busy) begin
            // 正在调度过程中
            if (pe_task_ready[pe_scheduler_ptr]) begin
                // ✅ 成功分发任务，从队列移除
                queue_read_ptr <= queue_read_ptr + 1;
                queue_count <= queue_count - 1;
                task_dispatch_valid <= 1'b0;
                scheduler_busy <= 1'b0;
                // 只有成功分发后才更新指针
                pe_scheduler_ptr <= (pe_scheduler_ptr + 1) % NUM_PES;
            end else begin
                // PE不ready，尝试下一个PE（不移除队列任务）
                pe_scheduler_ptr <= (pe_scheduler_ptr + 1) % NUM_PES;
                // 继续保持调度状态
            end
        end else begin
            task_dispatch_valid <= 1'b0;
        end
    end
end

// 状态和控制信号
assign task_ready = (queue_count < TASK_QUEUE_DEPTH);
assign batch_task_ready = (queue_count < TASK_QUEUE_DEPTH);
assign perow_busy = (queue_count > 0) || (|pe_busy) || scheduler_busy;

// 计算总完成任务数
reg [31:0] total_completed;
integer k;
always @(*) begin
    total_completed = 32'b0;
    for (k = 0; k < NUM_PES; k = k + 1) begin
        total_completed = total_completed + pe_tasks_completed[k];
    end
end

assign total_tasks_completed = total_completed;
assign task_queue_count = queue_count;

// 结果向量输出接口 (简化版本)
assign result_vector_valid = 1'b0;
assign result_vector_addr = {ADDR_WIDTH{1'b0}};
assign result_vector_data = 16'b0;

endmodule

// 双端口BRAM模块（保持不变）
module dual_port_bram #(
    parameter DATA_WIDTH = 16,
    parameter ADDR_WIDTH = 12,
    parameter DEPTH = 4096
)(
    input wire clk,
    input wire rst_n,
    input wire write_enable,
    input wire [ADDR_WIDTH-1:0] write_addr,
    input wire [DATA_WIDTH-1:0] write_data,
    input wire read_enable,
    input wire [ADDR_WIDTH-1:0] read_addr,
    output reg [DATA_WIDTH-1:0] read_data
);

reg [DATA_WIDTH-1:0] memory [0:DEPTH-1];

always @(posedge clk) begin
    if (write_enable) begin
        memory[write_addr] <= write_data;
    end
end

always @(posedge clk) begin
    if (!rst_n) begin
        read_data <= {DATA_WIDTH{1'b0}};
    end else if (read_enable) begin
        read_data <= memory[read_addr];
    end
end

endmodule 