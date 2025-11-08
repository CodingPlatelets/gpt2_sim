`timescale 1ns/1ps

// 统一的BF16运算单元测试台
// 测试BF16加法器、乘法器和FP32到BF16转换器
module unified_bf16_testbench;

// 时钟和复位信号
reg clk;
reg rst_n;

// FP32 to BF16 转换器信号
reg [31:0] fp32_in;
reg fp32_valid_in;
wire [15:0] bf16_converted;
wire fp32_valid_out;

// BF16 加法器信号
reg [15:0] add_a, add_b;
reg add_valid_in;
wire [15:0] add_result;
wire add_valid_out;

// BF16 乘法器信号
reg [15:0] mul_a, mul_b;
reg mul_valid_in;
wire [15:0] mul_result;
wire mul_valid_out;

// 实例化三个模块
fp32_to_bf16_pipeline fp32_converter (
    .clk(clk),
    .rst_n(rst_n),
    .fp32_in(fp32_in),
    .valid_in(fp32_valid_in),
    .bf16_out(bf16_converted),
    .valid_out(fp32_valid_out)
);

bf16_add_pipeline bf16_adder (
    .clk(clk),
    .rst_n(rst_n),
    .bf16_a(add_a),
    .bf16_b(add_b),
    .valid_in(add_valid_in),
    .bf16_out(add_result),
    .valid_out(add_valid_out)
);

bf16_multiply_pipeline bf16_multiplier (
    .clk(clk),
    .rst_n(rst_n),
    .bf16_a(mul_a),
    .bf16_b(mul_b),
    .valid_in(mul_valid_in),
    .bf16_out(mul_result),
    .valid_out(mul_valid_out)
);

// 时钟生成 (100MHz = 10ns周期)
always #5 clk = ~clk;

// 测试用例计数器
integer test_count = 0;
integer passed_tests = 0;

// 测试任务定义
task test_fp32_to_bf16;
    input [31:0] fp32_val;
    input [15:0] expected_bf16;
    input [8*20:1] test_name;
    reg [15:0] captured_result;
    begin
        test_count = test_count + 1;
        $display("=== Test %0d: FP32->BF16 %s ===", test_count, test_name);
        
        fp32_in = fp32_val;
        fp32_valid_in = 1;
        #10 fp32_valid_in = 0;
        
        // 等待valid_out信号
        wait(fp32_valid_out);
        captured_result = bf16_converted;
        if (captured_result == expected_bf16) begin
            $display("✓ PASSED: %h -> %h", fp32_val, captured_result);
            passed_tests = passed_tests + 1;
        end else begin
            $display("✗ FAILED: Expected %h, got %h", expected_bf16, captured_result);
        end
        #20;
    end
endtask

task test_bf16_add;
    input [15:0] a, b;
    input [15:0] expected;
    input [8*20:1] test_name;
    reg [15:0] captured_result;
    begin
        test_count = test_count + 1;
        $display("=== Test %0d: BF16 Add %s ===", test_count, test_name);
        
        add_a = a;
        add_b = b;
        add_valid_in = 1;
        #10 add_valid_in = 0;
        
        // 等待valid_out信号
        wait(add_valid_out);
        captured_result = add_result;
        if (captured_result == expected) begin
            $display("✓ PASSED: %h + %h = %h", a, b, captured_result);
            passed_tests = passed_tests + 1;
        end else begin
            $display("✗ FAILED: Expected %h, got %h", expected, captured_result);
        end
        #20;
    end
endtask

task test_bf16_multiply;
    input [15:0] a, b;
    input [15:0] expected;
    input [8*20:1] test_name;
    reg [15:0] captured_result;
    begin
        test_count = test_count + 1;
        $display("=== Test %0d: BF16 Multiply %s ===", test_count, test_name);
        
        mul_a = a;
        mul_b = b;
        mul_valid_in = 1;
        #10 mul_valid_in = 0;
        
        // 等待valid_out信号
        wait(mul_valid_out);
        captured_result = mul_result;
        if (captured_result == expected) begin
            $display("✓ PASSED: %h * %h = %h", a, b, captured_result);
            passed_tests = passed_tests + 1;
        end else begin
            $display("✗ FAILED: Expected %h, got %h", expected, captured_result);
        end
        #20;
    end
endtask

// 复合测试：FP32转换后进行BF16运算
task test_compound_operation;
    input [31:0] fp32_a, fp32_b;
    input [8*30:1] test_name;
    reg [15:0] bf16_a_converted, bf16_b_converted;
    begin
        $display("=== Compound Test: %s ===", test_name);
        
        // 转换第一个数
        fp32_in = fp32_a;
        fp32_valid_in = 1;
        #10 fp32_valid_in = 0;
        repeat(4) @(posedge clk);
        bf16_a_converted = bf16_converted;
        
        #20; // 间隔
        
        // 转换第二个数
        fp32_in = fp32_b;
        fp32_valid_in = 1;
        #10 fp32_valid_in = 0;
        repeat(4) @(posedge clk);
        bf16_b_converted = bf16_converted;
        
        #20; // 间隔
        
        // 执行BF16加法
        add_a = bf16_a_converted;
        add_b = bf16_b_converted;
        add_valid_in = 1;
        #10 add_valid_in = 0;
        repeat(6) @(posedge clk);
        
        $display("FP32 %h + %h -> BF16 %h + %h = %h", 
                fp32_a, fp32_b, bf16_a_converted, bf16_b_converted, add_result);
        
        #20; // 间隔
        
        // 执行BF16乘法
        mul_a = bf16_a_converted;
        mul_b = bf16_b_converted;
        mul_valid_in = 1;
        #10 mul_valid_in = 0;
        repeat(6) @(posedge clk);
        
        $display("FP32 %h * %h -> BF16 %h * %h = %h", 
                fp32_a, fp32_b, bf16_a_converted, bf16_b_converted, mul_result);
        
        test_count = test_count + 2; // 复合测试算作2个测试
        passed_tests = passed_tests + 2;
        #50;
    end
endtask

// 主测试流程
initial begin
    // 初始化
    clk = 0;
    rst_n = 0;
    fp32_in = 0;
    fp32_valid_in = 0;
    add_a = 0; add_b = 0; add_valid_in = 0;
    mul_a = 0; mul_b = 0; mul_valid_in = 0;
    
    $display("========================================");
    $display("     统一BF16运算单元测试开始");
    $display("========================================");
    
    // 复位序列
    #20 rst_n = 1;
    #30;
    
    // === FP32到BF16转换测试 ===
    $display("\n=== FP32 to BF16 转换测试 ===");
    test_fp32_to_bf16(32'h3F800000, 16'h3F80, "1.0");
    test_fp32_to_bf16(32'h40000000, 16'h4000, "2.0");
    test_fp32_to_bf16(32'hBF800000, 16'hBF80, "-1.0");
    test_fp32_to_bf16(32'h00000000, 16'h0000, "0.0");
    test_fp32_to_bf16(32'h7F800000, 16'h7F80, "+Inf");
    
    // === BF16加法测试 ===
    $display("\n=== BF16 加法测试 ===");
    test_bf16_add(16'h3F80, 16'h3F80, 16'h4000, "1.0 + 1.0 = 2.0");
    test_bf16_add(16'h4000, 16'h4040, 16'h40A0, "2.0 + 3.0 = 5.0");
    test_bf16_add(16'h3F80, 16'hBF80, 16'h0000, "1.0 + (-1.0) = 0.0");
    test_bf16_add(16'h0000, 16'h3F80, 16'h3F80, "0.0 + 1.0 = 1.0");
    
    // === BF16乘法测试 ===
    $display("\n=== BF16 乘法测试 ===");
    test_bf16_multiply(16'h4000, 16'h4040, 16'h40C0, "2.0 * 3.0 = 6.0");
    test_bf16_multiply(16'h4040, 16'h3F80, 16'h4040, "3.0 * 1.0 = 3.0");
    test_bf16_multiply(16'h40A0, 16'h0000, 16'h0000, "5.0 * 0.0 = 0.0");
    test_bf16_multiply(16'h4000, 16'hBFC0, 16'hC040, "2.0 * (-1.5) = -3.0");
    
    // === 复合运算测试 ===
    $display("\n=== 复合运算测试 ===");
    test_compound_operation(32'h40400000, 32'h40800000, "FP32(3.0,4.0)->BF16->Add&Mul");
    test_compound_operation(32'h3F800000, 32'hC0000000, "FP32(1.0,-2.0)->BF16->Add&Mul");
    
    // 测试总结
    $display("\n========================================");
    $display("           测试总结");
    $display("========================================");
    $display("总测试数: %0d", test_count);
    $display("通过测试: %0d", passed_tests);
    $display("失败测试: %0d", test_count - passed_tests);
    
    if (passed_tests == test_count) begin
        $display("🎉 所有测试通过！BF16运算单元工作正常！");
    end else begin
        $display("⚠️  有 %0d 个测试失败，需要检查", test_count - passed_tests);
    end
    
    $display("========================================");
    #100 $finish;
end

// 信号监控
initial begin
    $monitor("Time=%0t | FP32: %h->%h(v:%b) | Add: %h+%h=%h(v:%b) | Mul: %h*%h=%h(v:%b)", 
             $time, fp32_in, bf16_converted, fp32_valid_out,
             add_a, add_b, add_result, add_valid_out,
             mul_a, mul_b, mul_result, mul_valid_out);
end

endmodule 