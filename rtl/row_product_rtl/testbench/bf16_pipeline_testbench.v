// Testbench for BF16 Pipeline Modules
// Tests FP32 to BF16 conversion, BF16 addition, and BF16 multiplication

`timescale 1ns / 1ps

module bf16_pipeline_testbench;

// Clock and reset
reg clk;
reg rst_n;

// FP32 to BF16 converter signals
reg [31:0] fp32_in;
reg fp32_valid_in;
wire [15:0] bf16_conv_out;
wire bf16_conv_valid_out;

// BF16 adder signals
reg [15:0] add_a, add_b;
reg add_valid_in;
wire [15:0] add_result;
wire add_valid_out;

// BF16 multiplier signals
reg [15:0] mul_a, mul_b;
reg mul_valid_in;
wire [15:0] mul_result;
wire mul_valid_out;

// Instantiate modules
fp32_to_bf16_pipeline fp32_to_bf16_inst (
    .clk(clk),
    .rst_n(rst_n),
    .fp32_in(fp32_in),
    .valid_in(fp32_valid_in),
    .bf16_out(bf16_conv_out),
    .valid_out(bf16_conv_valid_out)
);

bf16_add_pipeline bf16_add_inst (
    .clk(clk),
    .rst_n(rst_n),
    .bf16_a(add_a),
    .bf16_b(add_b),
    .valid_in(add_valid_in),
    .bf16_out(add_result),
    .valid_out(add_valid_out)
);

bf16_multiply_pipeline bf16_mul_inst (
    .clk(clk),
    .rst_n(rst_n),
    .bf16_a(mul_a),
    .bf16_b(mul_b),
    .valid_in(mul_valid_in),
    .bf16_out(mul_result),
    .valid_out(mul_valid_out)
);

// Clock generation
initial begin
    clk = 0;
    forever #5 clk = ~clk; // 100MHz clock
end

// Test sequence
initial begin
    // Initialize signals
    rst_n = 0;
    fp32_in = 32'h0;
    fp32_valid_in = 1'b0;
    add_a = 16'h0;
    add_b = 16'h0;
    add_valid_in = 1'b0;
    mul_a = 16'h0;
    mul_b = 16'h0;
    mul_valid_in = 1'b0;
    
    // Reset pulse
    #20 rst_n = 1;
    #10;
    
    $display("Starting BF16 Pipeline Tests");
    $display("================================");
    
    // Test 1: FP32 to BF16 conversion
    $display("\nTest 1: FP32 to BF16 Conversion");
    $display("--------------------------------");
    
    // Test case 1: 3.14159 (0x40490FDB)
    fp32_in = 32'h40490FDB;
    fp32_valid_in = 1'b1;
    #10 fp32_valid_in = 1'b0;
    
    // Wait for result
    wait(bf16_conv_valid_out);
    #10;
    $display("FP32: 0x%h -> BF16: 0x%h", 32'h40490FDB, bf16_conv_out);
    
    // Test case 2: 1.5 (0x3FC00000)
    fp32_in = 32'h3FC00000;
    fp32_valid_in = 1'b1;
    #10 fp32_valid_in = 1'b0;
    
    wait(bf16_conv_valid_out);
    #10;
    $display("FP32: 0x%h -> BF16: 0x%h", 32'h3FC00000, bf16_conv_out);
    
    // Test 2: BF16 Addition
    $display("\nTest 2: BF16 Addition");
    $display("---------------------");
    
    // Test case 1: 1.5 + 2.25 (BF16: 0x3FC0 + 0x4010)
    add_a = 16'h3FC0;  // 1.5 in BF16
    add_b = 16'h4010;  // 2.25 in BF16
    add_valid_in = 1'b1;
    #10 add_valid_in = 1'b0;
    
    wait(add_valid_out);
    #10;
    $display("BF16 Add: 0x%h + 0x%h = 0x%h", 16'h3FC0, 16'h4010, add_result);
    
    // Test case 2: Special case - infinity + finite
    add_a = 16'h7F80;  // +infinity
    add_b = 16'h4000;  // 2.0
    add_valid_in = 1'b1;
    #10 add_valid_in = 1'b0;
    
    wait(add_valid_out);
    #10;
    $display("BF16 Add: 0x%h + 0x%h = 0x%h (inf + finite)", 16'h7F80, 16'h4000, add_result);
    
    // Test 3: BF16 Multiplication
    $display("\nTest 3: BF16 Multiplication");
    $display("---------------------------");
    
    // Test case 1: 2.0 * 3.0 (BF16: 0x4000 * 0x4040)
    mul_a = 16'h4000;  // 2.0 in BF16
    mul_b = 16'h4040;  // 3.0 in BF16
    mul_valid_in = 1'b1;
    #10 mul_valid_in = 1'b0;
    
    wait(mul_valid_out);
    #10;
    $display("BF16 Mul: 0x%h * 0x%h = 0x%h", 16'h4000, 16'h4040, mul_result);
    
    // Test case 2: Special case - zero * infinity
    mul_a = 16'h0000;  // 0.0
    mul_b = 16'h7F80;  // +infinity
    mul_valid_in = 1'b1;
    #10 mul_valid_in = 1'b0;
    
    wait(mul_valid_out);
    #10;
    $display("BF16 Mul: 0x%h * 0x%h = 0x%h (0 * inf)", 16'h0000, 16'h7F80, mul_result);
    
    // Test 4: Pipeline throughput test
    $display("\nTest 4: Pipeline Throughput");
    $display("---------------------------");
    
    // Send multiple inputs back-to-back
    $display("Sending multiple FP32 values...");
    
    fp32_in = 32'h40000000; // 2.0
    fp32_valid_in = 1'b1;
    #10;
    
    fp32_in = 32'h40400000; // 3.0
    #10;
    
    fp32_in = 32'h40800000; // 4.0
    #10;
    
    fp32_in = 32'h40A00000; // 5.0
    #10;
    
    fp32_valid_in = 1'b0;
    
    // Wait for all outputs
    repeat(10) begin
        if (bf16_conv_valid_out) begin
            $display("Pipeline output: 0x%h", bf16_conv_out);
        end
        #10;
    end
    
    $display("\nAll tests completed!");
    #100;
    $finish;
end

// Monitor signals
initial begin
    $monitor("Time: %0t | FP32_Conv: %b | Add: %b | Mul: %b", 
             $time, bf16_conv_valid_out, add_valid_out, mul_valid_out);
end

// Optional: Generate VCD file for waveform viewing
initial begin
    $dumpfile("bf16_pipeline_test.vcd");
    $dumpvars(0, bf16_pipeline_testbench);
end

endmodule 