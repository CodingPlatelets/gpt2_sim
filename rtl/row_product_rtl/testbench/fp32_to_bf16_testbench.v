`timescale 1ns/1ps

module fp32_to_bf16_testbench;

// Clock and reset
reg clk;
reg rst_n;

// Test inputs
reg [31:0] fp32_in;
reg valid_in;

// Test outputs
wire [15:0] bf16_out;
wire valid_out;

// Instantiate the DUT (Device Under Test)
fp32_to_bf16_pipeline dut (
    .clk(clk),
    .rst_n(rst_n),
    .fp32_in(fp32_in),
    .valid_in(valid_in),
    .bf16_out(bf16_out),
    .valid_out(valid_out)
);

// Clock generation (100MHz = 10ns period)
always #5 clk = ~clk;

// Test procedure
initial begin
    // Initialize
    clk = 0;
    rst_n = 0;
    fp32_in = 32'h0;
    valid_in = 0;
    
    // Reset sequence
    #20 rst_n = 1;
    #10;
    
    // Test 1: Convert 1.0 (FP32: 0x3F800000 -> BF16: 0x3F80)
    $display("Test 1: 1.0");
    fp32_in = 32'h3F800000;
    valid_in = 1;
    #10 valid_in = 0;
    
    // Wait for pipeline delay (3 cycles)
    repeat(4) @(posedge clk);
    if (valid_out && bf16_out == 16'h3F80)
        $display("✓ Test 1 PASSED: 1.0 -> %h", bf16_out);
    else
        $display("✗ Test 1 FAILED: Expected 3F80, got %h", bf16_out);
    
    // Test 2: Convert 2.0 (FP32: 0x40000000 -> BF16: 0x4000)
    $display("Test 2: 2.0");
    fp32_in = 32'h40000000;
    valid_in = 1;
    #10 valid_in = 0;
    
    repeat(4) @(posedge clk);
    if (valid_out && bf16_out == 16'h4000)
        $display("✓ Test 2 PASSED: 2.0 -> %h", bf16_out);
    else
        $display("✗ Test 2 FAILED: Expected 4000, got %h", bf16_out);
    
    // Test 3: Convert -1.0 (FP32: 0xBF800000 -> BF16: 0xBF80)
    $display("Test 3: -1.0");
    fp32_in = 32'hBF800000;
    valid_in = 1;
    #10 valid_in = 0;
    
    repeat(4) @(posedge clk);
    if (valid_out && bf16_out == 16'hBF80)
        $display("✓ Test 3 PASSED: -1.0 -> %h", bf16_out);
    else
        $display("✗ Test 3 FAILED: Expected BF80, got %h", bf16_out);
    
    // Test 4: Convert 0.0 (FP32: 0x00000000 -> BF16: 0x0000)
    $display("Test 4: 0.0");
    fp32_in = 32'h00000000;
    valid_in = 1;
    #10 valid_in = 0;
    
    repeat(4) @(posedge clk);
    if (valid_out && bf16_out == 16'h0000)
        $display("✓ Test 4 PASSED: 0.0 -> %h", bf16_out);
    else
        $display("✗ Test 4 FAILED: Expected 0000, got %h", bf16_out);
    
    // Test 5: Convert 3.14159 (approximately π)
    $display("Test 5: 3.14159 (π)");
    fp32_in = 32'h40490FDA; // 3.14159 in FP32
    valid_in = 1;
    #10 valid_in = 0;
    
    repeat(4) @(posedge clk);
    $display("Test 5: 3.14159 -> %h", bf16_out);
    
    // Test 6: Convert a small number (0.125)
    $display("Test 6: 0.125");
    fp32_in = 32'h3E000000; // 0.125 in FP32
    valid_in = 1;
    #10 valid_in = 0;
    
    repeat(4) @(posedge clk);
    if (valid_out && bf16_out == 16'h3E00)
        $display("✓ Test 6 PASSED: 0.125 -> %h", bf16_out);
    else
        $display("✗ Test 6 FAILED: Expected 3E00, got %h", bf16_out);
    
    // Test 7: Convert positive infinity
    $display("Test 7: +Infinity");
    fp32_in = 32'h7F800000; // +Inf in FP32
    valid_in = 1;
    #10 valid_in = 0;
    
    repeat(4) @(posedge clk);
    if (valid_out && bf16_out == 16'h7F80)
        $display("✓ Test 7 PASSED: +Inf -> %h", bf16_out);
    else
        $display("✗ Test 7 FAILED: Expected 7F80, got %h", bf16_out);
    
    // Test 8: Convert negative infinity
    $display("Test 8: -Infinity");
    fp32_in = 32'hFF800000; // -Inf in FP32
    valid_in = 1;
    #10 valid_in = 0;
    
    repeat(4) @(posedge clk);
    if (valid_out && bf16_out == 16'hFF80)
        $display("✓ Test 8 PASSED: -Inf -> %h", bf16_out);
    else
        $display("✗ Test 8 FAILED: Expected FF80, got %h", bf16_out);
    
    // Test 9: Convert NaN
    $display("Test 9: NaN");
    fp32_in = 32'h7FC00000; // NaN in FP32
    valid_in = 1;
    #10 valid_in = 0;
    
    repeat(4) @(posedge clk);
    // For NaN, we expect the high 16 bits with some modification
    $display("Test 9: NaN -> %h", bf16_out);
    
    // Test 10: Test a number that requires rounding
    $display("Test 10: Rounding test");
    fp32_in = 32'h3F800001; // Slightly larger than 1.0
    valid_in = 1;
    #10 valid_in = 0;
    
    repeat(4) @(posedge clk);
    $display("Test 10: 1.0+ -> %h", bf16_out);
    
    $display("All tests completed");
    #50 $finish;
end

// Monitor signals
initial begin
    $monitor("Time=%0t, clk=%b, rst_n=%b, valid_in=%b, fp32_in=%h, valid_out=%b, bf16_out=%h", 
             $time, clk, rst_n, valid_in, fp32_in, valid_out, bf16_out);
end

endmodule 