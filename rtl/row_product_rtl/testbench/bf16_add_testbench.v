// Testbench for BF16 Addition Pipeline
`timescale 1ns/1ps

module bf16_add_testbench;

    // Inputs
    reg clk;
    reg rst_n;
    reg valid_in;
    reg [15:0] bf16_a;
    reg [15:0] bf16_b;

    // Outputs
    wire valid_out;
    wire [15:0] bf16_out;

    // Instantiate the Unit Under Test (UUT)
    bf16_add_pipeline uut (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .bf16_a(bf16_a),
        .bf16_b(bf16_b),
        .valid_out(valid_out),
        .bf16_out(bf16_out)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // 100MHz clock
    end

    // Test sequence
    initial begin
        // Initialize inputs
        rst_n = 0;
        valid_in = 0;
        bf16_a = 16'h0000;
        bf16_b = 16'h0000;

        // Wait for reset
        #20;
        rst_n = 1;
        #10;

        // Test case 1: Simple addition (1.0 + 1.0 = 2.0)
        // BF16 1.0 = 0x3F80, BF16 2.0 = 0x4000
        $display("Test 1: 1.0 + 1.0");
        valid_in = 1;
        bf16_a = 16'h3F80; // 1.0 in BF16
        bf16_b = 16'h3F80; // 1.0 in BF16
        #10;
        valid_in = 0;
        
        // Wait for result (5 cycles pipeline)
        repeat(6) @(posedge clk);
        if (valid_out)
            $display("Result: %h (Expected: 4000)", bf16_out);
        else
            $display("No valid output yet");
        #10;

        // Test case 2: Zero addition (0.0 + 1.0 = 1.0)
        $display("Test 2: 0.0 + 1.0");
        valid_in = 1;
        bf16_a = 16'h0000; // 0.0 in BF16
        bf16_b = 16'h3F80; // 1.0 in BF16
        #10;
        valid_in = 0;
        
        repeat(6) @(posedge clk);
        if (valid_out)
            $display("Result: %h (Expected: 3F80)", bf16_out);
        #10;

        // Test case 3: Negative addition (1.0 + (-1.0) = 0.0)
        $display("Test 3: 1.0 + (-1.0)");
        valid_in = 1;
        bf16_a = 16'h3F80; // 1.0 in BF16
        bf16_b = 16'hBF80; // -1.0 in BF16
        #10;
        valid_in = 0;
        
        repeat(6) @(posedge clk);
        if (valid_out)
            $display("Result: %h (Expected: 0000)", bf16_out);
        #10;

        // Test case 4: Small numbers (0.5 + 0.5 = 1.0)
        $display("Test 4: 0.5 + 0.5");
        valid_in = 1;
        bf16_a = 16'h3F00; // 0.5 in BF16
        bf16_b = 16'h3F00; // 0.5 in BF16
        #10;
        valid_in = 0;
        
        repeat(6) @(posedge clk);
        if (valid_out)
            $display("Result: %h (Expected: 3F80)", bf16_out);
        #10;

        $display("All tests completed");
        $finish;
    end

    // Monitor outputs
    initial begin
        $monitor("Time=%t, clk=%b, rst_n=%b, valid_in=%b, bf16_a=%h, bf16_b=%h, valid_out=%b, bf16_out=%h",
                 $time, clk, rst_n, valid_in, bf16_a, bf16_b, valid_out, bf16_out);
    end

endmodule 