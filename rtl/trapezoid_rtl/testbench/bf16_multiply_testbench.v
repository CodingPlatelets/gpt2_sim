// Testbench for BF16 Multiplication Pipeline
// Tests the BF16 floating-point multiplication pipeline module
`timescale 1ns/1ps

module bf16_multiply_testbench;

    // Parameters
    parameter CLK_PERIOD = 10; // 100MHz

    // Inputs
    reg clk;
    reg rst_n;
    reg valid_in;
    reg [15:0] bf16_a;
    reg [15:0] bf16_b;

    // Outputs
    wire valid_out;
    wire [15:0] result;

    // Test tracking
    integer test_count;
    integer pass_count;
    integer fail_count;

    // Instantiate the Unit Under Test (UUT)
    bf16_multiply_pipeline uut (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .bf16_a(bf16_a),
        .bf16_b(bf16_b),
        .valid_out(valid_out),
        .result(result)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    // Task for running individual tests
    task run_test;
        input [15:0] a_val;
        input [15:0] b_val;
        input [15:0] expected;
        input [256*8-1:0] test_name;
        
        reg found_result;
        integer timeout_counter;
        
        begin
            $display("Running test: %s", test_name);
            $display("  A=%h, B=%h, Expected=%h", a_val, b_val, expected);
            
            @(posedge clk);
            valid_in = 1;
            bf16_a = a_val;
            bf16_b = b_val;
            @(posedge clk);
            valid_in = 0;
            
            // Wait for pipeline output with timeout
            found_result = 0;
            timeout_counter = 0;
            
            while (!found_result && timeout_counter < 15) begin
                @(posedge clk);
                timeout_counter = timeout_counter + 1;
                
                if (valid_out) begin
                    found_result = 1;
                    $display("  Result=%h (cycle %d)", result, timeout_counter);
                    if (result == expected) begin
                        $display("  PASS");
                        pass_count = pass_count + 1;
                    end else begin
                        $display("  FAIL - Expected %h, got %h", expected, result);
                        fail_count = fail_count + 1;
                    end
                end
            end
            
            if (!found_result) begin
                $display("  FAIL - No valid output after %d cycles", timeout_counter);
                fail_count = fail_count + 1;
            end
            
            // Wait a few more cycles for output to clear
            repeat(3) @(posedge clk);
            
            test_count = test_count + 1;
            $display("");
        end
    endtask

    // Test sequence
    initial begin
        $dumpfile("bf16_multiply_test.vcd");
        $dumpvars(0, bf16_multiply_testbench);

        // Initialize
        test_count = 0;
        pass_count = 0;
        fail_count = 0;
        
        rst_n = 0;
        valid_in = 0;
        bf16_a = 16'h0000;
        bf16_b = 16'h0000;

        // Reset sequence
        #(CLK_PERIOD*5);
        rst_n = 1;
        #(CLK_PERIOD*2);

        $display("==== BF16 Multiplication Pipeline Testbench ====");
        $display("Clock period: %d ns", CLK_PERIOD);
        $display("");

        // Test cases
        run_test(16'h3F80, 16'h3F80, 16'h3F80, "1.0 * 1.0 = 1.0");
        run_test(16'h4000, 16'h4000, 16'h4080, "2.0 * 2.0 = 4.0");
        run_test(16'h3F80, 16'h4000, 16'h4000, "1.0 * 2.0 = 2.0");
        run_test(16'h0000, 16'h3F80, 16'h0000, "0.0 * 1.0 = 0.0");
        run_test(16'h3F80, 16'h0000, 16'h0000, "1.0 * 0.0 = 0.0");
        run_test(16'h3F00, 16'h4000, 16'h3F80, "0.5 * 2.0 = 1.0");
        run_test(16'hBF80, 16'h3F80, 16'hBF80, "(-1.0) * 1.0 = -1.0");
        run_test(16'hBF80, 16'hBF80, 16'h3F80, "(-1.0) * (-1.0) = 1.0");
        run_test(16'h3E80, 16'h3E80, 16'h3D80, "0.25 * 0.25 = 0.0625");

        // Pipeline stress test - multiple rapid inputs
        $display("Pipeline stress test - rapid inputs:");
        repeat(3) begin
            @(posedge clk);
            valid_in = 1;
            bf16_a = 16'h3F80;
            bf16_b = 16'h3F00;
        end
        @(posedge clk);
        valid_in = 0;
        
        // Wait for all outputs
        repeat(10) @(posedge clk);
        $display("Stress test completed");

        // Edge cases
        $display("Edge case tests:");
        run_test(16'h7F80, 16'h3F80, 16'h7F80, "Infinity * 1.0 = Infinity");
        run_test(16'hFF80, 16'h3F80, 16'hFF80, "-Infinity * 1.0 = -Infinity");

        // Summary
        $display("==== Test Summary ====");
        $display("Total tests: %d", test_count);
        $display("Passed: %d", pass_count);
        $display("Failed: %d", fail_count);
        
        if (fail_count == 0) begin
            $display("ALL TESTS PASSED!");
        end else begin
            $display("SOME TESTS FAILED!");
        end

        repeat(5) @(posedge clk);
        $finish;
    end

    // Monitor for debugging
    initial begin
        $monitor("Time=%t, clk=%b, rst_n=%b, valid_in=%b, A=%h, B=%h, valid_out=%b, result=%h",
                 $time, clk, rst_n, valid_in, bf16_a, bf16_b, valid_out, result);
    end

    // Timeout protection
    initial begin
        #(CLK_PERIOD * 500);
        $display("TIMEOUT - Test did not complete");
        $finish;
    end

endmodule 