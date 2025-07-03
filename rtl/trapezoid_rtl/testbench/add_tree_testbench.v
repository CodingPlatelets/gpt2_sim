// Testbench for Add Tree
// Tests the tree structure for parallel addition with multiple PEs
`timescale 1ns/1ps

module add_tree_testbench;

    // Parameters
    parameter PE_NUM = 4;
    parameter M = 4;
    parameter N = 8;
    parameter TREE_LEVELS = 2; // log2(PE_NUM)
    parameter CLK_PERIOD = 10; // 100MHz

    // Inputs
    reg clk;
    reg rst_n;
    reg valid_in;
    reg [PE_NUM*32-1:0] input_map_queue;
    reg [PE_NUM*16-1:0] input_indices;
    reg [PE_NUM*16-1:0] input_values;
    reg [PE_NUM-1:0] input_valid_mask;

    // Outputs
    wire valid_out;
    wire [15:0] output_index;
    wire [15:0] output_value;
    wire output_update_valid;

    // Test tracking
    integer test_count, pass_count, fail_count;

    // Instantiate the Unit Under Test (UUT)
    add_tree #(
        .PE_NUM(PE_NUM),
        .M(M),
        .N(N),
        .TREE_LEVELS(TREE_LEVELS)
    ) uut (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .input_map_queue(input_map_queue),
        .input_indices(input_indices),
        .input_values(input_values),
        .input_valid_mask(input_valid_mask),
        .valid_out(valid_out),
        .output_index(output_index),
        .output_value(output_value),
        .output_update_valid(output_update_valid)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    // Task for setting up inputs
    task setup_inputs;
        input [15:0] index0, index1, index2, index3;
        input [15:0] value0, value1, value2, value3;
        input [3:0] valid_mask;
        
        begin
            input_indices = {index3, index2, index1, index0};
            input_values = {value3, value2, value1, value0};
            input_valid_mask = valid_mask;
            
            // Setup input_map_queue (simplified for testing)
            input_map_queue = {32'h00000000, 32'h00000000, 32'h00000000, 32'h00000000};
        end
    endtask

    // Task for running a test
    task run_test;
        input [15:0] index0, index1, index2, index3;
        input [15:0] value0, value1, value2, value3;
        input [3:0] valid_mask;
        input [256*8-1:0] test_name;
        
        integer cycle_count;
        
        begin
            $display("Running test: %s", test_name);
            $display("  Indices: [%h, %h, %h, %h]", index0, index1, index2, index3);
            $display("  Values:  [%h, %h, %h, %h]", value0, value1, value2, value3);
            $display("  Valid mask: %b", valid_mask);
            
            setup_inputs(index0, index1, index2, index3, value0, value1, value2, value3, valid_mask);
            
            @(posedge clk);
            valid_in = 1;
            @(posedge clk);
            valid_in = 0;
            
            // Wait for output (pipeline latency)
            cycle_count = 0;
            while (!valid_out && cycle_count < 20) begin
                @(posedge clk);
                cycle_count = cycle_count + 1;
            end
            
            if (valid_out) begin
                $display("  Output after %d cycles:", cycle_count);
                $display("    Index: %h, Value: %h, Update valid: %b", 
                         output_index, output_value, output_update_valid);
                $display("  PASS");
                pass_count = pass_count + 1;
            end else begin
                $display("  FAIL - No valid output after %d cycles", cycle_count);
                fail_count = fail_count + 1;
            end
            
            test_count = test_count + 1;
            $display("");
            
            // Wait for pipeline to settle
            repeat(5) @(posedge clk);
        end
    endtask

    // Test sequence
    initial begin
        $dumpfile("add_tree_test.vcd");
        $dumpvars(0, add_tree_testbench);

        // Initialize
        test_count = 0;
        pass_count = 0;
        fail_count = 0;
        
        rst_n = 0;
        valid_in = 0;
        input_map_queue = 0;
        input_indices = 0;
        input_values = 0;
        input_valid_mask = 0;

        // Reset sequence
        #(CLK_PERIOD*5);
        rst_n = 1;
        #(CLK_PERIOD*2);

        $display("==== Add Tree Testbench ====");
        $display("Parameters: PE_NUM=%d, M=%d, N=%d, TREE_LEVELS=%d", PE_NUM, M, N, TREE_LEVELS);
        $display("");

        // Test case 1: Two same indices to be added
        run_test(
            16'h0001, 16'h0001, 16'h0002, 16'h0003,  // indices
            16'h3F80, 16'h3F80, 16'h3F00, 16'h3E80,  // values (1.0, 1.0, 0.5, 0.25)
            4'b1111,                                   // all valid
            "Same indices addition"
        );

        // Test case 2: All different indices (no addition)
        run_test(
            16'h0001, 16'h0002, 16'h0003, 16'h0004,  // indices
            16'h3F80, 16'h3F80, 16'h3F00, 16'h3E80,  // values
            4'b1111,                                   // all valid
            "All different indices"
        );

        // Test case 3: Partial valid inputs
        run_test(
            16'h0001, 16'h0002, 16'h0003, 16'h0004,  // indices
            16'h3F80, 16'h3F80, 16'h3F00, 16'h3E80,  // values
            4'b1100,                                   // only first two valid
            "Partial valid inputs"
        );

        // Test case 4: Multiple same indices
        run_test(
            16'h0001, 16'h0001, 16'h0001, 16'h0002,  // indices
            16'h3F80, 16'h3F80, 16'h3F80, 16'h3F00,  // values (1.0, 1.0, 1.0, 0.5)
            4'b1111,                                   // all valid
            "Multiple same indices"
        );

        // Test case 5: Zero values
        run_test(
            16'h0001, 16'h0001, 16'h0002, 16'h0002,  // indices
            16'h0000, 16'h3F80, 16'h0000, 16'h3F00,  // values (0.0, 1.0, 0.0, 0.5)
            4'b1111,                                   // all valid
            "Zero values"
        );

        // Test case 6: Single valid input
        run_test(
            16'h0001, 16'h0002, 16'h0003, 16'h0004,  // indices
            16'h3F80, 16'h3F80, 16'h3F00, 16'h3E80,  // values
            4'b0001,                                   // only first valid
            "Single valid input"
        );

        // Pipeline stress test
        $display("--- Pipeline Stress Test ---");
        $display("Multiple rapid inputs:");
        
        repeat(3) begin
            setup_inputs(16'h0001, 16'h0001, 16'h0002, 16'h0002,
                        16'h3F80, 16'h3F00, 16'h3E80, 16'h3E80,
                        4'b1111);
            
            @(posedge clk);
            valid_in = 1;
            @(posedge clk);
            valid_in = 0;
            
            // Short wait between inputs
            repeat(2) @(posedge clk);
        end
        
        // Wait for all outputs
        repeat(20) @(posedge clk);
        $display("Stress test completed");

        // Test case 7: Boundary index values
        run_test(
            16'h0000, 16'h0000, 16'hFFFF, 16'hFFFF,  // indices (min and max)
            16'h3F80, 16'h3F80, 16'h3F00, 16'h3F00,  // values
            4'b1111,                                   // all valid
            "Boundary index values"
        );

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
        $monitor("Time=%t, clk=%b, rst_n=%b, valid_in=%b, valid_out=%b, output_index=%h, output_value=%h",
                 $time, clk, rst_n, valid_in, valid_out, output_index, output_value);
    end

    // Timeout protection
    initial begin
        #(CLK_PERIOD * 1000);
        $display("TIMEOUT - Test did not complete");
        $finish;
    end

endmodule 