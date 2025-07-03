// Testbench for Compute Units
// Tests multiply_unit, add_unit, and mac_unit modules
`timescale 1ns/1ps

module compute_units_testbench;

    // Parameters
    parameter CLK_PERIOD = 10; // 100MHz

    // Common signals
    reg clk;
    reg rst_n;
    integer test_count, pass_count, fail_count;

    // Multiply unit signals
    reg mul_input_valid;
    reg [15:0] mul_input1, mul_input2;
    reg [15:0] mul_sft_index;
    wire mul_valid_out;
    wire [15:0] mul_result;
    wire [15:0] mul_index_out;

    // Add unit signals  
    reg add_input_valid;
    reg [15:0] add_input1, add_input2;
    reg [15:0] add_sft_index;
    wire add_valid_out;
    wire [15:0] add_result;
    wire [15:0] add_index_out;

    // MAC unit signals
    reg mac_input_valid;
    reg [15:0] mac_input1, mac_input2;
    reg [15:0] mac_initial_acc;
    reg mac_set_initial_acc;
    wire mac_valid_out;
    wire [15:0] mac_result;

    // Instantiate the Units Under Test
    multiply_unit mul_uut (
        .clk(clk),
        .rst_n(rst_n),
        .input_valid(mul_input_valid),
        .input1(mul_input1),
        .input2(mul_input2),
        .sft_index(mul_sft_index),
        .valid_out(mul_valid_out),
        .result(mul_result),
        .index_out(mul_index_out)
    );

    add_unit add_uut (
        .clk(clk),
        .rst_n(rst_n),
        .input_valid(add_input_valid),
        .input1(add_input1),
        .input2(add_input2),
        .sft_index(add_sft_index),
        .valid_out(add_valid_out),
        .result(add_result),
        .index_out(add_index_out)
    );

    mac_unit mac_uut (
        .clk(clk),
        .rst_n(rst_n),
        .input_valid(mac_input_valid),
        .input1(mac_input1),
        .input2(mac_input2),
        .initial_acc(mac_initial_acc),
        .set_initial_acc(mac_set_initial_acc),
        .valid_out(mac_valid_out),
        .result(mac_result)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    // Task for testing multiply unit
    task test_multiply_unit;
        input [15:0] a_val;
        input [15:0] b_val;
        input [15:0] index_val;
        input [15:0] expected_result;
        input [15:0] expected_index;
        input [256*8-1:0] test_name;
        
        reg found_result;
        integer timeout_counter;
        
        begin
            $display("Testing Multiply Unit: %s", test_name);
            $display("  Input: %h * %h, index=%h", a_val, b_val, index_val);
            
            @(posedge clk);
            mul_input_valid = 1;
            mul_input1 = a_val;
            mul_input2 = b_val;
            mul_sft_index = index_val;
            @(posedge clk);
            mul_input_valid = 0;
            
            // Wait for pipeline output with timeout
            found_result = 0;
            timeout_counter = 0;
            
            while (!found_result && timeout_counter < 20) begin
                @(posedge clk);
                timeout_counter = timeout_counter + 1;
                
                if (mul_valid_out) begin
                    found_result = 1;
                    $display("  Result: %h, Index: %h (cycle %d)", mul_result, mul_index_out, timeout_counter);
                    if (mul_result == expected_result && mul_index_out == expected_index) begin
                        $display("  PASS");
                        pass_count = pass_count + 1;
                    end else begin
                        $display("  FAIL - Expected result=%h, index=%h", expected_result, expected_index);
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

    // Task for testing add unit
    task test_add_unit;
        input [15:0] a_val;
        input [15:0] b_val;
        input [15:0] index_val;
        input [15:0] expected_result;
        input [15:0] expected_index;
        input [256*8-1:0] test_name;
        
        reg found_result;
        integer timeout_counter;
        
        begin
            $display("Testing Add Unit: %s", test_name);
            $display("  Input: %h + %h, index=%h", a_val, b_val, index_val);
            
            @(posedge clk);
            add_input_valid = 1;
            add_input1 = a_val;
            add_input2 = b_val;
            add_sft_index = index_val;
            @(posedge clk);
            add_input_valid = 0;
            
            // Wait for pipeline output with timeout
            found_result = 0;
            timeout_counter = 0;
            
            while (!found_result && timeout_counter < 20) begin
                @(posedge clk);
                timeout_counter = timeout_counter + 1;
                
                if (add_valid_out) begin
                    found_result = 1;
                    $display("  Result: %h, Index: %h (cycle %d)", add_result, add_index_out, timeout_counter);
                    if (add_result == expected_result && add_index_out == expected_index) begin
                        $display("  PASS");
                        pass_count = pass_count + 1;
                    end else begin
                        $display("  FAIL - Expected result=%h, index=%h", expected_result, expected_index);
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

    // Task for testing MAC unit
    task test_mac_unit;
        input [15:0] a_val;
        input [15:0] b_val;
        input [15:0] acc_val;
        input [15:0] expected_result;
        input [256*8-1:0] test_name;
        
        reg found_result;
        integer timeout_counter;
        
        begin
            $display("Testing MAC Unit: %s", test_name);
            $display("  Input: %h * %h + %h", a_val, b_val, acc_val);
            
            // Set initial accumulator
            @(posedge clk);
            mac_set_initial_acc = 1;
            mac_initial_acc = acc_val;
            @(posedge clk);
            mac_set_initial_acc = 0;
            
            // Start MAC operation
            @(posedge clk);
            mac_input_valid = 1;
            mac_input1 = a_val;
            mac_input2 = b_val;
            @(posedge clk);
            mac_input_valid = 0;
            
            // Wait for pipeline output with timeout
            found_result = 0;
            timeout_counter = 0;
            
            while (!found_result && timeout_counter < 25) begin
                @(posedge clk);
                timeout_counter = timeout_counter + 1;
                
                if (mac_valid_out) begin
                    found_result = 1;
                    $display("  Result: %h (cycle %d)", mac_result, timeout_counter);
                    if (mac_result == expected_result) begin
                        $display("  PASS");
                        pass_count = pass_count + 1;
                    end else begin
                        $display("  FAIL - Expected %h", expected_result);
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
        $dumpfile("compute_units_test.vcd");
        $dumpvars(0, compute_units_testbench);

        // Initialize
        test_count = 0;
        pass_count = 0;
        fail_count = 0;
        
        rst_n = 0;
        
        // Initialize multiply unit signals
        mul_input_valid = 0;
        mul_input1 = 16'h0000;
        mul_input2 = 16'h0000;
        mul_sft_index = 16'h0000;
        
        // Initialize add unit signals
        add_input_valid = 0;
        add_input1 = 16'h0000;
        add_input2 = 16'h0000;
        add_sft_index = 16'h0000;
        
        // Initialize MAC unit signals
        mac_input_valid = 0;
        mac_input1 = 16'h0000;
        mac_input2 = 16'h0000;
        mac_initial_acc = 16'h0000;
        mac_set_initial_acc = 0;

        // Reset sequence
        #(CLK_PERIOD*5);
        rst_n = 1;
        #(CLK_PERIOD*2);

        $display("==== Compute Units Testbench ====");
        $display("Testing multiply_unit, add_unit, and mac_unit");
        $display("");

        // Test Multiply Unit
        $display("--- Multiply Unit Tests ---");
        test_multiply_unit(16'h3F80, 16'h4000, 16'h0001, 16'h4000, 16'h0001, "1.0 * 2.0 = 2.0");
        test_multiply_unit(16'h3F00, 16'h3F00, 16'h0002, 16'h3E80, 16'h0002, "0.5 * 0.5 = 0.25");
        test_multiply_unit(16'h0000, 16'h3F80, 16'h0003, 16'h0000, 16'h0003, "0.0 * 1.0 = 0.0");

        // Test Add Unit
        $display("--- Add Unit Tests ---");
        test_add_unit(16'h3F80, 16'h3F80, 16'h0010, 16'h4000, 16'h0010, "1.0 + 1.0 = 2.0");
        test_add_unit(16'h3F00, 16'h3F00, 16'h0011, 16'h3F80, 16'h0011, "0.5 + 0.5 = 1.0");
        test_add_unit(16'h3F80, 16'hBF80, 16'h0012, 16'h0000, 16'h0012, "1.0 + (-1.0) = 0.0");

        // Test MAC Unit
        $display("--- MAC Unit Tests ---");
        test_mac_unit(16'h3F80, 16'h4000, 16'h3F80, 16'h40C0, "(1.0 * 2.0) + 1.0 = 3.0");
        test_mac_unit(16'h3F00, 16'h3F00, 16'h3F00, 16'h3F00, "(0.5 * 0.5) + 0.5 = 0.75");
        test_mac_unit(16'h0000, 16'h3F80, 16'h4000, 16'h4000, "(0.0 * 1.0) + 2.0 = 2.0");

        // Pipeline stress tests
        $display("--- Pipeline Stress Tests ---");
        
        // Multiple rapid multiply operations
        $display("Multiply unit stress test:");
        repeat(3) begin
            @(posedge clk);
            mul_input_valid = 1;
            mul_input1 = 16'h3F80;
            mul_input2 = 16'h3F80;
            mul_sft_index = 16'h0100;
        end
        @(posedge clk);
        mul_input_valid = 0;
        repeat(15) @(posedge clk);
        
        // Multiple rapid add operations
        $display("Add unit stress test:");
        repeat(3) begin
            @(posedge clk);
            add_input_valid = 1;
            add_input1 = 16'h3F80;
            add_input2 = 16'h3F00;
            add_sft_index = 16'h0200;
        end
        @(posedge clk);
        add_input_valid = 0;
        repeat(15) @(posedge clk);

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

    // Timeout protection
    initial begin
        #(CLK_PERIOD * 1000);
        $display("TIMEOUT - Test did not complete");
        $finish;
    end

endmodule 