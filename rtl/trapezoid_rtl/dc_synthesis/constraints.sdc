# Synopsys Design Constraints (SDC) for Trapezoid Pipeline
# 14nm process constraints

# Set units
set_units -time ns -resistance MOhm -capacitance fF -voltage V -current uA

# Clock definition - 1GHz target frequency (1ns period)
create_clock -name "clk" -period 1.0 [get_ports clk]
set_clock_uncertainty 0.05 [get_clocks clk]
set_clock_transition 0.02 [get_clocks clk]

# Reset signal
set_false_path -from [get_ports rst_n]

# Input constraints
set_input_delay 0.2 -clock clk [all_inputs]
set_input_transition 0.02 [all_inputs]
set_driving_cell -lib_cell BUFX2M [all_inputs]

# Output constraints  
set_output_delay 0.2 -clock clk [all_outputs]
set_load 0.01 [all_outputs]

# Environmental conditions for 14nm
set_operating_conditions -max_library tcbn14ffcllvt -max ss_1p08v_125c
set_wire_load_model -name "Small" -library tcbn14ffcllvt

# Area constraints
set_max_area 0

# Power optimization
set_dynamic_optimization true

# Don't touch clock tree
set_dont_touch_network [get_clocks clk]

# Group related paths
group_path -name "input_to_reg" -from [all_inputs] -to [all_registers]
group_path -name "reg_to_reg" -from [all_registers] -to [all_registers]  
group_path -name "reg_to_output" -from [all_registers] -to [all_outputs]

# Set critical path priorities
set_critical_range 0.1 [current_design]

# Memory and register optimization
set_flatten true -effort medium
set_structure -boolean_optimization true
set_structure -timing_optimization true 