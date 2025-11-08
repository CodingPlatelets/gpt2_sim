# File List for Trapezoid Pipeline Synthesis
# List all Verilog source files in compilation order

set file_list [list \
    "../bf16_add_pipeline.v" \
    "../bf16_multiply_pipeline.v" \
    "../compute_units.v" \
    "../add_tree.v" \
    "../trapezoid_pipeline.v" \
    "../trapezoid_pe_array.v" \
]

# Set top module
set top_module "trapezoid_pe_array"

# Set target library for 14nm process
# Note: Replace with actual 14nm library files
set target_library_list [list \
    "tcbn14ffcllvt.db" \
]

set link_library [list "*" $target_library_list]
set symbol_library [list "tcbn14ffcllvt.sdb"]

# Set work library
set work_lib "WORK" 