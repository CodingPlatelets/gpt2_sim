import numpy as np
from bf16_sim import BF16AddPipeline, BF16MultiplyPipeline, FP32toBF16Pipeline
import struct

# --- Helper Functions ---
def bf16_to_float(bf16):
    """Converts a BF16 (uint16) to a Python float."""
    if bf16 is None: return None # Handle None input
    try:
        fp32_bits = np.uint32(bf16) << 16
        return struct.unpack('>f', struct.pack('>I', fp32_bits))[0]
    except Exception as e:
        print(f"Error converting bf16 value {bf16} to float: {e}")
        return float('nan') # Return NaN on error

def bf16_to_float_block(bf16_block):
    """Converts a list/array of BF16 (uint16) to a list of Python floats."""
    if bf16_block is None: return None
    return [bf16_to_float(bf16) for bf16 in bf16_block]

def convert_through_pipeline(value):
    """Converts a Python float to BF16 (uint16) using the simulation pipeline."""
    # Ensure input is float
    try:
        float_value = float(value)
    except (ValueError, TypeError):
        print(f"Warning: Could not convert {value} to float for BF16 conversion. Using 0.")
        float_value = 0.0

    temp_pipeline = FP32toBF16Pipeline()
    # Assuming run_simulation takes a list of (value, valid) tuples
    temp_pipeline.run_simulation([(float_value, True)], print_states=False)
    # Assuming outputs is a list of dicts with a 'bf16' key
    return temp_pipeline.outputs[0]["bf16"] if temp_pipeline.outputs else np.uint16(0)

# --- Pipeline Components ---

class MACUnit:
    """Simulates a MAC unit with an internal BF16 multiply pipeline."""
    def __init__(self):
        self.multliply_pipeline = BF16MultiplyPipeline()
        # State: (a, b, valid_flag)
        # valid_flag = True means this is new input to be processed
        # valid_flag = False means the input was processed, just keep state for potential future empty ticks
        self.state = None

    def is_busy(self):
        """Checks if the unit has pending state or the internal pipeline is active."""
        mp = self.multliply_pipeline
        # Check if internal pipeline has any valid stage
        pipeline_active = mp.stage1_valid or mp.stage2_valid or mp.stage3_valid or mp.stage4_valid or bool(mp.outputs)
        return self.state is not None or pipeline_active

    def get_input(self, a, b):
        """Sets the input state for the next tick."""
        # Input values a, b are expected to be uint16 (BF16)
        self.state = (a, b, True) # Mark as new valid input

    def tick(self):
        """Advances the internal multiply pipeline by one cycle."""
        output_value = None
        valid_input_this_cycle = False

        if self.state: # Check if there is state to process
            a, b, valid_flag = self.state
            if valid_flag: # If it's new input, feed it to the pipeline
                self.multliply_pipeline.clock_cycle(bf16_a=a, bf16_b=b, valid=True)
                valid_input_this_cycle = True
                # Mark state as consumed but keep values for potential empty ticks
                self.state = (a, b, False)
            else:
                # State already consumed, just tick the pipeline without new input
                self.multliply_pipeline.clock_cycle(valid=False)
        else:
            # No state, tick the pipeline without new input
            self.multliply_pipeline.clock_cycle(valid=False)

        # Check if the pipeline produced an output in this cycle
        # Assuming clock_cycle returns a dict with 'valid_output' and 'output_value'
        # Or check the outputs list if clock_cycle doesn't return directly
        # Modification: Check the outputs list directly as per original bf16_sim structure
        if self.multliply_pipeline.outputs:
             # Assuming outputs list stores results, pop the oldest
            output_value = self.multliply_pipeline.outputs.pop(0)
            # Optional: If output is produced, maybe clear the state?
            # self.state = None # This might prevent empty ticks if needed later

        return output_value # Returns uint16 (BF16) or None

class MACLine:
    """Simulates a line of MAC units."""
    def __init__(self, mac_width):
        self.mac_width = mac_width
        self.macs = [MACUnit() for _ in range(mac_width)]
        # No internal input queue; input is set via get_input by PipelineSimulator

    def get_input(self, input_data):
        """Sets the input state for all MAC units for the next tick."""
        # input_data should be [a_block, b_block] where blocks are lists/arrays of uint16
        if input_data and len(input_data) == 2:
            a_block, b_block = input_data
            for i, mac in enumerate(self.macs):
                if i < len(a_block) and i < len(b_block):
                    mac.get_input(a_block[i], b_block[i])
                else:
                    # No valid input for this MAC unit, clear its state
                    mac.state = None
        else:
            # No valid input data provided, clear state for all MAC units
            for mac in self.macs:
                mac.state = None

    def reset(self):
        """Resets all MAC units."""
        for mac in self.macs:
            mac.state = None
            mac.multliply_pipeline.reset() # Also reset internal pipeline

    def is_active(self):
        """Checks if any MAC unit is busy."""
        return any(mac.is_busy() for mac in self.macs)

    def tick(self):
        """Advances all MAC units by one cycle and collects results."""
        results = []
        # Unconditionally tick every MAC unit
        for mac in self.macs:
            r = mac.tick() # r is uint16 (BF16) or None
            if r is not None:
                results.append(r)

        # Return list of uint16 results, or empty list if none
        return results

class AdderTree:
    """Simulates a two-stage adder tree using BF16 add pipelines."""
    def __init__(self):
        # Input buffer for values coming from MACLine in the current cycle
        self.values_in = []
        # Buffer for intermediate results from the first stage
        self.intermediate_results = []
        # Two adders for the first stage (4 inputs -> 2 outputs)
        self.add_pipeline_lower = [BF16AddPipeline(), BF16AddPipeline()]
        # One adder for the second stage (2 inputs -> 1 output)
        self.add_pipeline_upper = BF16AddPipeline()

    def add_values(self, values):
        """Receives new values (list of uint16) from MACLine."""
        # Input values are expected to be uint16 (BF16)
        if values:
            self.values_in.extend(values)

    def is_active(self):
        """Checks if there are pending inputs, intermediate results, or active internal pipelines."""
        lower_pipelines_active = any(
            p.stage1_valid or p.stage2_valid or p.stage3_valid or p.stage4_valid or bool(p.outputs)
            for p in self.add_pipeline_lower
        )
        upper_pipeline_active = (
            self.add_pipeline_upper.stage1_valid or self.add_pipeline_upper.stage2_valid or
            self.add_pipeline_upper.stage3_valid or self.add_pipeline_upper.stage4_valid or
            bool(self.add_pipeline_upper.outputs)
        )
        return bool(self.values_in) or bool(self.intermediate_results) or lower_pipelines_active or upper_pipeline_active

    def reset(self):
        """Resets the adder tree state and internal pipelines."""
        self.values_in = []
        self.intermediate_results = []
        for p in self.add_pipeline_lower:
            p.reset()
        self.add_pipeline_upper.reset()

    def tick_first_stage(self):
        """Advances the first stage adders."""
        new_intermediate_results = []

        # Prepare pairs for the lower adders from values_in
        inputs_for_lower = []
        temp_values = self.values_in[:] # Work on a copy
        while len(temp_values) >= 2:
            inputs_for_lower.append((temp_values.pop(0), temp_values.pop(0)))
        # Clear the input buffer as values are paired (or left if odd)
        self.values_in = temp_values # Keep the potential odd one out

        # Tick the lower adders
        for i, adder in enumerate(self.add_pipeline_lower):
            if i < len(inputs_for_lower):
                val1, val2 = inputs_for_lower[i]
                adder.clock_cycle(bf16_a=val1, bf16_b=val2, valid=True)
            else:
                # No new input pair for this adder, tick with valid=False
                adder.clock_cycle(valid=False)

            # Collect output if available
            if adder.outputs:
                new_intermediate_results.append(adder.outputs.pop(0))

        # Add newly produced intermediate results to the buffer
        if new_intermediate_results:
            self.intermediate_results.extend(new_intermediate_results)

        # Return value indicates if new intermediate results were generated *this cycle*
        # This might not be the best signal for PipelineSimulator, but reflects activity.
        return bool(new_intermediate_results)

    def tick_second_stage(self):
        """Advances the second stage adder."""
        final_result = None

        # Check if there are enough intermediate results for the upper adder
        if len(self.intermediate_results) >= 2:
            val1 = self.intermediate_results.pop(0)
            val2 = self.intermediate_results.pop(0)
            self.add_pipeline_upper.clock_cycle(bf16_a=val1, bf16_b=val2, valid=True)
        else:
            # Not enough inputs, tick with valid=False
            self.add_pipeline_upper.clock_cycle(valid=False)

        # Collect final output if available
        if self.add_pipeline_upper.outputs:
            final_result = self.add_pipeline_upper.outputs.pop(0)

        return final_result # Returns uint16 (BF16) or None

# --- Status Tracking ---

class PipelineStatus:
    """Records the status of the pipeline at each cycle."""
    def __init__(self):
        self.cycles = []

    def add_cycle(self, cycle_num, status_dict):
        """Adds a status record for a given cycle."""
        self.cycles.append({
            "cycle": cycle_num,
            **status_dict
        })

    def print_status(self, detailed=False):
        """Prints the recorded pipeline history."""
        print("\n===== Pipeline Execution History =====")
        if not self.cycles:
            print("No history recorded.")
            return

        for cycle_data in self.cycles:
            cycle_num = cycle_data['cycle']
            print(f"\nCycle {cycle_num}:")

            # Input Info
            input_log = cycle_data.get("input")
            if input_log:
                coords = input_log.get("coords", "N/A")
                block_idx = input_log.get("block_idx", "N/A")
                print(f"  Input: C{coords} - Block {block_idx}")
            else:
                print("  Input: None (Draining)")

            # Activity Status
            print(f"  MACLine Active: {cycle_data.get('mac_line_active', 'N/A')}")
            print(f"  AdderTree Active: {cycle_data.get('adder_tree_active', 'N/A')}")
            print(f"  Coordinate Queue Length: {cycle_data.get('ijQueue_len', 'N/A')}")

            # Result Output
            result_log = cycle_data.get("result")
            if result_log:
                coords, value = result_log
                value_float = bf16_to_float(value) # Convert BF16 result to float for printing
                print(f"  Output: C{coords} -> Partial Sum {value_float} ({hex(value) if value is not None else 'None'})")

            # Completed Element (If tracked externally)
            completed_log = cycle_data.get("completed")
            if completed_log:
                 coords = completed_log.get("coords", "N/A")
                 value = completed_log.get("value", "N/A")
                 value_float = bf16_to_float(value) # Convert BF16 result to float
                 print(f"  Completed: C{coords} = {value_float} ({hex(value) if value is not None else 'None'})")

            # Detailed Debug Info
            if detailed and "debug_info" in cycle_data:
                print("  Debug Info:")
                for key, value in cycle_data["debug_info"].items():
                    print(f"    {key}: {value}")

# --- Main Pipeline Simulator ---

class PipelineSimulator:
    """Simulates the entire MAC pipeline including BF16 units."""
    def __init__(self, mac_width=4):
        self.mac_width = mac_width
        self.mac_line = MACLine(mac_width)
        self.adder_tree = AdderTree()
        self.clock_cycle = 0
        # Queue to store coordinates corresponding to data flowing through the pipeline
        self.ijQueue = []
        self.status = PipelineStatus()

    def process_block(self, a_block, b_block, coords, is_last_block):
        """
        Processes one clock cycle of the pipeline.
        Accepts new input blocks (a_block, b_block - lists of uint16) if available.
        Returns the final result (uint16) and its coordinates if produced this cycle.
        """
        # --- 1. Advance all internal pipelines unconditionally ---
        # Stage 4: Advance second stage adder and get potential final result
        final_result_bf16 = self.adder_tree.tick_second_stage() # uint16 or None

        # Stage 3: Advance first stage adders
        self.adder_tree.tick_first_stage() # Advances internally, results stored in intermediate_results

        # Stage 2: Advance MAC line
        mac_results_bf16 = self.mac_line.tick() # list of uint16

        # --- 2. Handle data flow between stages and new input ---
        output_coords = None
        if final_result_bf16 is not None:
            if self.ijQueue:
                output_coords = self.ijQueue.pop(0) # Match result with oldest coordinate
            else:
                # Should not happen if logic is correct, indicates coordinate tracking issue
                print(f"Warning: Pipeline produced result {final_result_bf16} but ijQueue is empty!")

        # Feed Stage 2 results (MAC outputs) into Stage 3 (Adder Tree input)
        if mac_results_bf16:
            self.adder_tree.add_values(mac_results_bf16)

        # Feed Stage 1 input (new data) into Stage 2 (MAC Line input)
        new_input_data = None
        if a_block is not None and b_block is not None:
            # a_block, b_block are expected to be lists/arrays of uint16
            new_input_data = [a_block, b_block]
            self.ijQueue.append(coords) # Add coordinates to the tracking queue

        # Set the input for the MAC Line for the *next* cycle's tick
        self.mac_line.get_input(new_input_data)

        # --- 3. Record Status ---
        post_status = {
            "mac_line_active": self.mac_line.is_active(),
            "adder_tree_active": self.adder_tree.is_active(),
            "ijQueue_len": len(self.ijQueue),
            "result": (output_coords, final_result_bf16) if final_result_bf16 is not None and output_coords is not None else None,
            "debug_info": {
                 "input_received": new_input_data is not None,
                 "mac_outputs_count": len(mac_results_bf16),
                 # Add more internal states if needed
            }
        }
        self.status.add_cycle(self.clock_cycle, post_status)

        # --- 4. Return result ---
        if final_result_bf16 is not None and output_coords is not None:
            return (output_coords, final_result_bf16) # Return (coords, uint16)
        else:
            return None

    def reset(self):
        """Resets the simulator and all components."""
        self.mac_line.reset()
        self.adder_tree.reset()
        self.clock_cycle = 0
        self.ijQueue = []
        self.status = PipelineStatus()

    def is_active(self):
        """Checks if any component of the pipeline is still processing data."""
        # Active if MAC line or Adder Tree is active, or if coordinates are still queued
        # (meaning data is still flowing towards the output)
        return self.mac_line.is_active() or self.adder_tree.is_active() or bool(self.ijQueue)

# --- Matrix Multiplication Function ---

def pipeline_matmul(A, B, verbose=False):
    """
    Performs matrix multiplication using the simulated BF16 pipeline.
    A, B are NumPy arrays (e.g., int or float).
    Returns C as a NumPy float32 array (results converted back from BF16).
    """
    assert A.shape[1] == B.shape[0], "Matrix dimensions mismatch"
    m, k = A.shape
    n = B.shape[1]
    # Initialize result matrix C with zeros (float32 for accumulation)
    C = np.zeros((m, n), dtype=np.float32)
    logs = [] # For external logging if needed

    sim = PipelineSimulator()

    input_queue = [] # Stores blocks to be fed into the pipeline

    # Prepare all input blocks
    for i in range(m):
        for j in range(n):
            a_row = A[i, :]
            b_col = B[:, j]
            k_len = len(a_row)

            for block_start in range(0, k_len, sim.mac_width):
                a_block_orig = a_row[block_start : block_start + sim.mac_width]
                b_block_orig = b_col[block_start : block_start + sim.mac_width]

                # Pad if necessary
                pad_len = sim.mac_width - len(a_block_orig)
                if pad_len > 0:
                    # Pad with 0.0 for float input before conversion
                    a_block_padded = np.pad(a_block_orig.astype(np.float32), (0, pad_len), constant_values=0.0)
                    b_block_padded = np.pad(b_block_orig.astype(np.float32), (0, pad_len), constant_values=0.0)
                else:
                    a_block_padded = a_block_orig.astype(np.float32)
                    b_block_padded = b_block_orig.astype(np.float32)

                # Convert padded blocks to BF16 (uint16)
                a_block_bf16 = [convert_through_pipeline(a) for a in a_block_padded]
                b_block_bf16 = [convert_through_pipeline(b) for b in b_block_padded]

                block_index = block_start // sim.mac_width
                is_last_block = (block_start + sim.mac_width >= k_len)

                input_queue.append({
                    "a_block": a_block_bf16, # Now uint16
                    "b_block": b_block_bf16, # Now uint16
                    "coords": (i, j),
                    "block_index": block_index,
                    "is_last_block": is_last_block # Note: is_last_block is not currently used by sim
                })

    last_coords = None # Keep track for draining phase

    # Main simulation loop
    while input_queue or sim.is_active():
        current_input_log = None
        completed_element_log = None # Placeholder if external tracking were used

        a_block_to_sim = None
        b_block_to_sim = None
        coords_to_sim = None
        is_last_to_sim = False # Default value

        if input_queue:
            data = input_queue.pop(0)
            a_block_to_sim = data["a_block"] # uint16 list/array
            b_block_to_sim = data["b_block"] # uint16 list/array
            coords_to_sim = data["coords"]
            is_last_to_sim = data["is_last_block"]
            last_coords = coords_to_sim

            current_input_log = {
                "coords": coords_to_sim,
                "block_idx": data["block_index"],
                "is_last": is_last_to_sim
            }
        else:
            # Draining phase: feed None to process_block
            coords_to_sim = last_coords # Use last known coords for context if needed
            is_last_to_sim = True # Assume end during draining
            current_input_log = None

        # Run one simulation cycle
        partial_result_info = sim.process_block(
            a_block_to_sim, b_block_to_sim, coords_to_sim, is_last_to_sim
        )

        # Process output from the pipeline
        if partial_result_info is not None:
            result_coords, result_value_bf16 = partial_result_info # result is uint16
            # Convert BF16 partial sum back to float for accumulation in C
            result_value_float = bf16_to_float(result_value_bf16)

            if result_value_float is not None and not np.isnan(result_value_float):
                 i, j = result_coords
                 # --- WARNING: Direct accumulation relies on timing coincidence ---
                 # This assumes partial sums for C[i][j] arrive contiguously enough
                 # before other elements' sums interfere significantly.
                 # This is NOT a robust way to handle accumulation in a general
                 # interleaved pipeline simulation. Use element_tracker for robustness.
                 C[i][j] += result_value_float
            else:
                 print(f"Warning: Received None or NaN partial result for C{result_coords} at cycle {sim.clock_cycle}")


        # Update status log (external to simulator's internal log)
        if verbose and hasattr(sim.status, "cycles") and sim.status.cycles:
            # Add input/completed info to the cycle recorded by the simulator
            latest_cycle_status = sim.status.cycles[-1]
            latest_cycle_status["input"] = current_input_log
            # 'completed' would require external tracking (like element_tracker)
            latest_cycle_status["completed"] = completed_element_log

        # Advance clock
        sim.clock_cycle += 1

    # --- Post-simulation ---
    # The loop using element_tracker is removed as it wasn't being updated.
    # If element_tracker were used, the final check/accumulation would happen here.

    if verbose:
        sim.status.print_status(detailed=False) # Print history from simulator's log
        # Print external logs if any were generated
        for log in logs:
            print(log)

    return C # Return float32 accumulated result

# --- Verification ---

def verify_result(m=4, k=7, n=5, random_seed=42, verbose=True):
    """
    Verifies the pipeline_matmul result against NumPy's direct calculation.
    Note: Comparison uses np.allclose due to potential precision differences
          between BF16 simulation and FP32 NumPy calculation.
    """
    np.random.seed(random_seed)
    # Generate integer matrices, easier to reason about
    A = np.random.randint(0, 10, size=(m, k)).astype(np.float32)
    B = np.random.randint(0, 10, size=(k, n)).astype(np.float32)

    print("Matrix A (FP32):")
    print(A)
    print("\nMatrix B (FP32):")
    print(B)

    # Calculate reference result using NumPy (FP32)
    print("\nCalculating C_np = A @ B (NumPy FP32)...")
    C_np = A @ B
    print("NumPy Result (FP32):")
    print(C_np)

    # Calculate result using the BF16 pipeline simulation
    print("\nCalculating C_mac using BF16 pipeline simulation...")
    C_mac = pipeline_matmul(A, B, verbose=verbose) # Returns float32
    print("\nPipeline Simulation Result (Accumulated Float32):")
    print(C_mac)

    # Compare results
    # Use np.allclose for floating-point comparison, allowing for small differences
    # Adjust atol (absolute tolerance) and rtol (relative tolerance) if needed
    print("\nComparing results...")
    if np.allclose(C_mac, C_np, atol=1e-1): # Increased tolerance for BF16
        print("Verification SUCCESSFUL: Results are close!")
    else:
        print("Verification FAILED: Results differ significantly!")
        print("Difference (C_mac - C_np):")
        print(C_mac - C_np)

# --- Main Execution ---

if __name__ == '__main__':
    # Example usage:
    verify_result(m=4, k=10, n=5, random_seed=42, verbose=False) # Set verbose=True for detailed pipeline trace