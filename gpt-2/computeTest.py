import unittest
import torch
import numpy as np
from compute_sim import ComputeSimulator


class TestComputeSimulator(unittest.TestCase):
    def setUp(self):
        # Initialize the simulator before each test
        self.sim = ComputeSimulator()
        # Use small capacities for testing tiling
        self.sim.initialize(
            sram_capacity=1000,  # Small SRAM to test tiling
            ddr_capacity=10000,
            mac_units=2,
            add_units=2,
            mac_delay=1,
            add_delay=1
        )
        
        # Check if GPU is available for acceleration
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda':
            print("Running tests with GPU acceleration")
        else:
            print("Running tests on CPU (GPU not available)")
    
    def test_matrix_multiply_small(self):
        """Test matrix multiplication with small matrices that fit in SRAM"""
        A = torch.tensor([[1., 2.], [3., 4.]]).to(self.device)
        B = torch.tensor([[5., 6.], [7., 8.]]).to(self.device)
        expected = torch.matmul(A, B)  # Expected: [[19, 22], [43, 50]]
        
        # Load matrices into DDR
        self.sim.ddr_load("A", A)
        self.sim.ddr_load("B", B)
        
        # Perform multiplication
        result = self.sim.matrix_multiply("A", "B", "C")
        
        # Check result
        self.assertTrue(torch.allclose(result, expected))
        # Verify result is in DDR
        self.assertIn("C", self.sim.ddr)
    
    def test_matrix_multiply_large(self):
        """Test matrix multiplication with matrices too large to fit in SRAM together"""
        # Create matrices that will trigger tiling
        A = torch.rand(20, 15, device=self.device)
        B = torch.rand(15, 20, device=self.device)
        expected = torch.matmul(A, B)
        
        # Load matrices into DDR
        self.sim.ddr_load("A", A)
        self.sim.ddr_load("B", B)
        
        # Perform multiplication
        result = self.sim.matrix_multiply("A", "B", "C")
        
        # Check result
        self.assertTrue(torch.allclose(result, expected, rtol=1e-5, atol=1e-5))
        # Verify result is in DDR
        self.assertIn("C", self.sim.ddr)
    
    def test_matrix_multiply_large_gpu(self):
        """Test matrix multiplication with large matrices using GPU acceleration"""
        if self.device != 'cuda':
            self.skipTest("GPU not available for this test")
            
        # Create larger matrices to benefit from GPU acceleration
        A = torch.rand(200, 150, device=self.device)
        B = torch.rand(150, 200, device=self.device)
        expected = torch.matmul(A, B)
        
        # Load matrices into DDR
        self.sim.ddr_load("A", A)
        self.sim.ddr_load("B", B)
        
        # Perform multiplication
        result = self.sim.matrix_multiply("A", "B", "C")
        
        # Check result
        self.assertTrue(torch.allclose(result, expected, rtol=1e-5, atol=1e-5))
        # Verify result is in DDR
        self.assertIn("C", self.sim.ddr)
    
    def test_matrix_add_small(self):
        """Test matrix addition with small matrices that fit in SRAM"""
        A = torch.tensor([[1., 2.], [3., 4.]]).to(self.device)
        B = torch.tensor([[5., 6.], [7., 8.]]).to(self.device)
        expected = A + B  # Expected: [[6, 8], [10, 12]]
        
        # Load matrices into DDR
        self.sim.ddr_load("A", A)
        self.sim.ddr_load("B", B)
        
        # Perform addition
        result = self.sim.matrix_add("A", "B", "C")
        
        # Check result
        self.assertTrue(torch.allclose(result, expected))
        # Verify result is in DDR
        self.assertIn("C", self.sim.ddr)
    
    def test_matrix_add_large(self):
        """Test matrix addition with matrices too large to fit in SRAM together"""
        A = torch.rand(20, 20, device=self.device)
        B = torch.rand(20, 20, device=self.device)
        expected = A + B
        
        # Load matrices into DDR
        self.sim.ddr_load("A", A)
        self.sim.ddr_load("B", B)
        
        # Perform addition
        result = self.sim.matrix_add("A", "B", "C")
        
        # Check result
        self.assertTrue(torch.allclose(result, expected))
        # Verify result is in DDR
        self.assertIn("C", self.sim.ddr)
    
    def test_matrix_add_large_gpu(self):
        """Test matrix addition with large matrices using GPU acceleration"""
        if self.device != 'cuda':
            self.skipTest("GPU not available for this test")
            
        # Create larger matrices to benefit from GPU acceleration
        A = torch.rand(500, 500, device=self.device)
        B = torch.rand(500, 500, device=self.device)
        expected = A + B
        
        # Load matrices into DDR
        self.sim.ddr_load("A", A)
        self.sim.ddr_load("B", B)
        
        # Perform addition
        result = self.sim.matrix_add("A", "B", "C")
        
        # Check result
        self.assertTrue(torch.allclose(result, expected))
        # Verify result is in DDR
        self.assertIn("C", self.sim.ddr)
    
    def test_cycle_counting(self):
        """Test that computation cycles are counted correctly"""
        A = torch.tensor([[1., 2.], [3., 4.]]).to(self.device)
        B = torch.tensor([[5., 6.], [7., 8.]]).to(self.device)
        
        # Load matrices into DDR
        self.sim.ddr_load("A", A)
        self.sim.ddr_load("B", B)
        
        # Record cycles before computation
        cycles_before = self.sim.cycles
        
        # Perform multiplication
        self.sim.matrix_multiply("A", "B", "C")
        
        # Check that cycles have increased
        self.assertGreater(self.sim.cycles, cycles_before)
        
        # Calculate expected cycle increase
        # 2x2 matrices with a 2x2 inner dimension = 8 MAC operations
        # With 2 MAC units, should take at least 4 cycles
        self.assertGreaterEqual(self.sim.cycles - cycles_before, 4)
    
    def test_memory_management(self):
        """Test that matrices are properly moved between SRAM and DDR"""
        A = torch.rand(10, 10, device=self.device)
        B = torch.rand(10, 10, device=self.device)
        
        # Load matrices into DDR
        self.sim.ddr_load("A", A)
        self.sim.ddr_load("B", B)
        
        # Matrices should be in DDR but not in SRAM
        self.assertIn("A", self.sim.ddr)
        self.assertIn("B", self.sim.ddr)
        self.assertNotIn("A", self.sim.sram)
        self.assertNotIn("B", self.sim.sram)
        
        # Perform addition, which should load matrices into SRAM
        self.sim.matrix_add("A", "B", "C")
        
        # Result should be in DDR
        self.assertIn("C", self.sim.ddr)
        
        # A and B should have been loaded to SRAM during computation
        # Note: They might be evicted later, so we're checking access_log
        found_a_load = False
        found_b_load = False
        for log_entry in self.sim.access_log:
            if "LOAD A" in log_entry and "from DDR to SRAM" in log_entry:
                found_a_load = True
            if "LOAD B" in log_entry and "from DDR to SRAM" in log_entry:
                found_b_load = True
        
        self.assertTrue(found_a_load, "Matrix A wasn't loaded to SRAM")
        self.assertTrue(found_b_load, "Matrix B wasn't loaded to SRAM")
    
    def test_matrix_dimensions_mismatch(self):
        """Test that dimension mismatch is detected for matrix multiplication"""
        A = torch.tensor([[1., 2.], [3., 4.]]).to(self.device)
        B = torch.tensor([[5., 6., 7.], [8., 9., 10.], [11., 12., 13.]]).to(self.device)
        
        # Load matrices into DDR
        self.sim.ddr_load("A", A)
        self.sim.ddr_load("B", B)
        
        # Multiplication should raise ValueError due to dimension mismatch
        with self.assertRaises(ValueError):
            self.sim.matrix_multiply("A", "B", "C")

    def test_matrix_add_dimensions_mismatch(self):
        """Test that dimension mismatch is detected for matrix addition"""
        A = torch.tensor([[1., 2.], [3., 4.]]).to(self.device)
        B = torch.tensor([[5., 6., 7.], [8., 9., 10.]]).to(self.device)
        
        # Load matrices into DDR
        self.sim.ddr_load("A", A)
        self.sim.ddr_load("B", B)
        
        # Addition should raise ValueError due to dimension mismatch
        with self.assertRaises(ValueError):
            self.sim.matrix_add("A", "B", "C")

    def test_store_in_sram(self):
        """Test storing result in SRAM instead of DDR"""
        A = torch.tensor([[1., 2.], [3., 4.]]).to(self.device)
        B = torch.tensor([[5., 6.], [7., 8.]]).to(self.device)
        
        # Load matrices into DDR
        self.sim.ddr_load("A", A)
        self.sim.ddr_load("B", B)
        
        # Perform multiplication with store_in_ddr=False
        self.sim.matrix_multiply("A", "B", "C", store_in_ddr=False)
        
        # Result should be in SRAM but not in DDR
        self.assertIn("C", self.sim.sram)
        self.assertNotIn("C", self.sim.ddr)
    
    def test_gpu_performance(self):
        """Test performance improvement when using GPU acceleration"""
        if self.device != 'cuda':
            self.skipTest("GPU not available for this test")
        
        # Create large matrices
        size = 1000
        A = torch.rand(size, size, device=self.device)
        B = torch.rand(size, size, device=self.device)
        
        # Load matrices into DDR
        self.sim.ddr_load("A", A)
        self.sim.ddr_load("B", B)
        
        # Record cycles before operation
        cycles_before = self.sim.cycles
        
        # Perform operation
        self.sim.matrix_add("A", "B", "C")
        
        # Check cycles used
        cycles_used = self.sim.cycles - cycles_before
        
        # Just a sanity check that computation happened
        self.assertGreater(cycles_used, 0)
        print(f"GPU-accelerated matrix addition ({size}x{size}) took {cycles_used} cycles")


if __name__ == "__main__":
    unittest.main()
