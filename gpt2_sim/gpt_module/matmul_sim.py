from trapezoid_module import TrapezoidPipeline
from scipy.sparse import csr_matrix
from hbm import store_csr_in_simple_blocks_fast
from bf16_module.utils import convert_through_pipeline

# A and B must to be bf16
class Matmul:
    def __init__(self, PE_num, PE_rows, data_num_per_cycle=256):
        self.PE_num = PE_num
        self.PE_rows = PE_rows
        self.data_num_per_cycle = data_num_per_cycle
        self.cycles = 0
        self.hbm_data = {}
        
        self.trapezoid_rows = [TrapezoidPipeline(1, 1, 1, self.PE_num) for _ in range(PE_rows)]

    def _convert_hbm_data_to_bf16(self, B_data_list):
        bf16_B_data_list = []
        for B_data in B_data_list:
            values_B = B_data.get("values", [])
            bf16_values_B = []
            for val in values_B:
                if val != 0:
                    bf16_values_B.append(convert_through_pipeline(float(val)))
                else:
                    bf16_values_B.append(0)
            bf16_B_data = {
                "values": bf16_values_B,
                "col_indices": B_data.get("col_indices", []),
                "row_ptr": B_data.get("row_ptr", []),
                "row_start_index": B_data.get("row_start_index", 0)
            }
            bf16_B_data_list.append(bf16_B_data)
        return bf16_B_data_list

    def load_from_hbm(self, matrix):

        block_from_hbm = store_csr_in_simple_blocks_fast(csr_matrix(matrix.T), self.data_num_per_cycle)
        self.hbm_data = self._convert_hbm_data_to_bf16(block_from_hbm)
    
    # A must to be bf16, B from hbm
    def forward(self, A, M, K, N, test=False):
        for trape in self.trapezoid_rows:
            trape.reset(M, K, N)
        main_trap = self.trapezoid_rows[0]
        result = main_trap.run_pipeline_hbm_multi([A], self.hbm_data, self.trapezoid_rows, -1)
        if test:
            return result["combined_c_matrix"]
        return result["combined_c_matrix_bf16"]
        
    





    


