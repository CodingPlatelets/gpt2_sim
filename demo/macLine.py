import numpy as np

class MACLineMatrixMultiplier:
    def __init__(self, mac_width=4, verbose=False):
        """
        mac_width: number of MAC units in one line (e.g., 4 means can do 4 multiplies per step)
        verbose: if True, prints detailed computation steps
        """
        self.mac_width = mac_width
        self.verbose = verbose

    def _mac_line(self, a_block, b_block):
        """Simulate MAC line operation for two vectors of size mac_width."""
        return np.array([a_block[i] * b_block[i] for i in range(self.mac_width)])

    def _reduce_tree(self, values):
        """Simulate binary tree reduction."""
        steps = []
        while len(values) > 1:
            if len(values) % 2 != 0:
                values = np.append(values, 0)  # Pad if odd length
            new_values = []
            for i in range(0, len(values), 2):
                summed = values[i] + values[i+1]
                new_values.append(summed)
                if self.verbose:
                    print(f"Add {values[i]} + {values[i+1]} = {summed}")
            values = np.array(new_values)
            steps.append(values)
        return values[0] if len(values) == 1 else 0

    def multiply(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Perform matrix multiplication A x B using MAC line simulation and tree reduction.
        A: m x k matrix
        B: k x n matrix
        Returns: m x n matrix
        """
        m, k1 = A.shape
        k2, n = B.shape
        assert k1 == k2, "Inner dimensions of A and B must match"

        C = np.zeros((m, n))

        for i in range(m):
            for j in range(n):
                mac_outputs = []

                for block_start in range(0, k1, self.mac_width):
                    a_block = A[i, block_start:block_start + self.mac_width]
                    b_block = B[block_start:block_start + self.mac_width, j]

                    if len(a_block) < self.mac_width:
                        a_block = np.pad(a_block, (0, self.mac_width - len(a_block)))
                        b_block = np.pad(b_block, (0, self.mac_width - len(b_block)))

                    mac_result = self._mac_line(a_block, b_block)
                    mac_outputs.extend(mac_result)

                    if self.verbose:
                        print(f"[MAC] A[{i}, {block_start}:{block_start+self.mac_width}] × "
                              f"B[{block_start}:{block_start+self.mac_width}, {j}] => {mac_result}")

                # Tree-style reduction
                result = self._reduce_tree(np.array(mac_outputs))
                C[i][j] = result

                if self.verbose:
                    print(f"Final C[{i}][{j}] = {result}")
                    print("-" * 50)

        return C
    
def verify_result(m=4, k=7, n=5, random_seed=42):
    """
    生成随机矩阵 A (m x k) 和 B (k x n)，
    计算 MACLineMatrixMultiplier 计算的 C 与 NumPy 直接矩阵乘法的比较。
    """
    np.random.seed(random_seed)
    # A = np.random.randint(0, 10, size=(m, k))
    # B = np.random.randint(0, 10, size=(k, n))
    A = np.random.uniform(0, 10, size=(m, k))
    B = np.random.uniform(0, 10, size=(k, n))

    mac_mult = MACLineMatrixMultiplier(mac_width=4, verbose=False)
    C_mac = mac_mult.multiply(A, B)
    C_np = A @ B

    # print("随机生成矩阵 A:")
    # print(A)
    # print("\n随机生成矩阵 B:")
    # print(B)
    # print("\nMACLineMatrixMultiplier 计算结果:")
    # print(C_mac)
    # print("\nNumPy 直接计算结果:")
    # print(C_np)
    
    if np.allclose(C_mac, C_np):
        print("\n验证成功：结果一致！")
    else:
        print("\n验证失败：结果不一致！")
        
import struct
if __name__ == "__main__":
    print(struct.pack('>h', 1023))
    verify_result(10,200,30)


