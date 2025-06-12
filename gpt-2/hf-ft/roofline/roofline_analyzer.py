import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches


class PerformanceAnalyzer:
    def __init__(self, d_model=1024, n_heads=16, d_ff=4096, vocab_size=50257, n_layers=24, fp16=True):
        """
        LLM 性能分析工具

        参数:
        d_model: 模型隐藏层维度 (默认: 1024)
        n_heads: 注意力头数 (默认: 16)
        d_ff: FFN层中间维度 (默认: 4096)
        vocab_size: 词表大小 (默认: 50257)
        n_layers: 层数 (默认: 24)
        fp16: 是否使用FP16精度 (默认: True)
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_ff = d_ff
        self.vocab_size = vocab_size
        self.bytes_per_value = 2 if fp16 else 4  # FP16精度，2字节/值，FP32精度，4字节/值
        self.n_layers = n_layers
        self.results = []

    def calculate_performance(self, B, seq_len, use_cache=True, elapsed=0.0):
        """
        计算指定配置下的性能指标

        参数:
        B: batch size
        seq_len: 序列长度
        use_cache: 是否使用缓存
        elapsed: 推理时间(ms)

        返回: (MHA层FLOPs, MHA层访存字节, 操作强度)
        """

        mem, flops = self.calculate_MHA(B, seq_len, use_cache)

        oi = flops / mem if mem > 0 else 0
        
        elapsed = elapsed / 1E3  # 转换为秒

        flops_s = flops / elapsed  

        result = {
            'B': B, 'seq_len': seq_len, 'use_cache': use_cache,
            'flops': flops,
            'memory_access': mem,
            'oi': oi,
            'elapsed': elapsed,
            'flops_s': flops_s
        }
        self.results.append(result)
        return flops, mem, oi, flops_s

    def calculate_MHA(self,B,seq_len,use_cache=True):
        S=seq_len
        K=seq_len
        if use_cache:
            S = 1
        # 1. QKV投影：[B,S,d_model] * [d_model,3*d_model] -> [B,S,3*d_model]
        # 读取输入x,读取权重Wq,Wk,Wv，写入Q/K/V
        mem_qkv = B * S * self.d_model
        mem_qkv += 3 * self.d_model * self.d_model  # 3*d_model^2
        mem_qkv += 3 * B * S * self.d_model
        mem_qkv *= self.bytes_per_value
        flops_qkv = 2 * B * S * self.d_model * 3 * self.d_model
        # 2. QK^T计算：[B,N_heads,S,d_head] * [B,N_heads,d_head,K] -> [B,N_heads,S,K]
        # 读取Q/K，写入S
        mem_qk = B * self.n_heads * S * self.d_head 
        mem_qk += B * self.n_heads * K * self.d_head #use_cache 时从KVcache中读取
        mem_qk += B * self.n_heads * S * K
        mem_qk *= self.bytes_per_value
        flops_qk = 2 * B * self.n_heads * S * self.d_head * K
        # 3. softmax [B,N_heads,S,K] -> [B,N_heads,S,K]
        # 读取S,写入P
        mem_softmax = B * self.n_heads * S * K
        mem_softmax += B * self.n_heads * S * K
        mem_softmax *= self.bytes_per_value
        flops_softmax = 5 * B * self.n_heads * S * K
        # 4. 注意力加权和：[B,N_heads,S,K] * [B,N_heads,K,d_head] -> [B,N_heads,S,d_head]
        # 读取P,读取V,写入O
        mem_pv = B * self.n_heads * S * K
        mem_pv += B * self.n_heads * K * self.d_head #use_cache 时从KVcache中读取
        mem_pv += B * self.n_heads * S * self.d_head
        mem_pv *= self.bytes_per_value
        flops_pv = 2 * B * self.n_heads * S * K * self.d_head
        # 5. 输出投影: [B,S,d_model] * [d_model,d_model] -> [B,S,d_model]
        # 读取O,读取权重W_o,写入O
        mem_out = B * S * self.d_model
        mem_out += self.d_model * self.d_model
        mem_out += B * S * self.d_model
        mem_out *= self.bytes_per_value
        flops_out = 2 * B * S * self.d_model * self.d_model
        # 6. 残差连接 忽略
        mem_acc = (mem_qkv + mem_qk + mem_softmax + mem_pv + mem_out)
        flops_acc = flops_qkv + flops_qk + flops_softmax + flops_pv + flops_out
        print(
            f"\nMHA Analysis Results, B={B}, seq_len={seq_len}, use_cache={use_cache}")
        print("=" * 120)
        print(
            f"{'Stage':<30} {'FLOPs (M)':>12} {'Mem (MB)':>12} {'OI(FLOPS/Byte)':>15}")
        print(
            f"{'QKV':<30} {flops_qkv/1e6:12.2f} {mem_qkv/1e6:12.3f} {flops_qkv/mem_qkv:12.3f}")
        print(
            f"{'QK':<30} {flops_qk/1e6:12.2f} {mem_qk/1e6:12.3f} {flops_qk/mem_qk:12.3f}")
        print(
            f"{'Softmax':<30} {flops_softmax/1e6:12.2f} {mem_softmax/1e6:12.3f} {flops_softmax/mem_softmax:12.3f}")
        print(
            f"{'PV':<30} {flops_pv/1e6:12.2f} {mem_pv/1e6:12.3f} {flops_pv/mem_pv:12.3f}")
        print(
            f"{'Out':<30} {flops_out/1e6:12.2f} {mem_out/1e6:12.3f} {flops_out/mem_out:12.3f}")
        print(
            f"{'Total':<30} {flops_acc/1e6:12.2f} {mem_acc/1e6:12.3f} {flops_acc/mem_acc:12.3f}")
        print("-" * 120)
        return mem_acc, flops_acc

    def plot_roofline(self, peak_flops=1671e12, mem_bw=4.8e12):
        """
        绘制Roofline分析图

        参数:
        peak_flops: 峰值计算性能 (TFLOPS/s)
        mem_bw: 内存带宽 (GB/s)
        """
        if not self.results:
            print("No results to plot. Run calculations first.")
            return

        plt.figure(figsize=(12, 8))
        plt.xscale('log')
        plt.yscale('log')

        oi_range = np.logspace(-1, 3, 500)
        roofline = np.minimum(peak_flops, mem_bw *
                              oi_range) / 1e12  # 转为TFLOPS/s
        plt.plot(oi_range, roofline, 'b-',
                 linewidth=2.5, label='Roofline Bound')

        knee_point = peak_flops / mem_bw
        plt.axvline(x=knee_point, color='gray', linestyle='--', alpha=0.7)
        plt.axhline(y=peak_flops/1e12, color='gray',
                    linestyle='--', alpha=0.7)  # 也要除以1e12

        markers = ['o', 's', '^', 'v', '<', '>', 'p', '*', 'h', 'D','x']
        colors = plt.cm.tab20.colors

        # 只绘制实际点，marker和理论点一致
        legend_labels = set()
        for i, res in enumerate(self.results):
            config_label = f"B={res['B']}, seq_len={res['seq_len']}, use_cache={res['use_cache']}"
            actual_perf = res['flops'] / res['elapsed'] / \
                1e12 if res['elapsed'] > 0 else None  # 转为TFLOPS/s

            marker = markers[i % len(markers)]
            color = colors[i % len(colors)]

            # 只为每种配置加一次label
            if actual_perf is not None and config_label not in legend_labels:
                plt.scatter(res['oi'], actual_perf, s=120,
                            color=color, marker=marker, label=config_label)
                legend_labels.add(config_label)
            elif actual_perf is not None:
                plt.scatter(res['oi'], actual_perf, s=120,
                            color=color, marker=marker)

            # 标注点的数值
            if actual_perf is not None:
                plt.text(res['oi']*1.05, actual_perf*1.1, f"({res['oi']:.2f}, {actual_perf:.2f})",
                         fontsize=10, color=color, ha='left', va='bottom', alpha=0.85)

        # 峰值和带宽标注
        plt.text(oi_range[-1]/2, (peak_flops/1e12)*1.1,
                 f'Comp. Peak: {peak_flops/1e12:.2f} TFLOPS/s',
                 fontsize=12, ha='right', va='bottom', color='blue', weight='bold')

        # 带宽标注
        bw_x = knee_point / 8
        bw_y = mem_bw * bw_x / 1e12  # 转为TFLOPS/s
        plt.text(bw_x, bw_y,
                 f'Mem BW: {mem_bw/1e9:.1f} GB/s',
                 fontsize=12, color='orange', weight='bold',
                 rotation=37, rotation_mode='anchor',
                 ha='left', va='bottom',
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="orange", alpha=0.7))


        plt.xlabel('Operational Intensity (FLOPS/Byte)', fontsize=14)
        plt.ylabel('Performance (TFLOPS/s)', fontsize=14)  # 改为TFLOPS/s
        plt.title('Llama3 Roofline Model Analysis on H200', fontsize=16)
        plt.grid(True, which="both", ls="--", alpha=0.5)

        # 去重图例
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(),
                   fontsize=10, loc='upper left')

        plt.tight_layout()
        plt.savefig("roofline_llama3.png", dpi=200)
        # plt.show()

    def print_results(self):
        """打印所有计算结果"""
        print(
            f"\nLLama3 Performance MHAAnalysis Results, bytes_per_value={self.bytes_per_value} Byte")
        print("=" * 140)
        print(
            f"{'Config':<30} {'FLOPs (M)':>12}{'MACs (M)':>14} {'Mem (MB)':>12} {'OI(FLOPS/Byte)':>15}{'Time(ms)':>10} {'Perf(TFLOPS/s)':>15}{'Throughput(tokens/s)':>15}")
        print("-" * 140)

        for res in self.results:
            config = f"B={res['B']},seq_len={res['seq_len']},use_cache={res['use_cache']}"
            flops_g = res['flops'] / 1e6
            macs_g = flops_g / 2
            mem_mb = res['memory_access'] / 1e6 
            time_ms = res['elapsed'] * 1e3
            flops_s = res['flops_s'] / 1e12
            throughput = res['B'] / res['elapsed']
            print(
                f"{config:<30} {flops_g:12.2f} {macs_g:12.2f} {mem_mb:12.3f} {res['oi']:12.3f} {time_ms:14.3f} {flops_s:15.3f} {throughput:15.3f}")

        print("=" * 140)
        print("Note: OI = Operational Intensity (FLOPS/Byte)")


# ====================== 示例使用 ======================
if __name__ == "__main__":
    # 1.H200 相关数据
    peak_flops = 1671e12/2 # TFLOPS FP16
    mem_bw = 4.8e12  # TB/s

    # 拐点
    knee_point = peak_flops / mem_bw

    # analyzer = GPT2PerformanceAnalyzer(
    #     d_model=1024,
    #     n_heads=16,
    #     d_ff=4096,
    #     n_layers=1,
    #     vocab_size=50257
    # )
    # llama3
    analyzer2 = PerformanceAnalyzer(
        d_model=4096,
        n_heads=32,
        d_ff=16384,
        n_layers=1,
        vocab_size=49152
    )


    print("llama3")

    B = 1
    analyzer2.calculate_performance(
        B, seq_len=512, use_cache=True, elapsed=0.3614)
    analyzer2.calculate_performance(
        B, seq_len=1024, use_cache=True, elapsed=0.3546)
    analyzer2.calculate_performance(
        B, seq_len=2048, use_cache=True, elapsed=0.359)
    analyzer2.calculate_performance(
        B, seq_len=4096, use_cache=True, elapsed=0.4116)

    analyzer2.calculate_performance(
        B, seq_len=512, use_cache=False, elapsed=0.394)
    analyzer2.calculate_performance(
        B, seq_len=1024, use_cache=False, elapsed=0.4488)
    analyzer2.calculate_performance(
        B, seq_len=2048, use_cache=False, elapsed=0.813)
    analyzer2.calculate_performance(
        B, seq_len=4096, use_cache=False, elapsed=1.7332)
    
    B = 8
    analyzer2.calculate_performance(
        B, seq_len=512, use_cache=True, elapsed=0.4064)
    analyzer2.calculate_performance(
        B, seq_len=1024, use_cache=True, elapsed=0.544)
    analyzer2.calculate_performance(
        B, seq_len=2048, use_cache=True, elapsed=0.7896)
    analyzer2.calculate_performance(
        B, seq_len=4096, use_cache=True, elapsed=1.4026)

    analyzer2.calculate_performance(
        B, seq_len=512, use_cache=False, elapsed=1.3234)
    analyzer2.calculate_performance(
        B, seq_len=1024, use_cache=False, elapsed=2.5862)
    analyzer2.calculate_performance(
        B, seq_len=2048, use_cache=False, elapsed=5.6734)
    analyzer2.calculate_performance(
        B, seq_len=4096, use_cache=False, elapsed=13.6074)


    # 3. 打印结果
    analyzer2.print_results()

    # 4. 绘制Roofline分析图 (使用图片提供的硬件参数)
    analyzer2.plot_roofline(peak_flops=peak_flops, mem_bw=mem_bw)
