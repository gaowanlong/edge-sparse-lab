# Phase 2 设计书：From Algorithm to Deployment

> 在 M1 Pro 上完成端侧稀疏化的完整验证链：算法 → 模式 → 运行时 → 分析

---

## 一、Phase 2 目标

v0.1 验证了基础流程，暴露了核心问题：

| v0.1 成果 | v0.1 短板 |
|---|---|
| Wanda 剪枝端到端跑通 | 50% 稀疏度下 PPL 崩到 129（仅 Wanda 不够） |
| 三种 mask 模式有单元测试 | 没在真实模型上跑过 N:M / block |
| PyTorch MPS 推理路径通 | 没有真正的端侧运行时（MLX 空了） |
| PPL 评估结果正确 | 没有 latency/memory benchmark |

Phase 2 目的是**填上这些缺口**，形成完整的研究循环：

```
算法设计 → 剪枝实施 → 模式选择 → 推理加速 → 系统分析
                ↑                          │
                └──── 结果反馈 ────────────┘
```

---

## 二、模块结构

```
src/edgesparse/
├── pruning/
│   ├── mask_utils.py        [v0.1] ✅
│   ├── magnitude.py         [v0.1] ✅
│   ├── wanda.py             [v0.1] ✅
│   ├── structured_nm.py     [v0.1] ✅
│   ├── block_sparse.py      [v0.1] ✅
│   └── sparsegpt.py         [NEW]  ← SparseGPT 算法
│
├── runtime/
│   ├── __init__.py          [v0.1] ⬜
│   ├── mlx_engine.py        [NEW]  ← MLX 推理后端
│   └── benchmark.py         [NEW]  ← latency / memory / bandwidth
│
├── system/
│   ├── __init__.py          [v0.1] ⬜
│   ├── analyze.py           [NEW]  ← per-layer 分析 / 可视化数据导出
│   └── report_viz.py        [NEW]  ← 对比实验报告生成 (Markdown + table)
│
├── eval/     [v0.1] ✅
├── models/   [v0.1] ✅
└── calibration/ [v0.1] ✅

scripts/
├── 00_download_model.py     [v0.1] ✅
├── 01_build_calibration.py  [v0.1] ✅
├── 02_prune.py              [v0.1] ✅ (补充分组对比模式)
├── 03_eval_quality.py       [v0.1] ✅
├── 04_benchmark.py          [NEW]  ← latency / memory 对比
├── 05_compare_methods.py    [NEW]  ← 多方法多模式批量对比
└── 06_convert_to_mlx.py     [NEW]  ← 导出 MLX 模型

tests/
├── test_sparsegpt.py        [NEW]  ← SparseGPT 单元测试
├── test_mlx_engine.py       [NEW]  ← MLX runtime 测试
└── test_analyze.py          [NEW]
```

---

## 三、四个方向的具体设计

### 方向 1：SparseGPT — 更好的剪枝质量

**动机**：Wanda 在 50% 时 PPL 从 19.99 崩到 129.56。SparseGPT 通过 Hessian 信息保留更重要的权重，同等稀疏度下质量明显更好。

**算法**：OBS (Optimal Brain Surgeon) 的 Layer-wise 近似

```
对于每层:
  1. 校准数据 forward，收集激活值 H = XᵀX
  2. 计算 Hessian 近似 H_hess = 2 * (H + λI)
  3. 贪心剪枝:
     for each step:
       找到最小化的 weight: ρ = w² / 2 * diag(H⁻¹)
       更新剩余 weight: δ = -w / H⁻¹[col,col] * H⁻¹[:,col]
```

**实现**：`pruning/sparsegpt.py`

- `SparseGPTPruner` 类，逐个 Linear 层处理
- 与 `wanda.py` 同接口：`sparsegpt_prune(model, activation_norms, sparsity, pattern)`
- 复用 v0.1 的 mask 模块创建 mask
- 只做 unstructured 和 N:M（SparseGPT 不适用于 block）

**验证**：Qwen3-0.6B 上 50% sparsity，目标 PPL < 100（Wanda 是 129）

---

### 方向 2：MLX Runtime — 真正的端侧推理

**动机**：PyTorch MPS 不支持稀疏 matmul 加速。MLX 是 Apple 官方为 Apple Silicon 做的框架，原生支持 block-sparse。

**架构**：

```
PyTorch 剪枝模型 (safetensors)
      │
      ▼
mlx_engine.convert(pytorch_model) → MLX 权重 (mx.array)
      │
      ├── dense inference:  mx.fast.layer_norm, mx.fast.linear
      ├── sparse inference:  自定义 block-sparse matmul (CPU fallback)
      └── benchmark:         对比 dense vs 各种 sparse 模式的 latency
```

**实现**：`runtime/mlx_engine.py`

- `MLXRuntime` 类，封装 MLX 模型的加载和推理
- `convert_to_mlx(model, tokenizer)` 将 PyTorch 模型转为 MLX 权重字典
- `generate(prompt, max_tokens)` 简单的自回归生成接口
- `benchmark_latency(pattern, sparsity)` 测试各种模式的推理速度

**注意**：MLX 目前对稀疏 matmul 的支持有限（主要依靠 block-sparse 的 struct 类型）。实际实现中可能需要：
1. 使用 mlx.core 的 block-sparse 类型（如果支持）
2. 或者用 masked fill 模拟（准确但不加速）
3. 或者用 custom kernel via metal compute

先做 #1/#2 让 pipeline 跑通，后续可以深入 Metal 自定义 kernel。

---

### 方向 3：多种 Pattern 的全面实验

**动机**：v0.1 只在单元测试验证了 N:M 和 block mask，没在真实模型上跑过。

**方案**：在 `05_compare_methods.py` 中批量跑所有组合

| 方法 | Pattern | 稀疏度 |
|---|---|---|
| Magnitude | unstructured | 25%, 50% |
| Wanda | unstructured | 25%, 50% |
| Wanda | N:M 2:4 | 50% |
| Wanda | N:M 4:8 | 50% |
| Wanda | block 4×8 | 50% |
| Wanda | block 8×8 | 50% |
| SparseGPT | unstructured | 50%, 75% |
| SparseGPT | N:M 2:4 | 50% |

输出一个 Markdown 对比表：

| 方法 | Pattern | 稀疏率 | PPL | Speedup |
|---|---|---|---|---|
| Dense | — | 0% | 19.99 | 1.0× |
| Wanda | unstr | 37% | 129.56 | ~? |
| SparseGPT | unstr | 37% | ? | ~? |
| Wanda | 2:4 | 50% | ? | ~? |
| ... | ... | ... | ... | ... |

---

### 方向 4：System Analysis — Benchmark + 可视化

**动机**：回答「稀疏化在 M1 Pro 上到底加速了多少？」

**`system/analyze.py`** — Per-layer 分析
- sparsity per layer (已实现) + weight distribution histograms
- per-layer PPL contribution（哪层剪坏了最影响质量）
- 导出 matplotlib 用的 JSON 数据

**`system/report_viz.py`** — 对比报告
- 读取多个实验的 json 报告，合并生成 Markdown 对比表
- 调用 matplotlib（如果有）或输出数据表格
- 输出完整的 report.md

**`system/benchmark.py`**（在 `runtime/benchmark.py` 中实现）
- 测量：forward latency (per token), peak memory, bandwidth
- 对比：dense vs wanda/unstr vs wanda/2:4 vs magnitude
- 结果输出 JSON + Markdown 表

---

## 四、新增脚本

### `scripts/05_compare_methods.py`

批量跑所有方法 × 模式的组合，生成统一报告。

```bash
python scripts/05_compare_methods.py \
  --model Qwen/Qwen3-0.6B \
  --methods magnitude wanda sparsegpt \
  --patterns unstructured nm block \
  --sparsities 0.25 0.50 \
  --output outputs/comparison
```

### `scripts/04_benchmark.py`

推理速度 benchmark。

```bash
python scripts/04_benchmark.py \
  --model outputs/runs/qwen3_0_6b_wanda50 \
  --prompt-length 1024 \
  --gen-length 128 \
  --repeats 5
```

### `scripts/06_convert_to_mlx.py`

将剪枝后的模型转换为 MLX 格式。

```bash
python scripts/06_convert_to_mlx.py \
  --model outputs/runs/qwen3_0_6b_wanda50 \
  --output outputs/mlx/qwen3_0_6b_wanda50
```

---

## 五、实施路线图

```
Phase 2 (当前批次)
├── Step 1: SparseGPT       ← 核心算法
├── Step 2: MLX Runtime     ← 运行时
├── Step 3: System Analysis ← 分析工具
├── Step 4: 批量实验脚本     ← 串起所有
└── Step 5: v0.2 发布       ← commit + push tag
```

各步骤预估工作量：

| Step | 文件数 | 新代码行 | 依赖 |
|---|---|---|---|
| 1. SparseGPT | 2 (pr, test) | ~300 | v0.1 pruning 框架 |
| 2. MLX Runtime | 3 (runtime, test, script) | ~400 | mlx + mlx-lm |
| 3. System Analysis | 2 (system) | ~250 | matplotlib (可选) |
| 4. 批量实验 | 2 (scripts) | ~350 | 以上全部完成 |
| 5. 发布 | 1 commit | — | 以上全部完成 |

---

## 六、验收标准

- [ ] SparseGPT 50% PPL < 100（Wanda baseline 129）
- [ ] MLX 能加载 pruned 模型 + 运行生成
- [ ] 至少输出一次完整的 pattern × method 对比表
- [ ] benchmark 跑出 dense vs sparse 的 latency 数据
- [ ] pytest 全部通过（含新测试）
- [ ] Qwen3-0.6B 在 2:4 模式下跑通
- [ ] 发布 v0.2 tag 到 GitHub
