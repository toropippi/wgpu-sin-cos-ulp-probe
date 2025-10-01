# wgpu-sin-cos-ulp-probe

**A Python tool for measuring and comparing whether `sin(x)` and `cos(x)` are implemented as standard (≤2 ULP guaranteed) or as approximate “fast-math” versions across multiple GPU backends (Vulkan, D3D12, OpenGL) using [wgpu-py].**

It dispatches WGSL compute shaders with the built-in `sin/cos` functions, reads back GPU results, and evaluates ULP distances against double-precision reference values.

複数のGPUバックエンド（Vulkan、D3D12、OpenGL）における`sin(x)`と`cos(x)`が標準版(2ULP以内保証)か近似版(fast math)になるかを測定・比較するPythonツール（[wgpu-py]を使用）。  

組み込みの`sin/cos`関数を持つWGSLをディスパッチし、結果を読み取り、倍精度参照値とのULP距離を評価します。  

## Features
- Backend switching (Vulkan, D3D12, OpenGL, Metal on macOS)
- ULP error measurement for selected test values
- Tabular output for easy comparison
- バックエンド切り替え（Vulkan、D3D12、OpenGL、macOS上のMetal）
- 選択したテスト値に対するULP誤差測定
- 比較しやすい表形式出力
