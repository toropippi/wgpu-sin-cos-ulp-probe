# Python 3.10+
# pip install wgpu numpy
# 使い方:
#   1) まとめて実行:  python multi_backend_ulp_probe.py
#      -> OSに応じて D3D12/Vulkan/OpenGL を順に起動し、結果を1つの表として表示
#   2) バックエンド個別実行:  python multi_backend_ulp_probe.py --backend Vulkan

import os, sys, math, argparse, subprocess
import numpy as np

# ========= ULP（float32）計測ユーティリティ =========

def _float32_to_ordered_uint32(a: np.ndarray) -> np.ndarray:
    assert a.dtype == np.float32
    u = a.view(np.uint32).copy()
    sign = (u & 0x80000000) != 0
    u = np.where(sign, (np.uint32(0x80000000) - u), (u + np.uint32(0x80000000)))
    return u

def _ulp_distance_f32(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # ±0.0 を +0.0 へ正規化（ULP=0にしたい）
    a = a.copy(); b = b.copy()
    a[a == 0.0] = np.float32(0.0)
    b[b == 0.0] = np.float32(0.0)

    finite = np.isfinite(a) & np.isfinite(b)
    oa = _float32_to_ordered_uint32(a)
    ob = _float32_to_ordered_uint32(b)
    out = np.empty_like(oa, dtype=np.int64)
    out[:] = np.iinfo(np.int32).max  # non-finite sentinel
    diff = np.abs(oa.astype(np.int64) - ob.astype(np.int64))
    out[finite] = diff[finite]
    return out

# ========= 子プロセス側: 実測ロジック（WGSLは組込みsin/cosを使用） =========

def _run_gpu_and_print_rows(backend: str):
    # ここでバックエンドを環境変数で固定（インポート前に設定）
    if backend and backend.lower() != "auto":
        os.environ["WGPU_BACKEND_TYPE"] = backend

    import wgpu  # インポート時にバックエンドが決まる

    TEST_X = np.array([
        114514.0,
        0.000123,
        math.pi * 32768.0,
        math.pi * (2**22),
        1.0, 10.0, 1000.0, -1.0, -1000.0
    ], dtype=np.float32)

    WGSL = """
@group(0) @binding(0) var<storage, read>        xin : array<f32>;
@group(0) @binding(1) var<storage, read_write>  ys  : array<f32>;
@group(0) @binding(2) var<storage, read_write>  yc  : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = gid.x;
    if (i < arrayLength(&xin)) {
        let x = xin[i];
        // 近似挙動を検証するため、組込みの sin/cos をそのまま使用
        ys[i] = sin(x);
        yc[i] = cos(x);
    }
}
    """

    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync()

    nbytes = TEST_X.nbytes
    buf_in  = device.create_buffer_with_data(
        data=TEST_X.tobytes(),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
    )
    buf_sin = device.create_buffer(
        size=nbytes,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
    )
    buf_cos = device.create_buffer(
        size=nbytes,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
    )

    shader = device.create_shader_module(code=WGSL)
    bgl = device.create_bind_group_layout(entries=[
        {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
        {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {"type": wgpu.BufferBindingType.storage}},
        {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {"type": wgpu.BufferBindingType.storage}},
    ])
    pl = device.create_pipeline_layout(bind_group_layouts=[bgl])
    pipe = device.create_compute_pipeline(layout=pl, compute={"module": shader, "entry_point": "main"})
    bg = device.create_bind_group(layout=bgl, entries=[
        {"binding": 0, "resource": {"buffer": buf_in,  "offset": 0, "size": nbytes}},
        {"binding": 1, "resource": {"buffer": buf_sin, "offset": 0, "size": nbytes}},
        {"binding": 2, "resource": {"buffer": buf_cos, "offset": 0, "size": nbytes}},
    ])

    enc = device.create_command_encoder()
    c = enc.begin_compute_pass()
    c.set_pipeline(pipe)
    c.set_bind_group(0, bg, [], 0, 999_999)
    wg = (TEST_X.size + 63) // 64
    c.dispatch_workgroups(wg, 1, 1)
    c.end()
    device.queue.submit([enc.finish()])

    ysin = np.frombuffer(device.queue.read_buffer(buf_sin), dtype=np.float32).copy()
    ycos = np.frombuffer(device.queue.read_buffer(buf_cos), dtype=np.float32).copy()

    ref_sin = np.sin(TEST_X.astype(np.float64)).astype(np.float32)
    ref_cos = np.cos(TEST_X.astype(np.float64)).astype(np.float32)

    ulp_s = _ulp_distance_f32(ysin, ref_sin)
    ulp_c = _ulp_distance_f32(ycos, ref_cos)

    # 子プロセスは「行データのみ」を出力（親がヘッダ印字＆集約）
    be = (backend or "Auto")
    for i, xv in enumerate(TEST_X):
        print(f"{be:7} | {i:2d} | {xv:>15.6g} | {int(ulp_s[i]):>8d} | {int(ulp_c[i]):>8d} | "
              f"{ysin[i]:>12.6g} | {ref_sin[i]:>12.6g} | {ycos[i]:>12.6g} | {ref_cos[i]:>12.6g}")

# ========= 親プロセス側 =========

def _spawn_for_backend(backend: str):
    env = os.environ.copy()
    if backend and backend.lower() != "auto":
        env["WGPU_BACKEND_TYPE"] = backend
    args = [sys.executable, __file__, "--backend", backend]
    return subprocess.run(args, env=env, capture_output=True, text=True, check=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=None, help="Vulkan / D3D12 / OpenGL / Metal / Auto")
    args = ap.parse_args()

    if args.backend:
        # 子モード（行データのみ）
        _run_gpu_and_print_rows(args.backend)
        return

    # 親モード：OSに応じた順序でバックエンドを試行
    if os.name == "nt":  # Windows
        backends = ["D3D12", "Vulkan", "OpenGL"]
    else:  # Linux 他
        backends = ["Vulkan", "OpenGL"]

    # ヘッダは一度だけ印字
    print("backend | i |               x | ULP_sin | ULP_cos |     sin_gpu |     sin_ref |     cos_gpu |     cos_ref")
    print("--------+---+-----------------+---------+---------+------------+------------+------------+------------")

    for b in backends:
        proc = _spawn_for_backend(b)
        out = (proc.stdout or "").strip()
        if out:
            print(out)

if __name__ == "__main__":
    main()
