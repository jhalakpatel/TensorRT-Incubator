# RUN: %PYTHON %s
import time

import mlir_tensorrt.compiler.api as compiler
import mlir_tensorrt.compiler.ir as ir
import mlir_tensorrt.runtime.api as runtime
import numpy as np

ASM = """
func.func @main(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
  %1 = stablehlo.add %arg0, %arg1 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  func.return %1 : tensor<1xf32>
}
"""


def stablehlo_add():
    # Build/parse the main function.
    with ir.Context() as context:
        m = ir.Module.parse(ASM)

        # Use the compiler API to compile to executable.
        client = compiler.CompilerClient(context)
        opts = compiler.StableHLOToExecutableOptions(
            client,
            ["--tensorrt-builder-opt-level=3", "--tensorrt-strongly-typed=false"],
        )
        opts.set_debug_options(False, [], "alloc_enqueue")
        exe = compiler.compiler_stablehlo_to_executable(client, m.operation, opts)

    # The RuntimeClient can and should persist across multiple Executables, RuntimeSessions, etc.
    # It is primarily an interface for creating and manipulating buffers.
    client = runtime.RuntimeClient()
    stream = client.create_stream()
    devices = client.get_devices()

    if len(devices) == 0:
        return

    session_options = runtime.RuntimeSessionOptions(num_devices=1, device_id=0)
    session = runtime.RuntimeSession(session_options, exe)

    arg0 = client.create_memref(
        np.array([1.0], dtype=np.float32).data,
        device=devices[0],
        stream=stream,
    )
    arg1 = client.create_memref(
        np.array([2.0], dtype=np.float32).data,
        device=devices[0],
        stream=stream,
    )

    arg2 = client.create_memref(
        np.zeros(shape=(1,), dtype=np.float32).data,
        device=devices[0],
        stream=stream,
    )
    session.execute_function(
        "main", in_args=[arg0, arg1], out_args=[arg2], stream=stream
    )

    data = np.asarray(client.copy_to_host(arg2, stream=stream))
    stream.sync()

    print(data)


if __name__ == "__main__":
    stablehlo_add()
