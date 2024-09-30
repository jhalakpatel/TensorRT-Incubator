module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64 : i64>, #dlti.dl_entry<!executor.ptr<host>, 64 : i64>, #dlti.dl_entry<!executor.ptr<device>, 64 : i64>>, executor.global_init_func = @executor_init_globals, executor.process_grid_shape = array<i64: 1, 1>} {
  executor.func private @__cuda_stream_create() -> !executor.ptr<host>
  executor.global @stream0 constant : !executor.ptr<host>
  executor.func private @_trtrt_enqueue(!executor.opaque<"trtrt_context">, !executor.ptr<host>, ...)
  executor.func private @_trtrt_create_runtime() -> !executor.opaque<"trtrt_runtime">
  executor.func private @_trtrt_create_context(!executor.opaque<"trtrt_engine">) -> !executor.opaque<"trtrt_context">
  executor.func private @_trtrt_load(!executor.opaque<"trtrt_runtime">, !executor.ptr<host>) -> !executor.opaque<"trtrt_engine">
  executor.global @tensorrt_runtime : !executor.opaque<"trtrt_runtime">
  executor.constant_resource @tensorrt_cluster_engine_data dense_resource<__elided__> : tensor<14164xi8>
  executor.global @tensorrt_cluster_exec_ctx constant : !executor.opaque<"trtrt_context">

  // Allocated pointer, aligned pointer, offset, sizes ..., strides ...
  !input_type = !executor.table<!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64, i64, i64>
  !output_type = !executor.table<!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64, i64, i64>


  // num_res, rank, ptr, shape, ...
  !packed_output_descriptors = !executor<i64, i64, i64, i64>

  func.func @main(%arg0: !input_type, %arg1: !input_type>) attributes {executor.function_metadata = #executor.func_meta<[memref<1xf32, #executor.memory_type<device>>, memref<1xf32, #executor.memory_type<device>>], [memref<1xf32, #executor.memory_type<device>>]} -> !output_type {
    %c0_i64 = executor.constant 0 : i64
    %c1_i64 = executor.constant 1 : i64
    %ctx = executor.get_global @tensorrt_cluster_exec_ctx : !executor.opaque<"trtrt_context">
    %stream = executor.get_global @stream0 : !executor.ptr<host>
    %arg0_device_ptr = executor.table.get %arg0[1] : <!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64, i64, i64>
    %arg1_device_ptr = executor.table.get %arg1[1] : <!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64, i64, i64>

    // Allocate output descriptor host memory
    %output_descriptors = executor.alloca !packed_output_descriptors: !executor.ptr<host>

    // Get offset into the output descriptor host memory for output 0. Rank is known at the compile time.
    %rank_offset_0 = executor.getoffset [%c0, %c1] : (i64, i64) -> i64, !packed_output_descriptors

    // Store rank into output_descriptors
    executor.store %c1, %output_descriptors + %rank_offset_0 : i64, !executor.ptr<host>

    // Create memref table for only inputs: ptr, offset, rank, shape0, shape1
    %input_memrefs = executor.table.create(%arg0_device_ptr, %c0_i64, %c1_i64, %c1_i64, %c1_i64, %arg1_device_ptr, %c0_i64, %c1_i64, %c1_i64, %c1_i64 : !executor.ptr<device>, i64, i64, i64, i64, !executor.ptr<device>, i64, i64, i64, i64) : <!executor.ptr<device>, i64, i64, i64, i64, !executor.ptr<device>, i64, i64, i64, i64>

    executor.call @_trtrt_alloc_enqueue(%ctx, %stream, %output_descriptors, %input_memrefs) : (!executor.opaque<"trtrt_context">, !executor.ptr<host>, !executor.ptr<host>, !executor.table<!executor.ptr<device>, i64, i64, i64, i64, !executor.ptr<device>, i64, i64, i64, i64>) -> ()

    // Get device pointer offset into the output descriptor host memory for output 0. Device pointer is available after enqueue.
    %device_ptr_offset_0 = executor.getoffset [%c0, %c2] : (i64, i64) -> i64, !packed_output_descriptors

    // Number of shapes is equal to rank i.e. known at compile time.
    // Get shape offsets into the output descriptor host memory for output 0. Shapes are available after enqueue.
    %shape_0_ptr_offset_0 = executor.getoffset [%c0, %c3] : (i64, i64) -> i64, !packed_output_descriptors
    %shape_1_ptr_offset_0 = executor.getoffset [%c0, %c4] : (i64, i64) -> i64, !packed_output_descriptors

    // Create output memref from output descriptors

    // ... For each result
    %o0.rank = executor.load %output_descriptors + %rank_offset_0 : (!executor.ptr<host>, i64) -> i64
    %o0.ptr = executor.load %output_descriptors + %device_ptr_offset_0 : (!executor.ptr<host>, i64) -> i64
    // ... For each shape i.e. 0 ... rank - 1
    %o0.s0 = executor.load %output_descriptors + %shape_0_ptr_offset_0 : (!executor.ptr<host>, i64) -> i64
    %o0.s1 = executor.load %output_descriptors + %shape_1_ptr_offset_0 : (!executor.ptr<host>, i64) -> i64

    // Allocated pointer, aligned pointer, offset, sizes, strides (TODO: How to compute stride here?)
    %5 = executor.table.create(%o0.ptr, %o0.ptr, %c0, %o0.s0, %o0.s1, %o0.s1, %c1_i64 : !executor.ptr<device>, !executor.ptr<device>, i64, i64, i64, i64, i64) : !output_type

    return %5 : !output_type
  }
  func.func private @executor_init_globals() {
    %0 = executor.call @__cuda_stream_create() : () -> !executor.ptr<host>
    executor.set_global %0, @stream0 : !executor.ptr<host>
    %1 = executor.call @_trtrt_create_runtime() : () -> !executor.opaque<"trtrt_runtime">
    executor.set_global %1, @tensorrt_runtime : !executor.opaque<"trtrt_runtime">
    %2 = executor.load_constant_resource @tensorrt_cluster_engine_data : !executor.ptr<host>
    %3 = executor.get_global @tensorrt_runtime : !executor.opaque<"trtrt_runtime">
    %4 = executor.call @_trtrt_load(%3, %2) : (!executor.opaque<"trtrt_runtime">, !executor.ptr<host>) -> !executor.opaque<"trtrt_engine">
    %5 = executor.call @_trtrt_create_context(%4) : (!executor.opaque<"trtrt_engine">) -> !executor.opaque<"trtrt_context">
    executor.set_global %5, @tensorrt_cluster_exec_ctx : !executor.opaque<"trtrt_context">
    return
  }
}
