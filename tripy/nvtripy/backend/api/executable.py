# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import base64
import inspect
from dataclasses import dataclass
from typing import Sequence, Tuple, Union

import mlir_tensorrt.runtime.api as runtime
from nvtripy import export
from nvtripy.backend.mlir import Executor
from nvtripy.backend.mlir import utils as mlir_utils
from nvtripy.common.exception import raise_error
from nvtripy.frontend import Tensor
from nvtripy.utils import json as json_utils
from nvtripy.utils.types import str_from_type_annotation


# TODO(MLIR-TRT #923): Can generalize `InputInfo` and drop this class.
@dataclass
class ArgInfo:
    shape_bounds: Sequence[Tuple[int, int]]
    """A sequence of tuple(min, max) indicating the bounds of each dimension"""
    dtype: "nvtripy.dtype"
    """The datatype of the argument"""


# Executable.__call__ is in the hot path for benchmarks, so we would not want additional overhead
@export.public_api(document_under="compiling_code", bypass_dispatch=["__call__"])
class Executable:
    """
    Represents a compiled executable generated by the compiler.

    .. seealso:: :func:`compile`
    """

    # The constructor is intentionally undocumented because it is not meant to be called by users.
    # TODO(#155): output_devices is not needed after they can be queried from executable
    def __init__(self, executable, arg_names, output_devices):
        self._executable = executable
        self._executor = Executor(self._executable)
        self._arg_names = arg_names
        self._num_expected_args = len(arg_names)
        self._output_devices = output_devices
        self._executable_signature = self._executable.get_signature("main")

        # Build a signature so the executable works with `inspect.signature`
        params = []
        for name in self._arg_names:
            params.append(inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=Tensor))

        return_annotation = Tensor if self._executable_signature.get_num_output_args() == 1 else Sequence[Tensor]

        self.__signature__ = inspect.Signature(params, return_annotation=return_annotation)

    @property
    def stream(self):
        return self._executor.stream

    @stream.setter
    def stream(self, stream):
        self._executor.stream = stream

    def __str__(self) -> str:
        params = [
            f"{name}: {str_from_type_annotation(param.annotation)}"
            for name, param in self.__signature__.parameters.items()
        ]
        return f"Executable({', '.join(params)}) -> {str_from_type_annotation(self.__signature__.return_annotation)}"

    @staticmethod
    def load(path: str) -> "nvtripy.Executable":
        """
        Loads a executable from the provided path.

        Args:
            path: The path from which to load the exectuable.

        Returns:
            The executable object loaded from the file.

        .. code-block:: python
            :linenos:
            :caption: Save and load executable

            import os
            import tempfile # doc: omit

            def add(a, b):
                return a + b

            # doc: no-print-locals compiled_add executable_file
            compiled_add = tp.compile(
                add,
                args=[
                    tp.InputInfo(shape=((1, 2, 3),), dtype=tp.float32),
                    tp.InputInfo(shape=((1, 2, 3),), dtype=tp.float32),
                ],
            )


            out_dir = tempfile.TemporaryDirectory().name # doc: omit
            # Assuming `out_dir` is the directory containing the executable:
            executable_file = os.path.join(out_dir, "executable.json")
            compiled_add.save(executable_file) # doc: omit
            assert os.path.exists(executable_file)
            loaded_executable = tp.Executable.load(executable_file)
        """

        return json_utils.load(path)

    def __call__(self, *args: Tensor, **kwargs: Tensor) -> Union[Tensor, Sequence[Tensor]]:
        """
        Invokes the executable with the specified tensor arguments.

        Args:
            *args: Positional arguments. Must be of type :class:`Tensor` .
            **kwargs: Keyword arguments. Must be of type :class:`Tensor` .

        Returns:
            The output :class:`Tensor` s of the compiled function.


        .. code-block:: python
            :linenos:

            def add(a, b):
                return a + b

            # doc: no-print-locals compiled_add
            compiled_add = tp.compile(
                add,
                args=[
                    tp.InputInfo((1,), dtype=tp.float32),
                    tp.InputInfo((1,), dtype=tp.float32),
                ],
            )

            a = tp.ones((1,), dtype=tp.float32)
            b = tp.ones((1,), dtype=tp.float32)

            out = compiled_add(a, b)
        """
        num_positional = len(args)
        NUM_ARGS = num_positional + len(kwargs)

        input_tensors = list(args)
        # Need to get arguments in the order of self._arg_names, which may be different from kwargs ordering.
        expected_kwargs = self._arg_names[num_positional:]
        for name in expected_kwargs:
            if name not in kwargs:
                raise_error(f"Missing argument: {name}", [f"Expected the following arguments: {self._arg_names}"])

            input_tensors.append(kwargs[name])
            del kwargs[name]

        if kwargs:
            raise_error(
                f"Extra keyword arguments: {list(kwargs.keys())}",
                [
                    f"Expected the following arguments: {self._arg_names}.\n"
                    f"Note: The following arguments were already provided as positional arguments: {self._arg_names[:num_positional]}"
                ],
            )

        # We do this after kwarg checks since those will be more informative (we can explain which arguments are missing/extra).

        if NUM_ARGS != self._num_expected_args:
            raise_error(
                "Incorrect number of arguments.",
                [
                    f"Expected {self._num_expected_args} arguments but got {NUM_ARGS}.\n"
                    f"Note: Expected arguments were: {self._arg_names}",
                ],
            )

        # The executor expects concrete tensors as inputs, so we need to eval() here.
        for tensor in input_tensors:
            tensor.eval()

        try:
            executor_outputs = self._executor.execute(
                self._output_devices, inputs=[tensor.trace_tensor for tensor in input_tensors]
            )
        except runtime.MTRTException as err:
            # TODO: Evaluate whether this should be moved into the executor
            if "function expects a memref type with element type" in str(err):
                # If the problem is a mismatched data type, we can provide a better error message than the executor can.
                expected_input_dtypes = [info.dtype for info in self._get_input_info()]
                for tensor, dtype, arg_name in zip(input_tensors, expected_input_dtypes, self._arg_names):
                    if tensor.dtype != dtype:
                        raise_error(
                            f"Unexpected tensor data type.",
                            [
                                f"For parameter {arg_name}, expected data type: {dtype} but got: {tensor.dtype}. Note: Argument was: ",
                                tensor,
                            ],
                        )
            elif "InternalError: failed to set input shape" in str(err) or "Runtime shape mismatch" in str(err):
                expected_input_shapes = [info.shape_bounds for info in self._get_input_info()]
                for tensor, expected_bounds, arg_name in zip(input_tensors, expected_input_shapes, self._arg_names):
                    shape = tensor.shape
                    for i in range(len(shape)):
                        if shape[i] < expected_bounds[i][0] or shape[i] > expected_bounds[i][1]:
                            min_shape, max_shape = zip(*expected_bounds)
                            raise_error(
                                f"Unexpected tensor shape.",
                                [
                                    f"For tensor: `{arg_name}`, expected a shape within the bounds: min={min_shape}, max={max_shape}, but got: {shape}.\n"
                                    f"Dimension {i} has a shape of {shape[i]}, which is not within the expected bounds of {list(expected_bounds[i])}.\n"
                                    f"Note: The provided argument was: ",
                                    tensor,
                                ],
                            )
            elif "Runtime stride mismatch" in str(err):
                # Just raise the error for now.
                raise raise_error(str(err))

            raise

        output_tensors = [Tensor.fast_init(output) for output in executor_outputs]
        if len(output_tensors) == 1:
            output_tensors = output_tensors[0]
        return output_tensors

    def _get_arg_info(self, idx):
        arg = self._executable_signature.get_arg(idx)
        arg = runtime.MemRefType(arg)
        arg_bound = self._executable_signature.get_arg_bound(idx)
        shape_bounds = tuple(zip(arg_bound.min(), arg_bound.max()))
        if len(shape_bounds) == 0:
            # For static shape arguments, get_arg_bound returns an empty list and we fallback to arg.shape
            shape_bounds = tuple((x, x) for x in arg.shape)
        return ArgInfo(shape_bounds, mlir_utils.convert_runtime_dtype_to_tripy_dtype(arg.dtype))

    def _get_input_info(self) -> Sequence[ArgInfo]:
        input_info = []
        for idx in range(self._executable_signature.get_num_input_args()):
            input_info.append(self._get_arg_info(idx))
        return input_info

    def _get_output_info(self) -> Sequence[ArgInfo]:
        output_info = []
        offset = self._executable_signature.get_num_input_args()
        for idx in range(self._executable_signature.get_num_output_args()):
            output_info.append(self._get_arg_info(idx + offset))
        return output_info

    def save(self, path: str) -> None:
        """
        Saves this executable to the provided path.

        Args:
            path: The path at which to save the executable.

        .. code-block:: python
            :linenos:
            :caption: Save executable

            import os
            import tempfile # doc: omit

            def add(a, b):
                return a + b

            # doc: no-print-locals compiled_add executable_file
            compiled_add = tp.compile(
                add,
                args=[
                    tp.InputInfo(shape=((1, 2, 3),), dtype=tp.float32),
                    tp.InputInfo(shape=((1, 2, 3),), dtype=tp.float32),
                ],
            )

            out_dir = tempfile.TemporaryDirectory().name # doc: omit
            # Assuming `out_dir` is the desired output directory:
            executable_file = os.path.join(out_dir, "executable.json")
            compiled_add.save(executable_file)
            assert os.path.exists(executable_file)
        """
        json_utils.save(self, path)


@json_utils.Encoder.register(Executable)
def encode_executable(executable):
    return {
        "arg_names": executable._arg_names,
        "output_devices": executable._output_devices,
        "executable": base64.b64encode(executable._executable.serialize()).decode(),
    }


@json_utils.Decoder.register(Executable)
def decode_executable(executable_dict):
    executable_bytes = base64.b64decode(executable_dict["executable"])
    return Executable(
        runtime.Executable(executable_bytes),
        executable_dict["arg_names"],
        executable_dict["output_devices"],
    )
