#
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
#

from collections.abc import Sequence
from dataclasses import dataclass

import nvtripy.trace.ops.utils as op_utils
from nvtripy.trace.ops.base import BaseTraceOp


@dataclass(repr=False)
class Convolution(BaseTraceOp):
    padding: Sequence[Sequence[int]]
    stride: Sequence[int]
    groups: int
    lhs_dilation: Sequence[int]
    rhs_dilation: Sequence[int]

    infer_rank = op_utils.InferRankPolicies.same_as_input()

    def infer_dtypes(self):
        self.outputs[0].dtype = self.inputs[0].dtype

    def to_flat_ir(self, inputs, outputs):
        from nvtripy.flat_ir.ops import ConvolutionOp

        ConvolutionOp.build(
            inputs,
            outputs,
            padding=self.padding,
            stride=self.stride,
            feature_group_count=self.groups,
            lhs_dilation=self.lhs_dilation,
            rhs_dilation=self.rhs_dilation,
        )
