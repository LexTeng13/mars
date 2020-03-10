# Copyright 1999-2020 Alibaba Group Holding Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from ... import opcodes as OperandDef
from ...config import options
from ...serialize import StringField
from ...context import get_context, RunningMode
from ...utils import calc_nsplits
from .core import TensorNoInput


class TensorFromVineyard(TensorNoInput):
    _op_type_ = OperandDef.TENSOR_FROM_VINEYARD

    # ObjectID in vineyard
    _object_id = StringField('object_id')

    def __init__(self, vineyard_socket=None, object_id=None, dtype=None,
                 gpu=None, sparse=None, **kw):
        super().__init__(
            _vineyard_socket=vineyard_socket, _object_id=object_id,
            _dtype=dtype, _gpu=gpu, _sparse=sparse, **kw)

    @property
    def vineyard_socket(self):
        return self._vineyard_socket

    @property
    def object_id(self):
        return self._object_id

    @classmethod
    def tile(cls, op):
        import vineyard

        ctx = get_context()
        if not options.vineyard.socket:
            raise RuntimeError('Not executed with vineyard')
        client = vineyard.connect(options.vineyard.socket)
        tensor = client.get(op.object_id)

        instances = tensor.instances
        chunk_map = {}
        for instance_id in instances:
            for chunk_id in tensor.local_chunks(instance_id):
                chunk = client.get_meta(chunk_id)
                chunk_index = tuple(int(x) for x in chunk['chunk_index'].split(' '))
                shape = tuple(int(x) for x in chunk['shape'].split(' '))
                chunk_map[chunk_index] = (instance_id, chunk['id'], shape)

        nsplits = calc_nsplits({chunk_index: shape
                                for chunk_index, (_, _, shape) in chunk_map.items()})
        if ctx.running_mode == RunningMode.distributed:
            metas = ctx.get_worker_metas()
            workers = {meta['vineyard']['instance_id']: addr for addr, meta in metas.items()}
        else:
            workers = '127.0.0.1'

        out_chunks = []
        for chunk_index, (instance_id, chunk_id, shape) in chunk_map.items():
            chunk_op = op.copy().reset_key()
            chunk_op._object_id = chunk_id
            if ctx.running_mode == RunningMode.distributed:
                chunk_op._expect_worker = workers[instance_id]
            else:
                chunk_op._expect_worker = workers
            out_chunks.append(chunk_op.new_chunk(None, shape=shape, index=chunk_index))

        new_op = op.copy()
        return new_op.new_tileables(op.inputs, chunks=out_chunks, nsplits=nsplits)

    @classmethod
    def execute(cls, ctx, op):
        import vineyard

        chunk = op.outputs[0]

        if not options.vineyard.socket:
            raise RuntimeError('Not executed with vineyard')
        client = vineyard.connect(options.vineyard.socket)
        # chunk has a tensor chunk
        tensor_chunk = client.get(op.object_id)
        ctx[chunk.key] = tensor_chunk.numpy()


def from_vineyard(tensor):
    import vineyard
    if isinstance(tensor, vineyard.GlobalObject):
        object_id = tensor.id
    else:
        object_id = tensor
    op = TensorFromVineyard(object_id=object_id, dtype=np.dtype('byte'), gpu=False)
    return op(shape=(np.nan,), chunk_size=(np.nan,))
