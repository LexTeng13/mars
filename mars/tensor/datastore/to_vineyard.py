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

from ... import opcodes as OperandDef
from ...serialize import KeyField, StringField
from ..datasource import tensor as astensor
from .core import TensorDataStore


class TensorVineyardDataStore(TensorDataStore):
    _op_type_ = OperandDef.TENSOR_STORE_VINEYARD_CHUNK

    _input = KeyField('input')
    # Location for vineyard's socket file
    _vineyard_socket = StringField('vineyard_socket')

    def __init__(self, vineyard_socket=None, dtype=None, sparse=None, **kw):
        super().__init__(
            _vineyard_socket=vineyard_socket, _dtype=dtype, _sparse=sparse, **kw)

    @property
    def vineyard_socket(self):
        return self._vineyard_socket

    @classmethod
    def _get_out_chunk(cls, op, in_chunk):
        chunk_op = op.copy().reset_key()
        out_chunk_shape = (0,) * in_chunk.ndim
        return chunk_op.new_chunk([in_chunk], shape=out_chunk_shape,
                                  index=in_chunk.index)

    @classmethod
    def _process_out_chunks(cls, op, out_chunks):
        merge_op = TensorVineyardDataStoreMerge(
            vineyard_socket=op.vineyard_socket, sparse=op.sparse, dtype=op.dtype)
        return merge_op.new_chunks(out_chunks, shape=out_chunks[0].shape,
                                   index=(0,) * out_chunks[0].ndim)

    @classmethod
    def tile(cls, op):
        return [super().tile(op)[0]]

    @classmethod
    def execute(cls, ctx, op):
        import vineyard

        client = vineyard.connect(op.vineyard_socket)

        tensor = vineyard.TensorBuilder.from_numpy(client, ctx[op.input.key])
        tensor = tensor.build(client)
        tensor.persist(client)

        # store the result object id to execution context
        ctx[op.outputs[0].key] = (client.instance_id, tensor.id)


class TensorVineyardDataStoreMerge(TensorDataStore):
    _op_type_ = OperandDef.TENSOR_STORE_VINEYARD

    _input = KeyField('input')
    # Location for vineyard's socket file
    _vineyard_socket = StringField('vineyard_socket')

    def __init__(self, vineyard_socket=None, dtype=None, sparse=None, **kw):
        super().__init__(
            _vineyard_socket=vineyard_socket, _dtype=dtype, _sparse=sparse, **kw)

    @property
    def vineyard_socket(self):
        return self._vineyard_socket

    @classmethod
    def _process_out_chunks(cls, op, out_chunks):
        if len(out_chunks) == 1:
            return out_chunks
        else:
            raise NotImplementedError('not implemented')

    @classmethod
    def tile(cls, op):
        return [super().tile(op)[0]]

    @classmethod
    def execute(cls, ctx, op):
        import vineyard

        client = vineyard.connect(op.vineyard_socket)
        tensor = vineyard.GlobalTensorBuilder(client)
        for in_chunk in op.inputs:
            instance_id, tensor_id = ctx[in_chunk.key]
            tensor.add_chunk(instance_id, tensor_id)
        tensor = tensor.build(client)
        tensor.persist(client)

        # store the result object id to execution context
        ctx[op.outputs[0].key] = tensor.id


def tovineyard(vineyard_socket, x):
    x = astensor(x)
    op = TensorVineyardDataStore(vineyard_socket, dtype=x.dtype, sparse=x.issparse())
    return op(x)
