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
from ..operands import DataFrameOperand, DataFrameOperandMixin, ObjectType


class DataFrameToVineyard(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.DATAFRAME_STORE_VINEYARD_CHUNK

    _input = KeyField('input')
    # Location for vineyard's socket file
    _vineyard_socket = StringField('vineyard_socket')

    def __init__(self, vineyard_socket=None, dtypes=None, **kw):
        super().__init__(
            _vineyard_socket=vineyard_socket, _dtypes=dtypes,
            _object_type=ObjectType.dataframe, **kw)

    def __call__(self, df):
        return self.new_dataframe([df], shape=(0, 0), dtypes=df.dtypes,
                                  index_value=df.index_value, columns_value=df.columns_value)


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
        merge_op = DataFrameToVineyardMerge(
            vineyard_socket=op.vineyard_socket)
        return merge_op.new_chunks(out_chunks, shape=out_chunks[0].shape,
                                   index=(0,) * out_chunks[0].ndim)

    @classmethod
    def tile(cls, op):
        in_df = op.inputs[0]

        out_chunks = []
        for chunk in in_df.chunks:
            out_chunk = cls._get_out_chunk(op, chunk)
            out_chunks.append(out_chunk)
        out_chunks = cls._process_out_chunks(op, out_chunks)

        new_op = op.copy()
        return new_op.new_dataframes(op.inputs, shape=op.outputs[0].shape, dtypes=in_df.dtypes,
                                     index_value=in_df.index_value,
                                     columns_value=in_df.columns_value,
                                     chunks=out_chunks,
                                     nsplits=((0,) * len(ns) for ns in in_df.nsplits))

    @classmethod
    def execute(cls, ctx, op):
        import vineyard

        client = vineyard.connect(op.vineyard_socket)
        df = vineyard.DataFrameBuilder(client)
        for name, value in ctx[op.inputs[0].key].iteritems():
            df.add(name, vineyard.TensorBuilder.from_numpy(client, value.to_numpy(copy=False)))
        df = df.build(client)
        df.persist(client)

        # store the result object id to execution context
        ctx[op.outputs[0].key] = (client.instance_id, df.id)


class DataFrameToVineyardMerge(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.DATAFRAME_STORE_VINEYARD

    _input = KeyField('input')
    # Location for vineyard's socket file
    _vineyard_socket = StringField('vineyard_socket')

    def __init__(self, vineyard_socket=None, dtypes=None, **kw):
        super().__init__(
            _vineyard_socket=vineyard_socket, _dtypes=dtypes,
            _object_type=ObjectType.dataframe, **kw)

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
        df = vineyard.GlobalDataFrameBuilder(client)
        for in_chunk in op.inputs:
            instance_id, df_id = ctx[in_chunk.key]
            df.add_chunk(instance_id, df_id)
        df = df.build(client)
        df.persist(client)

        # store the result object id to execution context
        ctx[op.outputs[0].key] = df.id


def to_vineyard(df, vineyard_socket):
    op = DataFrameToVineyard(vineyard_socket, dtypes=df.dtypes)
    return op(df)
