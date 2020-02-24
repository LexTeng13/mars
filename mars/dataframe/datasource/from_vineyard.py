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
import pandas as pd
from pandas.core.internals.blocks import Block
from pandas.core.internals.managers import BlockManager

from ... import opcodes as OperandDef
from ...serialize import StringField, UInt64Field
from ...context import get_context, RunningMode
from ...utils import calc_nsplits
from ..operands import DataFrameOperand, DataFrameOperandMixin, ObjectType


class DataFrameNoInput(DataFrameOperand, DataFrameOperandMixin):
    """
    DataFrame operand with no inputs.
    """

    def check_inputs(self, inputs):
        # no inputs
        if inputs and len(inputs) > 0:
            raise ValueError("DataFrame data source has no inputs")

    def _new_chunks(self, inputs, kws=None, **kw):
        shape = kw.get('shape', None)
        self.extra_params['shape'] = shape  # set shape to make the operand key different
        return super()._new_chunks(inputs, kws=kws, **kw)

    def _new_tileables(self, inputs, kws=None, **kw):
        shape = kw.get('shape', None)
        self.extra_params['shape'] = shape  # set shape to make the operand key different
        return super()._new_tileables(inputs, kws=kws, **kw)

    def __call__(self, shape, chunk_size=None):
        return self.new_dataframe(None, shape, dtypes=[], raw_chunk_size=chunk_size)


class DataFrameFromVineyard(DataFrameNoInput):
    _op_type_ = OperandDef.DATAFRAME_FROM_VINEYARD

    # Location for vineyard's socket file
    _vineyard_socket = StringField('vineyard_socket')
    # ObjectID in vineyard
    _object_id = UInt64Field('object_id')

    def __init__(self, vineyard_socket=None, object_id=None, **kw):
        super().__init__(
            _vineyard_socket=vineyard_socket, _object_id=object_id,
            _object_type=ObjectType.dataframe, **kw)

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
        client = vineyard.connect(op.vineyard_socket)
        df = client.get(op.object_id)

        instances = df.instances
        chunk_map = {}
        for instance_id in instances:
            for chunk_id in df.local_chunks(instance_id):
                chunk = client.get_meta(chunk_id) # FIXME: use get meta
                chunk_index = (int(chunk['chunk_index_row']), int(chunk['chunk_index_column']))
                chunk_map[chunk_index] = (instance_id, chunk['id'], (None, int(chunk['column_size'])))

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
        return new_op.new_dataframes(op.inputs, shape=(None, None), dtypes=[],
                                     chunks=out_chunks, nsplits=nsplits)

    @classmethod
    def execute(cls, ctx, op):
        import vineyard

        chunk = op.outputs[0]

        client = vineyard.connect(op.vineyard_socket)
        # chunk has a tensor chunk
        df_chunk = client.get(op.object_id)

        if not df_chunk.columns:
            return pd.DataFrame()

        # ensure zero-copy
        blocks = []
        index_size = 0
        for idx, name in enumerate(df_chunk.columns):
            value = df_chunk[name].numpy()
            blocks.append(Block(np.expand_dims(value, 0), slice(idx, idx + 1, 1)))
            index_size = len(value)
        ctx[chunk.key] = pd.DataFrame(BlockManager(blocks, [df_chunk.columns, np.arange(index_size)]))


def from_vineyard(vineyard_socket, object_id):
    op = DataFrameFromVineyard(vineyard_socket=vineyard_socket,
                               object_id=object_id)
    return op(shape=None)
