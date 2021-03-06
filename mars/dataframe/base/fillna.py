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

import itertools

import numpy as np
import pandas as pd

from ... import opcodes
from ...config import options
from ...core import Base, Entity
from ...operands import OperandStage
from ...serialize import StringField, AnyField, BoolField, Int64Field
from ..align import align_dataframe_dataframe, align_dataframe_series, align_series_series
from ..core import DATAFRAME_TYPE
from ..operands import DataFrameOperandMixin, DataFrameOperand, ObjectType


class FillNA(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = opcodes.FILL_NA

    _value = AnyField('value', on_serialize=lambda x: x.data if isinstance(x, Entity) else x)
    _method = StringField('method')
    _axis = AnyField('axis')
    _limit = Int64Field('limit')
    _downcast = AnyField('downcast')
    _use_inf_as_na = BoolField('use_inf_as_na')

    _output_limit = Int64Field('output_limit')

    def __init__(self, value=None, method=None, axis=None, limit=None, downcast=None,
                 use_inf_as_na=None, sparse=None, stage=None, gpu=None, object_type=None,
                 output_limit=None, **kw):
        super().__init__(_value=value, _method=method, _axis=axis, _limit=limit, _downcast=downcast,
                         _use_inf_as_na=use_inf_as_na, _sparse=sparse, _stage=stage, _gpu=gpu,
                         _object_type=object_type, _output_limit=output_limit, **kw)

    @property
    def value(self):
        return self._value

    @property
    def method(self):
        return self._method

    @property
    def axis(self):
        return self._axis

    @property
    def limit(self):
        return self._limit

    @property
    def downcast(self):
        return self._downcast

    @property
    def use_inf_as_na(self):
        return self._use_inf_as_na

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        if self._method is None and len(inputs) > 1:
            self._value = self._inputs[1]

    @property
    def output_limit(self):
        return self._output_limit or 1

    @staticmethod
    def _get_first_slice(op, df, end):
        if op.method == 'bfill':
            if op.object_type == ObjectType.series:
                return df.iloc[:end]
            else:
                if op.axis == 1:
                    return df.iloc[:, :end]
                else:
                    return df.iloc[:end, :]
        else:
            if op.object_type == ObjectType.series:
                return df.iloc[-end:]
            else:
                if op.axis == 1:
                    return df.iloc[:, -end:]
                else:
                    return df.iloc[-end:, :]

    @classmethod
    def _execute_map(cls, ctx, op):
        input_data = ctx[op.inputs[0].key]
        limit = op.limit
        axis = op.axis
        method = op.method

        filled = input_data.fillna(method=method, axis=axis, limit=limit, downcast=op.downcast)
        ctx[op.outputs[0].key] = cls._get_first_slice(op, filled, 1)
        del filled

    @classmethod
    def _execute_combine(cls, ctx, op):
        axis = op.axis
        method = op.method
        limit = op.limit

        input_data = ctx[op.inputs[0].key]
        if limit is not None:
            n_summaries = (len(op.inputs) - 1) // 2
            summaries = [ctx[inp.key] for inp in op.inputs[1:1 + n_summaries]]
        else:
            summaries = [ctx[inp.key] for inp in op.inputs[1:]]

        if not summaries:
            ctx[op.outputs[0].key] = input_data.fillna(method=method, axis=axis, limit=limit,
                                                       downcast=op.downcast)
            return

        valid_summary = cls._get_first_slice(
            op, pd.concat(summaries, axis=axis).fillna(method=method, axis=axis), 1)

        if method == 'bfill':
            concat_df = pd.concat([input_data, valid_summary], axis=axis)
        else:
            concat_df = pd.concat([valid_summary, input_data], axis=axis)

        concat_df.fillna(method=method, axis=axis, inplace=True, limit=limit,
                         downcast=op.downcast)
        ctx[op.outputs[0].key] = cls._get_first_slice(op, concat_df, -1)

    @classmethod
    def execute(cls, ctx, op):
        try:
            pd.set_option('mode.use_inf_as_na', op.use_inf_as_na)
            if op.stage == OperandStage.map:
                cls._execute_map(ctx, op)
            elif op.stage == OperandStage.combine:
                cls._execute_combine(ctx, op)
            else:
                input_data = ctx[op.inputs[0].key]
                value = getattr(op, 'value', None)
                if isinstance(op.value, (Base, Entity)):
                    value = ctx[op.value.key]
                ctx[op.outputs[0].key] = input_data.fillna(
                    value=value, method=op.method, axis=op.axis, limit=op.limit, downcast=op.downcast)
        finally:
            pd.reset_option('mode.use_inf_as_na')

    @classmethod
    def _tile_one_by_one(cls, op):
        in_df = op.inputs[0]
        in_value_df = op.value if isinstance(op.value, (Base, Entity)) else None
        df = op.outputs[0]

        new_chunks = []
        for c in in_df.chunks:
            inputs = [c] if in_value_df is None else [c, in_value_df.chunks[0]]
            kw = c.params
            new_op = op.copy().reset_key()
            new_chunks.append(new_op.new_chunk(inputs, **kw))

        kw = df.params.copy()
        kw.update(dict(chunks=new_chunks))
        new_op = op.copy().reset_key()
        return new_op.new_tileables(op.inputs, **kw)

    @classmethod
    def _build_combine(cls, op, input_chunks, summary_chunks, idx, is_forward=True):
        c = input_chunks[idx]

        summaries_to_concat = []

        idx_range = list(range(idx) if is_forward else range(idx + 1, len(summary_chunks)))
        for i in idx_range:
            summaries_to_concat.append(summary_chunks[i])

        new_chunk_op = op.copy().reset_key()
        new_chunk_op._stage = OperandStage.combine

        chunks_to_concat = [c] + summaries_to_concat
        return new_chunk_op.new_chunk(chunks_to_concat, **c.params)

    @classmethod
    def _tile_directional_dataframe(cls, op):
        in_df = op.inputs[0]
        df = op.outputs[0]
        is_forward = op.method == 'ffill'

        n_rows, n_cols = in_df.chunk_shape

        # map to get individual results and summaries
        src_chunks = np.empty(in_df.chunk_shape, dtype=np.object)
        summary_chunks = np.empty(in_df.chunk_shape, dtype=np.object)
        for c in in_df.chunks:
            new_chunk_op = op.copy().reset_key()
            new_chunk_op._stage = OperandStage.map
            if op.axis == 1:
                summary_shape = (c.shape[0], 1)
            else:
                summary_shape = (1, c.shape[1])
            src_chunks[c.index] = c
            summary_chunks[c.index] = new_chunk_op.new_chunk(
                [c], shape=summary_shape, dtypes=df.dtypes)

        # combine summaries into results
        output_chunk_array = np.empty(in_df.chunk_shape, dtype=np.object)
        if op.axis == 1:
            for row in range(n_rows):
                row_src = src_chunks[row, :]
                row_summaries = summary_chunks[row, :]
                for col in range(n_cols):
                    output_chunk_array[row, col] = cls._build_combine(
                        op, row_src, row_summaries, col, is_forward)
        else:
            for col in range(n_cols):
                col_src = src_chunks[:, col]
                col_summaries = summary_chunks[:, col]
                for row in range(n_rows):
                    output_chunk_array[row, col] = cls._build_combine(
                        op, col_src, col_summaries, row, is_forward)

        output_chunks = list(output_chunk_array.reshape((n_rows * n_cols,)))
        new_op = op.copy().reset_key()
        return new_op.new_tileables(op.inputs, shape=in_df.shape, nsplits=in_df.nsplits,
                                    chunks=output_chunks, dtypes=df.dtypes,
                                    index_value=df.index_value, columns_value=df.columns_value)

    @classmethod
    def _tile_directional_series(cls, op):
        in_series = op.inputs[0]
        series = op.outputs[0]
        forward = op.method == 'ffill'

        # map to get individual results and summaries
        summary_chunks = np.empty(in_series.chunk_shape, dtype=np.object)
        for c in in_series.chunks:
            new_chunk_op = op.copy().reset_key()
            new_chunk_op._stage = OperandStage.map
            summary_chunks[c.index] = new_chunk_op.new_chunk([c], shape=(1,), dtype=series.dtype)

        # combine summaries into results
        output_chunks = [
            cls._build_combine(op, in_series.chunks, summary_chunks, i, forward)
            for i in range(len(in_series.chunks))
        ]
        new_op = op.copy().reset_key()
        return new_op.new_tileables(op.inputs, shape=in_series.shape, nsplits=in_series.nsplits,
                                    chunks=output_chunks, dtype=series.dtype,
                                    index_value=series.index_value)

    @classmethod
    def _tile_both_dataframes(cls, op):
        in_df = op.inputs[0]
        in_value = op.inputs[1]
        df = op.outputs[0]

        nsplits, out_shape, left_chunks, right_chunks = align_dataframe_dataframe(in_df, in_value)
        out_chunk_indexes = itertools.product(*(range(s) for s in out_shape))

        out_chunks = []
        for idx, left_chunk, right_chunk in zip(out_chunk_indexes, left_chunks, right_chunks):
            out_chunk = op.copy().reset_key().new_chunk([left_chunk, right_chunk],
                                                        shape=(np.nan, np.nan), index=idx)
            out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_dataframes(op.inputs, df.shape,
                                     nsplits=tuple(tuple(ns) for ns in nsplits),
                                     chunks=out_chunks, dtypes=df.dtypes,
                                     index_value=df.index_value, columns_value=df.columns_value)

    @classmethod
    def _tile_dataframe_series(cls, op):
        left, right = op.inputs[0], op.inputs[1]
        df = op.outputs[0]

        nsplits, out_shape, left_chunks, right_chunks = align_dataframe_series(left, right, axis=1)
        out_chunk_indexes = itertools.product(*(range(s) for s in out_shape))

        out_chunks = []
        for out_idx, df_chunk in zip(out_chunk_indexes, left_chunks):
            series_chunk = right_chunks[out_idx[1]]
            kw = dict(shape=(np.nan, df_chunk.shape[1]), columns_value=df_chunk.columns_value)
            out_chunk = op.copy().reset_key().new_chunk([df_chunk, series_chunk], index=out_idx, **kw)
            out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_dataframes(op.inputs, df.shape,
                                     nsplits=tuple(tuple(ns) for ns in nsplits),
                                     chunks=out_chunks, dtypes=df.dtypes,
                                     index_value=df.index_value, columns_value=df.columns_value)

    @classmethod
    def _tile_both_series(cls, op):
        left, right = op.inputs[0], op.inputs[1]
        df = op.outputs[0]

        nsplits, out_shape, left_chunks, right_chunks = align_series_series(left, right)

        out_chunks = []
        for idx, left_chunk, right_chunk in zip(range(out_shape[0]), left_chunks, right_chunks):
            out_chunk = op.copy().reset_key().new_chunk([left_chunk, right_chunk],
                                                        shape=(np.nan,), index=(idx,))
            out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_seriess(op.inputs, df.shape,
                                  nsplits=tuple(tuple(ns) for ns in nsplits),
                                  chunks=out_chunks, dtype=df.dtype,
                                  index_value=df.index_value, name=df.name)

    @classmethod
    def tile(cls, op):
        in_df = op.inputs[0]
        if len(in_df.chunks) == 1 and \
                (not isinstance(op.value, (Base, Entity)) or len(op.value.chunks) == 1):
            return cls._tile_one_by_one(op)
        elif op.method is not None:
            if op.object_type == ObjectType.dataframe:
                return cls._tile_directional_dataframe(op)
            else:
                return cls._tile_directional_series(op)
        elif not isinstance(op.value, (Base, Entity)):
            return cls._tile_one_by_one(op)
        elif isinstance(op.value, DATAFRAME_TYPE):
            return cls._tile_both_dataframes(op)
        elif op.object_type == ObjectType.dataframe:
            return cls._tile_dataframe_series(op)
        else:
            return cls._tile_both_series(op)

    def __call__(self, a, value_df=None):
        method = getattr(self, 'method', None)
        if method == 'backfill':
            method = 'bfill'
        elif method == 'pad':
            method = 'ffill'
        self._method = method
        axis = getattr(self, 'axis', None) or 0
        if axis == 'index':
            axis = 0
        elif axis == 'columns':
            axis = 1
        self._axis = axis

        inputs = [a]
        if value_df is not None:
            inputs.append(value_df)
        if isinstance(a, DATAFRAME_TYPE):
            return self.new_dataframe(inputs, shape=a.shape, dtypes=a.dtypes, index_value=a.index_value,
                                      columns_value=a.columns_value)
        else:
            return self.new_series(inputs, shape=a.shape, dtype=a.dtype, index_value=a.index_value)


def fillna(df, value=None, method=None, axis=None, inplace=False, limit=None, downcast=None):
    if value is None and method is None:
        raise ValueError("Must specify a fill 'value' or 'method'.")
    elif value is not None and method is not None:
        raise ValueError("Cannot specify both 'value' and 'method'.")

    if df.op.object_type == ObjectType.series and isinstance(value, (DATAFRAME_TYPE, pd.DataFrame)):
        raise ValueError('"value" parameter must be a scalar, dict or Series, but you passed a "%s"'
                         % type(value).__name__)

    if downcast is not None:
        raise NotImplementedError('Currently argument "downcast" is not implemented yet')
    if limit is not None:
        raise NotImplementedError('Currently argument "limit" is not implemented yet')

    if isinstance(value, (Base, Entity)):
        value, value_df = None, value
    else:
        value_df = None

    use_inf_as_na = options.dataframe.mode.use_inf_as_na
    op = FillNA(value=value, method=method, axis=axis, limit=limit, downcast=downcast,
                use_inf_as_na=use_inf_as_na, object_type=df.op.object_type)
    out_df = op(df, value_df=value_df)
    if inplace:
        df.data = out_df.data
    else:
        return out_df


def ffill(df, axis=None, inplace=False, limit=None, downcast=None):
    return fillna(df, method='ffill', axis=axis, inplace=inplace, limit=limit, downcast=downcast)


def bfill(df, axis=None, inplace=False, limit=None, downcast=None):
    return fillna(df, method='bfill', axis=axis, inplace=inplace, limit=limit, downcast=downcast)
