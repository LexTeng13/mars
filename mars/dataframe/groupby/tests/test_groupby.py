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

import mars.dataframe as md
from mars import opcodes
from mars.dataframe.core import DataFrameGroupBy, SeriesGroupBy, DataFrame
from mars.dataframe.groupby.core import DataFrameGroupByOperand, DataFrameShuffleProxy
from mars.dataframe.groupby.aggregation import DataFrameGroupByAgg
from mars.dataframe.operands import ObjectType
from mars.operands import OperandStage
from mars.tests.core import TestBase


class Test(TestBase):
    def testGroupBy(self):
        df = pd.DataFrame({'a': [3, 4, 5, 3, 5, 4, 1, 2, 3],
                           'b': [1, 3, 4, 5, 6, 5, 4, 4, 4]})
        mdf = md.DataFrame(df, chunk_size=2)
        grouped = mdf.groupby('c2')

        self.assertIsInstance(grouped, DataFrameGroupBy)
        self.assertIsInstance(grouped.op, DataFrameGroupByOperand)

        grouped = grouped.tiles()
        self.assertEqual(len(grouped.chunks), 5)
        for chunk in grouped.chunks:
            self.assertIsInstance(chunk.op, DataFrameGroupByOperand)

        series = pd.Series([3, 4, 5, 3, 5, 4, 1, 2, 3])
        ms = md.Series(series, chunk_size=3)
        grouped = ms.groupby(lambda x: x + 1)

        self.assertIsInstance(grouped, SeriesGroupBy)
        self.assertIsInstance(grouped.op, DataFrameGroupByOperand)

        grouped = grouped.tiles()
        self.assertEqual(len(grouped.chunks), 3)
        for chunk in grouped.chunks:
            self.assertIsInstance(chunk.op, DataFrameGroupByOperand)

        with self.assertRaises(TypeError):
            ms.groupby(lambda x: x + 1, as_index=False)

    def testGroupByAgg(self):
        df = pd.DataFrame({'a': np.random.choice([2, 3, 4], size=(20,)),
                           'b': np.random.choice([2, 3, 4], size=(20,))})
        mdf = md.DataFrame(df, chunk_size=3)
        r = mdf.groupby('a').agg('sum')
        self.assertIsInstance(r.op, DataFrameGroupByAgg)
        self.assertIsInstance(r, DataFrame)
        self.assertEqual(r.op.method, 'tree')
        r = r.tiles()
        self.assertEqual(len(r.chunks), 1)
        self.assertEqual(r.chunks[0].op.stage, OperandStage.agg)
        self.assertEqual(len(r.chunks[0].inputs), 1)
        self.assertEqual(len(r.chunks[0].inputs[0].inputs), 2)

        df = pd.DataFrame({'c1': range(10),
                           'c2': np.random.choice(['a', 'b', 'c'], (10,)),
                           'c3': np.random.rand(10)})
        mdf = md.DataFrame(df, chunk_size=2)
        r = mdf.groupby('c2').sum(method='shuffle')

        self.assertIsInstance(r.op, DataFrameGroupByAgg)
        self.assertIsInstance(r, DataFrame)

        r = r.tiles()
        self.assertEqual(len(r.chunks), 5)
        for chunk in r.chunks:
            self.assertIsInstance(chunk.op, DataFrameGroupByAgg)
            self.assertEqual(chunk.op.stage, OperandStage.agg)
            self.assertIsInstance(chunk.inputs[0].op, DataFrameGroupByOperand)
            self.assertEqual(chunk.inputs[0].op.stage, OperandStage.reduce)
            self.assertIsInstance(chunk.inputs[0].inputs[0].op, DataFrameShuffleProxy)
            self.assertIsInstance(chunk.inputs[0].inputs[0].inputs[0].op, DataFrameGroupByOperand)
            self.assertEqual(chunk.inputs[0].inputs[0].inputs[0].op.stage, OperandStage.map)

            agg_chunk = chunk.inputs[0].inputs[0].inputs[0].inputs[0]
            self.assertEqual(agg_chunk.op.stage, OperandStage.map)

        # test unknown method
        with self.assertRaises(ValueError):
            mdf.groupby('c2').sum(method='not_exist')

    def testGroupByApplyTransform(self):
        df1 = pd.DataFrame({'a': [3, 4, 5, 3, 5, 4, 1, 2, 3],
                            'b': [1, 3, 4, 5, 6, 5, 4, 4, 4],
                            'c': list('aabaaddce')})

        def apply_df(df):
            return df.sort_index()

        def apply_series(s):
            return s.sort_index()

        mdf = md.DataFrame(df1, chunk_size=3)

        applied = mdf.groupby('b').apply(apply_df).tiles()
        pd.testing.assert_series_equal(applied.dtypes, df1.dtypes)
        self.assertEqual(applied.shape, (np.nan, 3))
        self.assertEqual(applied.op._op_type_, opcodes.GROUPBY_APPLY)
        self.assertEqual(applied.op.object_type, ObjectType.dataframe)
        self.assertEqual(len(applied.chunks), 3)
        self.assertEqual(applied.chunks[0].shape, (np.nan, 3))
        pd.testing.assert_series_equal(applied.chunks[0].dtypes, df1.dtypes)

        applied = mdf.groupby('b').apply(lambda df: df.a).tiles()
        self.assertEqual(applied.dtype, df1.a.dtype)
        self.assertEqual(applied.shape, (np.nan,))
        self.assertEqual(applied.op._op_type_, opcodes.GROUPBY_APPLY)
        self.assertEqual(applied.op.object_type, ObjectType.series)
        self.assertEqual(len(applied.chunks), 3)
        self.assertEqual(applied.chunks[0].shape, (np.nan,))
        self.assertEqual(applied.chunks[0].dtype, df1.a.dtype)

        applied = mdf.groupby('b').apply(lambda df: df.a.sum()).tiles()
        self.assertEqual(applied.dtype, df1.a.dtype)
        self.assertEqual(applied.shape, (np.nan,))
        self.assertEqual(applied.op._op_type_, opcodes.GROUPBY_APPLY)
        self.assertEqual(applied.op.object_type, ObjectType.series)
        self.assertEqual(len(applied.chunks), 3)
        self.assertEqual(applied.chunks[0].shape, (np.nan,))
        self.assertEqual(applied.chunks[0].dtype, df1.a.dtype)

        applied = mdf.groupby('b').transform(apply_df).tiles()
        pd.testing.assert_series_equal(applied.dtypes, df1.dtypes)
        self.assertEqual(applied.shape, (9, 3))
        self.assertEqual(applied.op._op_type_, opcodes.GROUPBY_TRANSFORM)
        self.assertTrue(applied.op.is_transform)
        self.assertEqual(applied.op.object_type, ObjectType.dataframe)
        self.assertEqual(len(applied.chunks), 3)
        self.assertEqual(applied.chunks[0].shape, (np.nan, 3))
        pd.testing.assert_series_equal(applied.chunks[0].dtypes, df1.dtypes)

        series1 = pd.Series([3, 4, 5, 3, 5, 4, 1, 2, 3])

        ms1 = md.Series(series1, chunk_size=3)
        applied = ms1.groupby(lambda x: x % 3).apply(apply_series).tiles()
        self.assertEqual(applied.dtype, series1.dtype)
        self.assertEqual(applied.shape, (np.nan,))
        self.assertEqual(applied.op._op_type_, opcodes.GROUPBY_APPLY)
        self.assertEqual(applied.op.object_type, ObjectType.series)
        self.assertEqual(len(applied.chunks), 3)
        self.assertEqual(applied.chunks[0].shape, (np.nan,))
        self.assertEqual(applied.chunks[0].dtype, series1.dtype)

        applied = ms1.groupby(lambda x: x % 3).transform(lambda x: x + 1).tiles()
        self.assertEqual(applied.dtype, series1.dtype)
        self.assertEqual(applied.shape, series1.shape)
        self.assertEqual(applied.op._op_type_, opcodes.GROUPBY_TRANSFORM)
        self.assertTrue(applied.op.is_transform)
        self.assertEqual(applied.op.object_type, ObjectType.series)
        self.assertEqual(len(applied.chunks), 3)
        self.assertEqual(applied.chunks[0].shape, (np.nan,))
        self.assertEqual(applied.chunks[0].dtype, series1.dtype)
