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

import operator

from ... import opcodes as OperandDef
from ...utils import classproperty
from .core import DataFrameBinOpMixin, DataFrameBinOp


class DataFrameTrueDiv(DataFrameBinOp, DataFrameBinOpMixin):
    _op_type_ = OperandDef.DIV

    _func_name = 'truediv'
    _rfunc_name = 'rtruediv'

    @classproperty
    def _operator(self):
        return operator.truediv


def truediv(df, other, axis='columns', level=None, fill_value=None):
    op = DataFrameTrueDiv(axis=axis, level=level, fill_value=fill_value, lhs=df, rhs=other)
    return op(df, other)


def rtruediv(df, other, axis='columns', level=None, fill_value=None):
    op = DataFrameTrueDiv(axis=axis, level=level, fill_value=fill_value, lhs=other, rhs=df)
    return op.rcall(df, other)
