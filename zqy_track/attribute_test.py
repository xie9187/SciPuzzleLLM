# -*- coding: utf-8 -*-
# 运行：
#   python -m unittest -v
import unittest
import math
import numpy as np
import pandas as pd
import sys
sys.path.append("..")
# 待你在 feature_space/attributes.py 中实现以下类与方法
from data_utils import Attribute, AttributeSet


class BaseFixture(unittest.TestCase):
    def make_df(self) -> pd.DataFrame:
        # 行为样本，列为属性；is_test 标记是否属于测试集
        # idx=2 为测试集，不参与空间统计
        return pd.DataFrame({
            "num_a":   [0.00, 0.10, 0.19, 0.21, np.nan],   # 数值列，含 NaN
            "num_b":   [-1.2, -1.05, -0.95, -0.49, -0.51], # 数值列，含负数
            "color":   ["red", "blue", "red", "green", "blue"],  # 离散列
            "shape":   ["circle", "square", "square", "circle", "circle"],  # 离散列
            "Test": [False,  False,   True,   False,  False]
        })


class TestAttributeNumeric(BaseFixture):
    def test_numeric_space_size_with_tolerance(self):
        df = self.make_df()
        series = df.loc[~df["Test"], "num_a"]  # [0.00, 0.10, 0.21, NaN]
        # 规则：bin = floor(value / tolerance)，忽略 NaN
        attr = Attribute.numeric(name="num_a", tolerance=0.1)
        self.assertEqual(attr.space_size(series), 3)  # 0,1,2 三个 bin

    def test_numeric_space_size_negative_values(self):
        df = self.make_df()
        series = df.loc[~df["Test"], "num_b"]  # [-1.2, -1.05, -0.49, -0.51]
        attr = Attribute.numeric(name="num_b", tolerance=0.2)
        # floor(x/0.2) -> [-6, -6, -3, -3] → 唯一 {-6,-3} → 2
        self.assertEqual(attr.space_size(series), 2)

    def test_numeric_all_nan_returns_zero(self):
        s = pd.Series([np.nan, np.nan, np.nan], name="nan_col")
        attr = Attribute.numeric(name="nan_col", tolerance=0.1)
        self.assertEqual(attr.space_size(s), 0)


class TestAttributeCategorical(BaseFixture):
    def test_categorical_basic(self):
        df = self.make_df()
        s_color = df.loc[~df["Test"], "color"]
        s_shape = df.loc[~df["Test"], "shape"]
        self.assertEqual(Attribute.categorical("color").space_size(s_color), 3)  # red/blue/green
        self.assertEqual(Attribute.categorical("shape").space_size(s_shape), 2)  # circle/square

    def test_categorical_ignores_nan(self):
        s = pd.Series(["a", None, np.nan, "a", "b"], name="cat")
        self.assertEqual(Attribute.categorical("cat").space_size(s), 2)


class TestAttributeSetFactory(BaseFixture):
    def test_infer_from_df_excludes_test_flag(self):
        df = self.make_df()
        aset = AttributeSet.infer_from_df(
            df=df,
            test_flag="Test",
            numeric_tolerances={"num_a": 0.1, "num_b": 0.2}  # 可选：为数值列提供 tol
        )
        names_types = {(a.name, a.kind) for a in aset.attributes}
        self.assertIn(("num_a", "numeric"), names_types)
        self.assertIn(("num_b", "numeric"), names_types)
        self.assertIn(("color", "categorical"), names_types)
        self.assertIn(("shape", "categorical"), names_types)
        # 不应包含 is_test 本身
        self.assertNotIn(("Test", "categorical"), names_types)
        self.assertNotIn(("Test", "numeric"), names_types)

    def test_infer_uses_default_tolerance_when_missing(self):
        df = self.make_df()
        aset = AttributeSet.infer_from_df(
            df=df,
            test_flag="Test",
            numeric_tolerances={"num_a": 0.1},   # 未提供 num_b 的 tol
            default_tolerance=0.2                 # 应用于缺省的数值列
        )
        tol_map = {a.name: a.tolerance for a in aset.attributes if a.kind == "numeric"}
        self.assertEqual(tol_map["num_a"], 0.1)
        self.assertEqual(tol_map["num_b"], 0.2)


class TestAttributeSetCompute(BaseFixture):
    def test_space_sizes_and_total_and_shares(self):
        df = self.make_df()
        aset = AttributeSet(
            attributes=[
                Attribute.numeric("num_a", tolerance=0.1),
                Attribute.numeric("num_b", tolerance=0.2),
                Attribute.categorical("color"),
                Attribute.categorical("shape"),
            ],
            test_flag="Test"
        )
        sizes = aset.space_sizes(df)  # 仅训练集
        self.assertEqual(sizes, {"num_a": 3, "num_b": 2, "color": 3, "shape": 2})

        total = aset.total_space(df)
        self.assertEqual(total, 10)

        shares = aset.space_shares(df)
        self.assertTrue(math.isclose(shares["num_a"], 3/10, rel_tol=1e-9))
        self.assertTrue(math.isclose(shares["num_b"], 2/10, rel_tol=1e-9))
        self.assertTrue(math.isclose(shares["color"], 3/10, rel_tol=1e-9))
        self.assertTrue(math.isclose(shares["shape"], 2/10, rel_tol=1e-9))
        self.assertEqual(sum(shares.values()), 1.0)

    def test_zero_total_space_handles_gracefully(self):
        df = pd.DataFrame({
            "num_only_nan": [np.nan, np.nan],
            "cat_only_nan": [np.nan, None],
            "Test": [False, False]
        })
        aset = AttributeSet(
            attributes=[
                Attribute.numeric("num_only_nan", tolerance=0.1),
                Attribute.categorical("cat_only_nan"),
            ],
            test_flag="Test"
        )
        sizes = aset.space_sizes(df)
        self.assertEqual(sizes["num_only_nan"], 0)
        self.assertEqual(sizes["cat_only_nan"], 0)
        self.assertEqual(aset.total_space(df), 0)
        self.assertEqual(aset.space_shares(df), {"num_only_nan": 0.0, "cat_only_nan": 0.0})


if __name__ == "__main__":
    unittest.main()
