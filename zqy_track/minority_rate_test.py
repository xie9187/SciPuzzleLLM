import unittest
import pandas as pd
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from TableState import TableState
from data_utils import calculate_minority_rate


class MinorityRateTest(unittest.TestCase):
    def build_table_state(self, col_values, column_name="col"):
        elem_df = pd.DataFrame(
            {
                "symbol": [f"E{i}" for i in range(len(col_values))],
                "Test": [False] * len(col_values),
            }
        ).set_index("symbol")
        test_df = elem_df.copy()
        state = TableState(elem_df.copy(), test_df.copy())
        state.elem_df.loc[:, column_name] = col_values
        return state

    def test_returns_report_when_majority_exists(self):
        state = self.build_table_state([1, 1, 1, 2])
        report = calculate_minority_rate(state)

        self.assertIsNotNone(report)
        self.assertEqual(report.majority_value, 1)
        self.assertAlmostEqual(report.majority_rate, 0.75)
        self.assertAlmostEqual(report.minority_rate, 0.25)
        self.assertEqual(report.minority_object, (('E3', 2),))

    def test_returns_none_when_majority_not_found(self):
        state = self.build_table_state([1, 1, 2, 2])
        self.assertIsNone(calculate_minority_rate(state))

    def test_ignores_missing_values(self):
        state = self.build_table_state([1, 1, 1, None, 2])
        report = calculate_minority_rate(state)

        self.assertIsNotNone(report)
        self.assertEqual(report.majority_value, 1)
        self.assertAlmostEqual(report.majority_rate, 0.75)
        self.assertAlmostEqual(report.minority_rate, 0.25)
        self.assertEqual(report.minority_object, (('E4', 2),))

    def test_handles_multiple_minority_values(self):
        state = self.build_table_state([1, 1, 1, 2, 3])
        report = calculate_minority_rate(state)

        self.assertIsNotNone(report)
        self.assertEqual(report.majority_value, 1)
        self.assertAlmostEqual(report.majority_rate, 0.6)
        self.assertAlmostEqual(report.minority_rate, 0.4)
        self.assertEqual(report.minority_object, (('E3', 2), ('E4', 3)))

    def test_limits_number_of_minority_values(self):
        state = self.build_table_state([1, 1, 1, 2, 3])
        report = calculate_minority_rate(state, top_k=1)

        self.assertIsNotNone(report)
        self.assertEqual(report.minority_object, (('E3', 2),))
        self.assertAlmostEqual(report.minority_rate, 0.4)

    def test_custom_column_name(self):
        state = self.build_table_state(["A", "A", "A", "B"], column_name="custom")
        report = calculate_minority_rate(state, column="custom")

        self.assertIsNotNone(report)
        self.assertEqual(report.majority_value, "A")
        self.assertAlmostEqual(report.minority_rate, 0.25)
        self.assertEqual(report.minority_object, (('E3', "B"),))

    def test_table_state_grouped_by_col(self):
        elem_df = pd.DataFrame(
            {
                "symbol": [f"E{i}" for i in range(6)],
                "Test": [False] * 6,
                "Category": ["A", "A", "B", "C", "C", "D"],
                "Group": ["X", "X", "Y", "Z", "Z", "Z"],
            }
        ).set_index("symbol")
        state = TableState(elem_df.copy(), elem_df.copy())
        state.elem_df.loc[:, "col"] = [1, 1, 1, 2, 2, 2]
        state.elem_df.loc[:, "row"] = [1, 2, 3, 1, 2, 3]
        state.infer_aset(numeric_tolerances={}, exclude=["row", "col"])

        reports = state.calculate_minority_rate()

        self.assertIn(1, reports)
        self.assertIn(2, reports)
        self.assertIn("Category", reports[1])
        self.assertIn("Group", reports[1])
        self.assertIn("Category", reports[2])
        self.assertNotIn("Group", reports[2])

        category_col1 = reports[1]["Category"]
        self.assertEqual(category_col1.minority_object, (("E2", "B"),))
        self.assertAlmostEqual(category_col1.majority_rate, 2 / 3, places=3)
        self.assertAlmostEqual(category_col1.minority_rate, 1 / 3, places=3)

        group_col1 = reports[1]["Group"]
        self.assertEqual(group_col1.minority_object, (("E2", "Y"),))
        self.assertAlmostEqual(group_col1.majority_rate, 2 / 3, places=3)

        category_col2 = reports[2]["Category"]
        self.assertEqual(category_col2.minority_object, (("E5", "D"),))
        self.assertAlmostEqual(category_col2.majority_rate, 2 / 3, places=3)


if __name__ == "__main__":
    unittest.main()
