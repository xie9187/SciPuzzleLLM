import os
import unittest
import pandas as pd


class TestInterpolatorTDD(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the provided table with row/col positions
        csv_path = os.path.join(os.getcwd(), "table_elem.csv")
        cls.df = pd.read_csv(csv_path)

    def test_four_neighbor_average_NewElem3(self):
        """
        NewElem3 is at (row=2, col=10). Its four direct neighbors are all numeric:
        up(1,10)=20.1797, down(3,10)=69.723, left(2,9)=30.973762, right(2,11)=47.867
        Expected = average of four = 42.1858655
        """
        from zqy_track.interpolator import Interpolator
        itp = Interpolator(self.df, main_attribute="Attribute2")
        pred = itp.predict_at(2, 10)
        self.assertTrue(abs(pred - 42.1858655) < 1e-6)

    def test_top_edge_rule_NewElem6(self):
        """
        NewElem6 at top edge (row=1, col=9). Updated rule:
        mean of horizontal context two steps: avg(A(1,8),A(1,7),A(1,10),A(1,11)).
        = avg(12.0107, 10.811, 20.1797, 22.98976928) = 16.49779232
        Must lie within range [15.999, 20.180] (it does).
        """
        from zqy_track.interpolator import Interpolator
        itp = Interpolator(self.df, main_attribute="Attribute2")
        pred = itp.predict_at(1, 9)
        self.assertTrue(abs(pred - 16.49779232) < 1e-9)
        lo, hi = 15.999, 20.180
        self.assertTrue(lo <= pred <= hi)

    def test_vertical_new_elem_horizontal_two_step_mean_NewElem7(self):
        """
        NewElem7 at (row=3, col=4) with vertical NewElem context.
        Use horizontal two-step mean of available values:
        mean(A(3,2)=58.6934, A(3,3)=58.933195, A(3,5)=63.546, A(3,6)=54.938045)
        = 59.02766 (within range [58.933, 63.546]).
        """
        from zqy_track.interpolator import Interpolator
        itp = Interpolator(self.df, main_attribute="Attribute2")
        pred = itp.predict_at(3, 4)
        self.assertTrue(abs(pred - 59.02766) < 1e-6)

    def test_corner_top_left_NewElem0(self):
        """
        NewElem0 is at top-left corner (row=min=1, col=min=1).
        Corner rule uses horizontal extrapolation when possible:
        pred_raw = 2*A(1,2) - A(1,3) = 2*6.941 - 14.0067 = -0.1247.
        Since it's outside range [1.008, 6.941], fallback to midpoint 3.9745.
        """
        from zqy_track.interpolator import Interpolator
        itp = Interpolator(self.df, main_attribute="Attribute2")
        pred = itp.predict_at(1, 1)
        self.assertTrue(abs(pred - 3.9745) < 1e-6)

    def test_left_edge_cross_neighbor_NewElem4(self):
        """
        NewElem4 at (row=3, col=1) is at left edge.
        Direct numeric neighbors: right (3,2)=58.6934; up/down are NewElem; left none.
        Cross-edge neighbor per spec: (row-1, max_col) = (2,11) = 47.867.
        Expected mean = (58.6934 + 47.867) / 2 = 53.2802, within range [50.941, 54.938].
        """
        from zqy_track.interpolator import Interpolator
        itp = Interpolator(self.df, main_attribute="Attribute2")
        pred = itp.predict_at(3, 1)
        self.assertTrue(abs(pred - 53.2802) < 1e-6)

    def test_right_edge_cross_neighbor(self):
        """
        Right edge at (row=2, col=max=11). Cross-edge neighbor considered (row+1, min_col)=(3,1) which is NewElem (no numeric),
        so mean of available direct numeric neighbors: up(1,11)=22.98976928, down(3,11)=74.9216.
        Expected = (22.98976928 + 74.9216) / 2 = 48.95568464.
        """
        from zqy_track.interpolator import Interpolator
        itp = Interpolator(self.df, main_attribute="Attribute2")
        pred = itp.predict_at(2, 11)
        self.assertTrue(abs(pred - 48.95568464) < 1e-9)

    def test_bottom_edge_horizontal_two_step_NewElem2(self):
        """
        Bottom edge at NewElem2 (row=6, col=2). Horizontal available within two steps:
        (6,1)=140.90765, (6,3)=144.242. Mean=142.574825 within range [140.908, 144.242].
        """
        from zqy_track.interpolator import Interpolator
        itp = Interpolator(self.df, main_attribute="Attribute2")
        pred = itp.predict_at(6, 2)
        self.assertTrue(abs(pred - 142.574825) < 1e-2)

    def test_predict_all_NewElem(self):
        """
        Test prediction for all NewElem entries in the table.
        """
        from zqy_track.interpolator import Interpolator
        itp = Interpolator(self.df, main_attribute="Attribute2")
        
        # Get all NewElem positions
        new_elems = []
        for idx, row in self.df.iterrows():
            if str(row['Element']).startswith('NewElem'):
                new_elems.append((int(row['row']), int(row['col'])))
        
        # Predict and verify each NewElem
        for row, col in new_elems:
            try:
                pred = itp.predict_at(row, col)
                print(f"Predicted NewElem at ({row},{col}): {pred}")
                # Verify prediction is within range if available
                ci = itp._get(row, col)
                if ci and ci.range_lo is not None and ci.range_hi is not None:
                    self.assertTrue(ci.range_lo <= pred <= ci.range_hi,
                                  f"Prediction {pred} not in range [{ci.range_lo}, {ci.range_hi}] for ({row},{col})")
            except ValueError as e:
                self.fail(f"Failed to predict NewElem at ({row},{col}): {str(e)}")
    

if __name__ == "__main__":
    unittest.main()
