import unittest
from autoML import MLSystem

class TestMLSystem(unittest.TestCase):
    def test_entire_workflow(self):
        system = MLSystem()
        result = system.run_entire_workflow("data/train.csv")
        self.assertTrue(result['success'], "The ML system workflow should have completed successfully")
        self.assertLess(result['RMSLE'], 0.2, "The model RMSLE should be below 0.2" )

if __name__ == '__main__':
    unittest.main()