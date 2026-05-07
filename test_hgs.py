import unittest
import numpy as np
from components.execution.hygese_solver import solve_hgs

class TestHGSSolver(unittest.TestCase):
    def test_solve_hgs_basic(self):
        """Test HGS with a tiny valid VRP instance."""
        # Simple triangle: Depot at (0,0), clients at (0,1) and (1,0)
        data = {
            'coordinates': {
                0: (0.0, 0.0),
                1: (0.0, 1.0),
                2: (1.0, 0.0)
            },
            'demands': [0, 1, 1],
            'capacity': 10,
            'num_vehicles': 1,
            'depot': 0,
            'num_nodes': 3
        }
        
        # Run solver
        cost, neighbors, routes, history, ttt = solve_hgs(
            data, 
            time_limit_seconds=1, 
            no_improvement_limit_iterations=100
        )
        
        # Verify output structure
        self.assertIsNotNone(cost)
        self.assertGreater(cost, 0)
        self.assertIsInstance(routes, list)
        self.assertGreater(len(routes), 0)
        self.assertIsInstance(history, list)
        
    def test_solve_hgs_no_coords(self):
        """Test that HGS fails gracefully if no coords/matrix provided."""
        data = {
            'demands': [0, 1],
            'capacity': 10,
            'num_vehicles': 1
        }
        with self.assertRaises(ValueError):
            solve_hgs(data)

if __name__ == '__main__':
    unittest.main()
