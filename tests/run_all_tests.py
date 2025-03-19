"""Run all tests for the VGPT project."""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))

# Import test functions
from tests.test_n8n_integration import run_test as run_db_test
from tests.simulate_n8n_workflow import run_simulation as run_workflow_simulation

def run_all_tests():
    """Run all tests in sequence."""
    print("======================================")
    print("VGPT Testing Suite")
    print("======================================\n")
    
    print("\n=== PHASE 1: Vector Database Integration Test ===")
    db_test_result = run_db_test()
    
    print("\n=== PHASE 2: n8n Workflow Simulation Test ===")
    workflow_test_result = run_workflow_simulation()
    
    # Print overall summary
    print("\n======================================")
    print("VGPT Testing Summary")
    print("======================================")
    print(f"Vector Database Test: {'PASSED' if db_test_result else 'FAILED'}")
    print(f"n8n Workflow Test: {'PASSED' if workflow_test_result else 'FAILED'}")
    
    overall_success = db_test_result and workflow_test_result
    print(f"\nOverall Test Result: {'PASSED' if overall_success else 'FAILED'}")
    
    return overall_success

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 