#!/usr/bin/env python3
"""
Main test runner for OptiX PyTorch Extension.

This script runs all tests using the unittest framework, compatible with
PyCharm and CLion's test integration.

Usage:
    python run_tests.py [options]

Options:
    --tests TEST_PATTERNS    Run specific tests matching pattern (comma-separated)
    --device N              Specify GPU device ID to use (default: 0)
    --verbose               Show verbose output
    --help                  Show this help message

Examples:
    # Run all tests
    python run_tests.py
    
    # Run only specific test modules
    python run_tests.py --tests test_resource_manager,test_texture
    
    # Run tests on GPU 1 
    python run_tests.py --device 1
    
    # Show verbose output
    python run_tests.py --verbose
"""

import sys
import os
import argparse
import unittest
import importlib
import torch

# Add the project root to the path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Run OptiX PyTorch Extension tests")
    parser.add_argument("--tests", type=str, default="", 
                      help="Comma-separated list of test patterns to run")
    parser.add_argument("--device", type=int, default=0, 
                      help="GPU device ID to use")
    parser.add_argument("--verbose", "-v", action="store_true", 
                      help="Show verbose output")
    return parser.parse_args()

def set_gpu_device(device_id):
    """Set the GPU device for testing"""
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        print(f"Using GPU device: {device_id} - {torch.cuda.get_device_name(device_id)}")
    else:
        print("CUDA not available. Using CPU for tests.")

def run_tests(test_patterns=None, verbose=False):
    """Run tests using unittest"""
    # Find all test modules
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                         "resource_system", "tests")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    if test_patterns:
        # Run specific tests
        for pattern in test_patterns:
            if not pattern.startswith('test_'):
                pattern = f'test_{pattern}'
            
            # Get matching test modules
            try:
                # Try direct import first
                module_name = f"resource_system.tests.{pattern}"
                module = importlib.import_module(module_name)
                sub_suite = loader.loadTestsFromModule(module)
                suite.addTest(sub_suite)
                print(f"Added tests from {module_name}")
            except ImportError:
                # Try pattern matching
                pattern_suite = loader.discover(test_dir, pattern=f"{pattern}*.py")
                if pattern_suite.countTestCases() > 0:
                    suite.addTest(pattern_suite)
                    print(f"Added tests matching pattern {pattern}*.py")
                else:
                    print(f"Warning: No tests found matching {pattern}")
    else:
        # Run all tests
        suite = loader.discover(test_dir, pattern="test_*.py")
    
    # Run tests
    verbosity = 2 if verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def main():
    args = parse_args()
    
    # Set GPU device
    set_gpu_device(args.device)
    
    # Parse test patterns
    test_patterns = args.tests.split(',') if args.tests else None
    
    # Run tests
    success = run_tests(
        test_patterns=test_patterns,
        verbose=args.verbose
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())