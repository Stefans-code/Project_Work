import unittest
import os

def run_all_tests_in_directory(directory='.'):
    """
    Discovers and runs all unittest test cases within the specified directory.

    Args:
        directory (str): The path to the directory containing the test files.
    """
    # Discover all test files in the directory.
    loader = unittest.TestLoader()
    suite = loader.discover(directory)

    # Run the tests.
    runner = unittest.TextTestRunner()
    result = runner.run(suite)

    return result

if __name__ == '__main__':
    # Get the directory where the current script is located.
    test_directory = os.path.dirname(os.path.abspath(__file__))

    # Run all tests in the current directory.
    result = run_all_tests_in_directory(test_directory)

    # Check for failures and exit with the appropriate code.
    if result.failures or result.errors:
        exit(1) # Indicates test failure
    else:
        exit(0) # Indicates test success