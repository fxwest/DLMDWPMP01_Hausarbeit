"""
This file contains all user defined Exceptions.
"""


class RowCountMismatchError(Exception):
    """
    The Ideal and Training Dataset do not have the same number of rows.
    """
    def __init__(self):
        error_msg = "Row count mismatch between Training and Ideal dataset"
        self.error_msg = error_msg


class EmptyLeastSquareError(Exception):
    """
    The needed Least Square Dataframe is empty.
    Please run Least Square Calculation first.
    """
    def __init__(self):
        error_msg = "The needed Least Square Dataframe is empty. Please run Least Square Calculation first!"
        self.error_msg = error_msg


class EmptyBestFitError(Exception):
    """
    The needed Best Fit Dataframe is empty.
    Please run Select Best Fit Calculation first.
    """
    def __init__(self):
        error_msg = "The needed Best Fit Dataframe is empty. Please run Select Best Fit Calculation first!"
        self.error_msg = error_msg


class EmptyMatchingResultError(Exception):
    """
    The needed Matching Result Dataframe is empty.
    Please run Matching Functions Calculation first.
    """
    def __init__(self):
        error_msg = "The needed Matching Result  Dataframe is empty. Please run Matching Functions Calculation first!"
        self.error_msg = error_msg
