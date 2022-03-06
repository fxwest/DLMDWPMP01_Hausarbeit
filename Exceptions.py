class RowCountMismatchError(Exception):
    def __init__(self):
        error_msg = "Row count mismatch between Training and Ideal dataset"
        self.error_msg = error_msg
