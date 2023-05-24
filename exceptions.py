class ProcessingError(Exception):
    """Base class for other exceptions"""
    def __init__(self, message, original_exception=None):
        self.message = message
        self.original_exception = original_exception
        super().__init__(self.message)


class DataVisualizationError(ProcessingError):
    """Exception raised for errors in the data visualization process."""
    def __init__(self, original_exception=None, 
                 message="Error occurred while visualizing the data."):
        if original_exception:
            message += f" Original error: {original_exception}"
        super().__init__(message, original_exception)


class TestDataProcessingError(ProcessingError):
    """Exception raised for errors in the test data processing process."""
    def __init__(self, original_exception=None, 
                 message="Error occurred while processing the test data."):
        if original_exception:
            message += f" Original error: {original_exception}"
        super().__init__(message, original_exception)


class BestFunctionsSelectionError(ProcessingError):
    """Exception raised for errors in the selection of best functions process."""
    def __init__(self, original_exception=None, 
                 message="Error occurred while selecting best functions."):
        if original_exception:
            message += f" Original error: {original_exception}"
        super().__init__(message, original_exception)