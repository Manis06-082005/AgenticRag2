import sys
import traceback
class CustomException(Exception):
    def __init__(self, error):
        self.error=error
        self.details=traceback.format_exc()
        super().__init__(self.details)
    def __str__(self):
        return self.details


# try:
#     result = 10 / 0

# except Exception as e:
#     raise CustomException(e)