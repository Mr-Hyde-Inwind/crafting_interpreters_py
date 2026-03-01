import threading
from .token import Token

class RuntimeException(RuntimeError):
    def __init__(self, token: Token, message: str):
        super().__init__(message)
        self.token = token

class Return(RuntimeError):
    def __init__(self, value: object):
        self.value = value


class ErrorManager():
    instance_dict = {}

    @staticmethod
    def get_manager_instance():
        tid = threading.get_native_id()
        return ErrorManager.instance_dict.setdefault(tid, ErrorManager())

    def __init__(self):
        self.__had_error = False
        self.__had_runtimeerror = False

    @property
    def had_error(self):
        return self.__had_error

    @property
    def had_runtimeerror(self):
        return self.__had_runtimeerror

    def set_error(self, line: int, where: str, message: str) -> str:
        self.__had_error = True
        return f'[line {line}] Error {where}: {message}'

    def set_runtimeerror(self, error: RuntimeException) -> str:
        self.__had_runtimeerror = True
        return f'{str(error)}\n[line {error.token.line}]'

    def reset_error(self):
        self.__had_error = False
        self.__had_runtimeerror = False

