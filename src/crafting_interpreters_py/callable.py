from abc import ABC, abstractmethod
from typing import List
from . import Stmt
from .environment import Environment

class InterpreterCallable(ABC):
    @abstractmethod
    def call(self, interpreter, arguments: List[object]): pass

    @abstractmethod
    def arity(self) -> int: pass

class InterpreterFunction(InterpreterCallable):
    def __init__(self, declaration: Stmt.Function):
        self.declaration = declaration

    def __str__(self):
        return f"<fn {self.declaration.name.lexeme}>"

    def arity(self) -> int:
        return len(self.declaration.params)

    def call(self, interpreter, arguments: List[object]):
        assert len(arguments) == len(self.declaration.params), "len(arguments) != len(params)"
        environment = Environment(interpreter.globals)
        for name, arg in zip(self.declaration.params, arguments):
            environment.define(name.lexeme, arg)

        interpreter.execute_block(self.declaration.body, environment)
        return None
