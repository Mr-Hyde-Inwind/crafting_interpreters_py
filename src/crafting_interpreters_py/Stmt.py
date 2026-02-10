from abc import ABC, abstractmethod
from . import Expr

class Visitor(ABC):
    @abstractmethod
    def visit_expression_stmt(self, statement: Expression): pass

    @abstractmethod
    def visit_print_stmt(self, statement: Print): pass

class Stmt(ABC):
    def accept(self, visitor: Visitor): pass

class Expression(Stmt):
    def __init__(self, expression: Expr):
        self.expression = expression

    def accept(self, visitor: Visitor):
        return visitor.visit_expression_stmt(self)

class Print(Stmt):
    def __init__(self, expression: Expr):
        self.expression = expression

    def accept(self, visitor: Visitor):
        return visitor.visit_print_stmt(self)
        
