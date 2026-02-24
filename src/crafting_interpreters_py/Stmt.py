from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
from . import Expr
from .token import Token

class Visitor(ABC):
    @abstractmethod
    def visit_expression_stmt(self, stmt: Expression):
        pass

    @abstractmethod
    def visit_print_stmt(self, stmt: Print):
        pass

    @abstractmethod
    def visit_var_stmt(self, stmt: Stmt):
        pass

    @abstractmethod
    def visit_block(self, stmt: Block):
        pass

    @abstractmethod
    def visit_if(self, stmt: If):
        pass

    @abstractmethod
    def visit_while(self, stmt: While):
        pass

    @abstractmethod
    def visit_function(self, stmt: Function):
        pass

class Stmt(ABC):
    @abstractmethod
    def accept(self, visitor: Visitor):
        pass

class Expression(Stmt):
    def __init__(self, expression: Expr.Expr):
        self.expression = expression

    def accept(self, visitor: Visitor):
        return visitor.visit_expression_stmt(self)

class Print(Stmt):
    def __init__(self, expression: Expr.Expr):
        self.expression = expression

    def accept(self, visitor: Visitor):
        return visitor.visit_print_stmt(self)


class Var(Stmt):
    def __init__(self, name, initializer):
        self.name = name
        self.initializer = initializer

    def accept(self, visitor: Visitor):
        return visitor.visit_var_stmt(self)

class Block(Stmt):
    def __init__(self, statements: List[Stmt]):
        self.statements = statements

    def accept(self, visitor: Visitor):
        return visitor.visit_block(self)

class If(Stmt):
    def __init__(self, condition: Expr.Expr, then_branch: Stmt|None, else_branch: Stmt|None):
        self.condition = condition
        self.then_branch = then_branch
        self.else_branch = else_branch

    def accept(self, visitor: Visitor):
        return visitor.visit_if(self)

class While(Stmt):
    def __init__(self, condition: Expr.Expr, body: Stmt):
        self.condition = condition
        self.body = body

    def accept(self, visitor: Visitor):
        return visitor.visit_while(self)

class Function(Stmt):
    def __init__(self, name: Token, params: List[Token], body: List[Stmt|None]):
        self.name = name
        self.params = params
        self.body = body

    def accept(self, visitor: Visitor):
        return visitor.visit_function(self)

