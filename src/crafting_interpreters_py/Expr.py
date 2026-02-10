from abc import ABC, abstractmethod

class Visitor(ABC):
    @abstractmethod
    def visit_literal(self, expression): pass
    @abstractmethod
    def visit_binary(self, expression): pass
    @abstractmethod
    def visit_grouping(self, expression): pass
    @abstractmethod
    def visit_unary(self, expression): pass

class Expr(ABC):
    @abstractmethod
    def accept(self, visitor: Visitor): pass

class Binary(Expr):
    def __init__(self, left: Expr, operator: Token, right: Expr):
        self.left = left
        self.operator = operator
        self.right = right

    def accept(self, visitor: Visitor):
        return visitor.visit_binary(self)

class Literal(Expr):
    def __init__(self, value):
        self.value = value

    def accept(self, visitor: Visitor):
        return visitor.visit_literal(self)

class Grouping(Expr):
    def __init__(self, expression: Expr):
        self.expression = expression

    def accept(self, visitor: Visitor):
        return visitor.visit_grouping(self)

class Unary(Expr):
    def __init__(self, operator: Token, right: Expr):
        self.operator = operator
        self.right = right

    def accept(self, visitor: Visitor):
        return visitor.visit_unary(self)
    
