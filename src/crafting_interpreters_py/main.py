import argparse
import operator
from pathlib import Path
from enum import Enum, auto
from typing import List, Dict
from abc import ABC, abstractmethod
from . import Expr
from . import Stmt

class TokenType(Enum):
    LEFT_PAREN = auto(); RIGHT_PAREN = auto()
    LEFT_BRACE = auto(); RIGHT_BRACE = auto()
    COMMA = auto(); SEMICOLON = auto(); DOT = auto()
    MINUS = auto(); PLUS = auto(); SLASH = auto(); STAR = auto()

    BANG = auto(); BANG_EQUAL = auto()
    EQUAL = auto(); EQUAL_EQUAL = auto()
    GREATER = auto(); GREATER_EQUAL = auto()
    LESS = auto(); LESS_EQUAL = auto()

    IDENTIFIER = auto(); STRING = auto(); NUMBER = auto()

    AND = auto(); OR = auto(); NOT = auto()
    CLASS = auto()
    IF = auto(); ELSE = auto()
    FUN = auto(); RETRUN = auto()
    TRUE = auto(); FALSE = auto()
    FOR = auto(); WHILE = auto()
    THIS = auto(); NIL = auto()
    PRINT = auto()
    VAR = auto()

    SUPER = auto()

    EOF = auto()


class Token():
    def __init__(self, token_type: TokenType, lexeme: str, literal, line: int):
        self.token_type = token_type
        self.lexeme = lexeme
        self.literal = literal
        self.line = line

    def __str__(self):
        return f'{self.token_type} {self.lexeme} {self.literal}'

had_error = False
had_runtimeerror = False

def report(line: int, where: str, message: str) -> None:
    print(f'[line {line}] Error {where}: {message}')
    global had_error
    had_error = True

def runtime_error(err: RuntimeException):
    print(f'{str(err)}\n[line {err.token.line}]')
    global had_runtimeerror
    had_runtimeerror = True

def error(token_or_line: Token|int, message: str) -> None:
    if isinstance(token_or_line, int):
        report(token_or_line, "", message)
    else:
        assert isinstance(token_or_line, Token)
        if token_or_line.token_type == TokenType.EOF:
            report(token_or_line.line, " at end", message)
        else:
            report(token_or_line.line, f" at '{token_or_line.lexeme}'", message)

class Scanner():
    keywords: Dict[str, TokenType] = {
        "and":      TokenType.AND,
        "class":    TokenType.CLASS,
        "else":     TokenType.ELSE,
        "false":    TokenType.FALSE,
        "for":      TokenType.FOR,
        "fun":      TokenType.FUN,
        "if":       TokenType.IF,
        "nil":      TokenType.NIL,
        "or":       TokenType.OR,
        "print":    TokenType.PRINT,
        "return":   TokenType.RETRUN,
        "super":    TokenType.SUPER,
        "this":     TokenType.THIS,
        "true":     TokenType.TRUE,
        "var":      TokenType.VAR,
        "while":    TokenType.WHILE,
    }

    def __init__(self, source: str):
        self.source = source
        self.tokens:List[Token] = []
        self.start: int = 0
        self.current: int = 0
        self.line: int = 1

    def is_at_end(self) -> bool:
        return self.current >= len(self.source)

    def advance(self) -> str:
        current_char: str = self.source[self.current]
        self.current += 1
        return current_char

    def next_match(self, expected: str) -> bool:
        if (self.is_at_end()):
            return False
        if (self.source[self.current] != expected):
            return False
        self.current += 1
        return True

    def peek(self) -> str:
        if self.is_at_end():
            return '\0'
        return self.source[self.current]

    def add_token(self, token_type: TokenType, literal = None):
        text: str = self.source[self.start:self.current]
        self.tokens.append(Token(token_type, text, literal, self.line))

    def scan_add_string(self):
        while self.peek() != '"' and not self.is_at_end():
            if self.peek() == '\n':
                self.line += 1
            self.advance()

        if self.is_at_end():
            error(self.line, "Unterminated string.")

        self.advance()
        self.add_token(TokenType.STRING, self.source[self.start + 1:self.current - 1])

    def peek_next(self) -> str:
        if self.current + 1 >= len(self.source):
            return '\0'
        return self.source[self.current + 1]

    def scan_add_number(self):
        while self.peek().isdigit():
            self.advance()
        if self.peek() == '.' and self.peek_next().isdigit():
            self.advance()
            while self.peek().isdigit():
                self.advance()

        self.add_token(TokenType.NUMBER, float(self.source[self.start:self.current]))

    def scan_add_identifier(self):
        while self.peek().isalnum():
            self.advance()
        token_type: TokenType|None = Scanner.keywords.get(self.source[self.start:self.current])
        if not token_type:
            token_type = TokenType.IDENTIFIER
        self.add_token(token_type)

    def scan_token(self):
        c: str = self.advance()
        match c:
            case '(': self.add_token(TokenType.LEFT_PAREN)
            case ')': self.add_token(TokenType.RIGHT_PAREN)
            case '{': self.add_token(TokenType.LEFT_BRACE)
            case '}': self.add_token(TokenType.RIGHT_BRACE)
            case ',': self.add_token(TokenType.COMMA)
            case '.': self.add_token(TokenType.DOT)
            case '-': self.add_token(TokenType.MINUS)
            case '+': self.add_token(TokenType.PLUS)
            case '*': self.add_token(TokenType.STAR)
            case ';': self.add_token(TokenType.SEMICOLON)

            case '"': self.scan_add_string()

            case '!':
                if self.next_match('='):
                    self.add_token(TokenType.BANG_EQUAL)
                else:
                    self.add_token(TokenType.BANG)
            case '=':
                if self.next_match('='):
                    self.add_token(TokenType.EQUAL_EQUAL)
                else:
                    self.add_token(TokenType.EQUAL)
            case '<':
                if self.next_match('='):
                    self.add_token(TokenType.LESS_EQUAL)
                else:
                    self.add_token(TokenType.LESS)
            case '>':
                if self.next_match('='):
                    self.add_token(TokenType.GREATER_EQUAL)
                else:
                    self.add_token(TokenType.GREATER)
            case '/':
                if (self.next_match('/')):
                    while (self.peek() != '\n' and not self.is_at_end()):
                        self.advance()
                else:
                    self.add_token(TokenType.SLASH)
            case ' ' | '\r' | '\t':
                pass
            case '\n':
                self.line += 1

            case _:
                if c.isdigit():
                    self.scan_add_number()
                elif c.isalpha():
                    self.scan_add_identifier()
                else:
                    error(self.line, "Unexpected character: {c}")

    def scan(self) -> List[Token]:
        while (not self.is_at_end()):
            self.start = self.current
            self.scan_token()

        self.tokens.append(Token(TokenType.EOF, "", None, self.line))
        return self.tokens

class AstPrinter(Expr.Visitor):
    def print(self, expression: Expr):
        return expression.accept(self)

    def parenthesize(self, name: str, expressions: List[Expr]):
        str_list = [expr.accept(self) for expr in expressions]
        return f'({name} {" ".join(str_list)})'

    def visit_literal(self, expression: Expr.Literal):
        if expression.value == None:
            return "nil"
        return str(expression.value)
    
    def visit_grouping(self, expression: Expr.Grouping):
        return self.parenthesize("group", [expression.expression])

    def visit_binary(self, expression: Expr.Binary):
        return self.parenthesize(expression.operator.lexeme,
                                 [expression.left, expression.right])

    def visit_unary(self, expression: Expr.Unary):
        return self.parenthesize(expression.operator.lexeme, [expression.right])

class Parser():
    class ParserError(RuntimeError):
        pass

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.current = 0

    def peek(self) -> Token:
        return self.tokens[self.current]

    def previous(self) -> Token:
        return self.tokens[self.current - 1]

    def is_at_end(self) -> bool:
        return self.peek().token_type == TokenType.EOF

    def advance(self) -> Token:
        if not self.is_at_end():
            self.current += 1
        return self.previous()

    def check(self, type: TokenType) -> bool:
        if self.is_at_end():
            return False
        return self.peek().token_type == type

    def match(self, *types) -> bool:
        for type in types:
            if self.check(type):
                self.advance()
                return True
        return False

    def error(self, token: Token, message: str) -> ParserError:
        error(token, message)
        return Parser.ParserError()

    def consume(self, type: TokenType, message: str):
        if self.check(type):
            return self.advance()
        raise self.error(self.peek(), message)

    def expression(self) -> Expr:
        return self.equality()

    def equality(self) -> Expr:
        expr: Expr = self.comparison()

        while self.match(TokenType.BANG, TokenType.BANG_EQUAL):
            operator: Token = self.previous()
            right: Expr = self.comparison()
            expr = Expr.Binary(expr, operator, right)

        return expr

    def comparison(self) -> Expr:
        expr: Expr = self.term()
        while self.match(TokenType.GREATER, TokenType.GREATER_EQUAL, TokenType.LESS, TokenType.LESS_EQUAL):
            operator: Token = self.previous()
            right: Expr = self.term()
            expr = Expr.Binary(expr, operator, right)

        return expr
    
    def term(self) -> Expr:
        expr: Expr = self.factor()
        while self.match(TokenType.PLUS, TokenType.MINUS):
            operator: Token = self.previous()
            right: Expr = self.factor()
            expr = Expr.Binary(expr, operator, right)

        return expr

    def factor(self) -> Expr:
        expr: Expr = self.unary()
        while self.match(TokenType.SLASH, TokenType.STAR):
            operator: Token = self.previous()
            right: Expr = self.unary()
            expr = Expr.Binary(expr, operator, right)

        return expr

    def unary(self) -> Expr:
        if self.match(TokenType.MINUS, TokenType.BANG):
            operator: Token = self.previous()
            right: Expr = self.unary()
            return Expr.Unary(operator, right)
        
        return self.primary()

    def primary(self) -> Expr:
        if self.match(TokenType.TRUE):
            return Expr.Literal(True)
        if self.match(TokenType.FALSE):
            return Expr.Literal(False)
        if self.match(TokenType.NIL):
            return Expr.Literal(None)

        if self.match(TokenType.NUMBER, TokenType.STRING):
            return Expr.Literal(self.previous().literal)

        if self.match(TokenType.LEFT_PAREN):
            expr: Expr = self.expression()
            self.consume(TokenType.RIGHT_PAREN, "Expect ')' after expression.")
            return Expr.Grouping(expr)

        raise self.error(self.peek(), "Expect expression.")

    def synchronize(self) -> None:
        self.advance()

        while not self.is_at_end():
            if self.previous().token_type == TokenType.SEMICOLON:
                return

            match self.peek().token_type:
                case TokenType.CLASS: return
                case TokenType.FUN: return
                case TokenType.VAR: return
                case TokenType.FOR: return
                case TokenType.IF: return
                case TokenType.WHILE: return
                case TokenType.PRINT: return
                case TokenType.RETRUN: return
            
            self.advance()

    def print_statement(self):
        value: Expr.Expr = self.expression()
        self.consume(TokenType.SEMICOLON, "Expect ';' after value.")
        return Stmt.Print(value)

    def expression_statement(self):
        expr: Expr.Expr = self.expression()
        self.consume(TokenType.SEMICOLON, "Expect ';' after expression.")
        return Stmt.Expression(expr)

    def statement(self):
        if self.match(TokenType.PRINT):
            return self.print_statement()
        else:
            return self.expression_statement()

    def parse(self) -> List[Stmt]:
        statements = []
        while not self.is_at_end():
            statements.append(self.statement())

        return statements
        
class RuntimeException(RuntimeError):
    def __init__(self, token: Token, message: str):
        super().__init__(message)
        self.token = token
        
class Interpreter(Expr.Visitor, Stmt.Visitor):
    def evaluate(self, expression: Expr):
        return expression.accept(self)

    def execute(self, statement: Stmt.Stmt):
        statement.accept(self)

    def interprte(self, statements: List[Stmt.Stmt]):
        try:
            for statement in statements:
                self.execute(statement)
        except RuntimeException as error:
            runtime_error(error)

    def is_truth(self, obj):
        if obj == None:
            return False
        if isinstance(obj, bool):
            return obj
        return True

    def is_equal(self, a, b):
        if a == None and b == None:
            return True
        if a == None:
            return False
        return a == b

    def check_number_operand(operator: Token, operand):
        if isinstance(operand, float):
            return
        raise RuntimeException(operator, "Operand must be a number.")

    def check_number_operands(self, operator: Token, left, right):
        if isinstance(left, float) and isinstance(right, float):
            return
        raise RuntimeException(operator, "Operands must be numbers.")

    def stringify(self, obj):
        if obj == None:
            return "nil"
        if isinstance(obj, float):
            if str(obj).endswith('.0'):
                return str(obj)[:-2]
        return str(obj)
    
    def visit_literal(self, expression: Expr.Literal):
        return expression.value

    def visit_grouping(self, expression: Expr.Grouping):
        return self.evaluate(expression.expression)

    def visit_unary(self, expression: Expr.Unary):
        right = self.evaluate(expression.right)

        match expression.operator.token_type:
            case TokenType.MINUS:
                self.check_number_operand(expression.operator, right)
                return -1.0 * float(right)
            case TokenType.BANG:
                return not self.is_truth(right)

        return None

    def visit_binary(self, expression: Expr.Binary):
        left = self.evaluate(expression.left)
        right = self.evaluate(expression.right)

        match expression.operator.token_type:
            case TokenType.MINUS:
                self.check_number_operands(expression.operator, left, right)
                return float(left) - float(right)
            case TokenType.SLASH:
                self.check_number_operands(expression.operator, left, right)
                return float(left) / float(right)
            case TokenType.STAR:
                self.check_number_operands(expression.operator, left, right)
                return float(left) * float(right)
            case TokenType.PLUS:
                if isinstance(left, float) and isinstance(right, float):
                    return float(left) + float(right)
                if isinstance(left, str) and isinstance(right, str):
                    return str(left) + str(right)
            case TokenType.GREATER:
                self.check_number_operands(expression.operator, left, right)
                return float(left) > float(right)
            case TokenType.GREATER_EQUAL:
                self.check_number_operands(expression.operator, left, right)
                return float(left) >= float(right)
            case TokenType.LESS:
                self.check_number_operands(expression.operator, left, right)
                return float(left) < float(right)
            case TokenType.LESS_EQUAL:
                self.check_number_operands(expression.operator, left, right)
                return float(left) <= float(right)
            case TokenType.BANG_EQUAL:
                return not self.is_equal(left, right)
            case TokenType.EQUAL_EQUAL:
                return self.is_equal(left, right)

        assert False, f"{str(expression.operator)}"

        return None

    def visit_expression_stmt(self, stmt: Stmt.Expression):
        self.evaluate(stmt.expression)
        return None

    def visit_print_stmt(self, stmt: Stmt.Print):
        value = self.evaluate(stmt.expression)
        print(self.stringify(value))
        return None

def get_args():
    parser = argparse.ArgumentParser()
    run_option = parser.add_mutually_exclusive_group(required = True)
    run_option.add_argument("-f", "--file", help = "Run source code")
    run_option.add_argument("-p", "--prompt", action="store_true", help = "Run in repl mode")

    args = parser.parse_args()
    return args

def run(source: str) -> None:
    scanner: Scanner = Scanner(source)
    tokens: List[Token] = scanner.scan()

    parser: Parser = Parser(tokens)
    statements: List[Stmt.Stmt] = parser.parse()

    interpreter: Interpreter = Interpreter()
    interpreter.interprte(statements)

    if had_error:
        exit(65)
    if had_runtimeerror:
        exit(70)

def run_prompt() -> None:
    print("Run REPL...")

    while (True):
        global had_error
        had_error = False
        global had_runtimeerror
        had_runtimeerror = False
        line: str = input("> ")
        if not line:
            break
        run(line)

def run_file(src: str | Path):
    print(f"Run source code with file: {src}")

def main() -> None:
    run_args = get_args()
    if run_args.prompt:
        run_prompt()
    else:
        run_file(run_args.file)

def debug() -> None:
    expression: Expr.Expr = Expr.Binary(
        Expr.Unary(Token(TokenType.MINUS, '-', None, 1), Expr.Literal(123)),
        Token(TokenType.STAR, '*', None, 1),
        Expr.Grouping(Expr.Literal(45.67))
    )

    print(AstPrinter().print(expression))

if __name__ == '__main__':
    debug()
    # main()
