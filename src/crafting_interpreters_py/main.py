from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Dict

from .token import Token, TokenType
from . import Expr
from . import Stmt
from .environment import Environment
from .error import RuntimeException, Return
from .callable import InterpreterCallable, InterpreterFunction



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
        """Scan the source code and return Token list

        Iter the string of source code, resolving to Token object.
        Stop scanning when touch the end of source code and append EOF Token to the list
        
        Returns:
            List[Token]: list of tokens scanned from raw string.
        """
        while (not self.is_at_end()):
            self.start = self.current
            self.scan_token()

        self.tokens.append(Token(TokenType.EOF, "", None, self.line))
        return self.tokens

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

    def check(self, type: TokenType) -> bool|None:
        if self.is_at_end():
            return None
        return self.peek().token_type == type

    def match(self, *types) -> bool:
        for type in types:
            if self.check(type):
                self.advance()
                return True
        return False

    def error(self, token: Token, message: str) -> ParserError:
        error(token, message)
        # debug
        #breakpoint()
        return Parser.ParserError()

    def consume(self, type: TokenType, message: str):
        if self.check(type):
            return self.advance()
        raise self.error(self.peek(), message)

    def expression(self) -> Expr.Expr:
        return self.assignment()

    def assignment(self) -> Expr.Expr:
        expr: Expr.Expr = self.or_expression()
        if self.match(TokenType.EQUAL):
            equals: Token = self.previous()
            value: Expr.Expr = self.assignment()

            if isinstance(expr, Expr.Variable):
                name: Token = expr.name
                return Expr.Assignment(name, value)

            self.error(equals, "Invalid assignment target.")

        return expr

    def or_expression(self) -> Expr.Expr:
        expr: Expr.Expr = self.and_expression()

        while (self.match(TokenType.OR)):
            operator: Token = self.previous()
            right: Expr.Expr = self.and_expression()
            expr = Expr.Logical(expr, operator, right)

        return expr

    def and_expression(self) -> Expr.Expr:
        expr: Expr.Expr = self.equality()

        while (self.match(TokenType.AND)):
            operator: Token = self.previous()
            right:Expr.Expr = self.equality()
            expr = Expr.Logical(expr, operator, right)

        return expr

    def equality(self) -> Expr.Expr:
        expr: Expr.Expr = self.comparison()

        while self.match(TokenType.BANG, TokenType.BANG_EQUAL):
            operator: Token = self.previous()
            right: Expr.Expr = self.comparison()
            expr = Expr.Binary(expr, operator, right)

        return expr

    def comparison(self) -> Expr.Expr:
        expr: Expr.Expr = self.term()
        while self.match(TokenType.GREATER, TokenType.GREATER_EQUAL, TokenType.LESS, TokenType.LESS_EQUAL):
            operator: Token = self.previous()
            right: Expr.Expr = self.term()
            expr = Expr.Binary(expr, operator, right)

        return expr
    
    def term(self) -> Expr.Expr:
        expr: Expr.Expr = self.factor()
        while self.match(TokenType.PLUS, TokenType.MINUS):
            operator: Token = self.previous()
            right: Expr.Expr = self.factor()
            expr = Expr.Binary(expr, operator, right)

        return expr

    def factor(self) -> Expr.Expr:
        expr: Expr.Expr = self.unary()
        while self.match(TokenType.SLASH, TokenType.STAR):
            operator: Token = self.previous()
            right: Expr.Expr = self.unary()
            expr = Expr.Binary(expr, operator, right)

        return expr

    def unary(self) -> Expr.Expr:
        if self.match(TokenType.MINUS, TokenType.BANG):
            operator: Token = self.previous()
            right: Expr.Expr = self.unary()
            return Expr.Unary(operator, right)
        
        return self.call()

    def call(self):
        expr: Expr.Expr = self.primary()

        while True:
            if self.match(TokenType.LEFT_PAREN):
                expr = self.finish_call(expr)
            else:
                break

        return expr

    def finish_call(self, callee: Expr.Expr) -> Expr.Expr:
        arguments: List[Expr.Expr] = []
        if not self.check(TokenType.RIGHT_PAREN):
            while True:
                if len(arguments) >= 127:
                    self.error(self.peek(), "Can't have more than 127 arguments.")
                arguments.append(self.expression())
                if not self.match(TokenType.COMMA):
                    break

        paren: Token = self.consume(TokenType.RIGHT_PAREN, "Expect ')' after arguments.")
        return Expr.Call(callee, paren, arguments)


    def primary(self) -> Expr.Expr:
        if self.match(TokenType.TRUE):
            return Expr.Literal(True)
        if self.match(TokenType.FALSE):
            return Expr.Literal(False)
        if self.match(TokenType.NIL):
            return Expr.Literal(None)

        if self.match(TokenType.NUMBER, TokenType.STRING):
            return Expr.Literal(self.previous().literal)

        if self.match(TokenType.IDENTIFIER):
            return Expr.Variable(self.previous())

        if self.match(TokenType.LEFT_PAREN):
            expr: Expr.Expr = self.expression()
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

    def block(self) -> List[Stmt.Stmt|None]:
        statements: List[Stmt.Stmt|None] = []
        while not self.check(TokenType.RIGHT_BRACE):
            statements.append(self.declaration())
        self.consume(TokenType.RIGHT_BRACE, "Expect '}' after block.")
        return statements

    def if_statement(self):
        self.consume(TokenType.LEFT_PAREN, f"expect '(' after if.")
        condition: Expr.Expr = self.expression()
        self.consume(TokenType.RIGHT_PAREN, f"unclosed parenthesize.")

        then_branch: Stmt.Stmt|None = self.statement()
        else_branch: Stmt.Stmt|None = self.statement() if self.match(TokenType.ELSE) else None

        return Stmt.If(condition, then_branch, else_branch)

    def while_statement(self):
        self.consume(TokenType.LEFT_PAREN, f"expect '(' after while.")
        condition: Expr.Expr = self.expression()
        self.consume(TokenType.RIGHT_PAREN, f"unclosed parenthesize.")
        body: Stmt.Stmt = self.statement()

        return Stmt.While(condition, body)

    def for_statement(self):
        self.consume(TokenType.LEFT_PAREN, f"Expect '(' after for.")

        initializer: Stmt.Stmt|None = None
        if self.match(TokenType.SEMICOLON):
            initializer = None
        elif self.match(TokenType.VAR):
            initializer = self.var_declaration()
        else:
            initializer = self.expression_statement()

        condition: Expr.Expr|None = None
        if not self.check(TokenType.SEMICOLON):
            condition = self.expression()
        self.consume(TokenType.SEMICOLON, f"Expect ';' after loop condition.")

        increment: Expr.Expr|None = None
        if not self.check(TokenType.RIGHT_PAREN):
            increment = self.expression()
        self.consume(TokenType.RIGHT_PAREN, f"Expect ')' after for clauses.")

        body: Stmt.Stmt = self.statement()

        if increment:
            body = Stmt.Block([body, Stmt.Expression(increment)])

        if condition == None:
            condition = Expr.Literal(True)
        body = Stmt.While(condition, body)

        if initializer:
            body = Stmt.Block([initializer, body])

        return body

    def return_statement(self):
        keyword: Token = self.previous()
        value: object = None
        if not self.check(TokenType.SEMICOLON):
            value = self.expression()

        self.consume(TokenType.SEMICOLON, "Expect ';' after return value.")
        return Stmt.Return(keyword, value)

    def statement(self):
        if self.match(TokenType.PRINT):
            return self.print_statement()
        elif self.match(TokenType.LEFT_BRACE):
            return Stmt.Block(self.block())
        elif self.match(TokenType.IF):
            return self.if_statement()
        elif self.match(TokenType.WHILE):
            return self.while_statement()
        elif self.match(TokenType.FOR):
            return self.for_statement()
        elif self.match(TokenType.RETRUN):
            return self.return_statement()
        else:
            return self.expression_statement()

    def var_declaration(self) -> Stmt.Stmt:
        name: Token = self.consume(TokenType.IDENTIFIER, "Expect variable name.")

        initializer: Expr.Expr|None = None
        if self.match(TokenType.EQUAL):
            initializer = self.expression()

        self.consume(TokenType.SEMICOLON, "Expect ';' after variable declaration.")
        return Stmt.Var(name, initializer)

    def function(self, kind: str):
        name: Token = self.consume(TokenType.IDENTIFIER, f"Expect {kind} name.")
        self.consume(TokenType.LEFT_PAREN, f"Expect '(' after {kind} name.")
        params: List[Token] = []
        if not self.check(TokenType.RIGHT_PAREN):
            while True:
                if len(params) >= 127:
                    self.error(self.peek(), "Can't have more than 127 parameters.")

                params.append(self.consume(TokenType.IDENTIFIER, "Expect parameter name."))

                if not self.match(TokenType.COMMA):
                    break
        self.consume(TokenType.RIGHT_PAREN, f"Expect ')' after parameters.")
        self.consume(TokenType.LEFT_BRACE, f"Expect '{{' before {kind} body.")

        body: List[Stmt.Stmt|None] = self.block()
        return Stmt.Function(name, params, body)

    def declaration(self):
        try:
            if self.match(TokenType.FUN):
                return self.function("function")
            if self.match(TokenType.VAR):
                return self.var_declaration()
            else:
                return self.statement()
        except Parser.ParserError as err:
            self.synchronize()
            return None

    def parse(self) -> List[Stmt.Stmt]:
        statements = []
        while not self.is_at_end():
            statements.append(self.declaration())

        return statements
        
class Interpreter(Expr.Visitor, Stmt.Visitor):
    def __init__(self):
        self.globals = Environment()
        self.environment = self.globals

        # Native Func
        clock_instance = type("ClockCallable", (InterpreterCallable,), {
            "arity": lambda self: 0,
            "call": lambda self, interp, args: __import__('time').time(),
            "__str__": lambda self: "<native fn>."
        })()
        self.globals.define("clock", clock_instance)
        
    
    def evaluate(self, expression: Expr.Expr):
        return expression.accept(self)

    def execute(self, statement: Stmt.Stmt):
        statement.accept(self)

    def interprete(self, statements: List[Stmt.Stmt]):
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
        return bool(obj)

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

    def visit_call(self, expr: Expr.Call) -> object:
        callee: object = self.evaluate(expr.callee)

        arguments: List[object] = []
        for argument in expr.arguments:
            arguments.append(self.evaluate(argument))

        if not isinstance(callee, InterpreterCallable):
            raise RuntimeException(expr.paren, "Can only call functions and classes.")

        #function: InterpreterCallable = InterpreterCallable(callee)
        function = callee
        if len(arguments) != function.arity():
            raise RuntimeException(expr.paren,
                    f"Expected {function.arity()} arguments but got {len(arguments)}.")
        return function.call(self, arguments)

    def visit_variable(self, expression: Expr.Variable):
        return self.environment.get(expression.name)

    def visit_assignment(self, expression: Expr.Assignment):
        value: object = self.evaluate(expression.value)
        self.environment.assign(expression.name, value)
        return value

    def visit_logical(self, expression: Expr.Logical):
        left: object = self.evaluate(expression.left)

        if expression.operator.token_type == TokenType.OR:
            if self.is_truth(left):
                return left
        else:
            if not self.is_truth(left):
                return left

        return self.evaluate(expression.right)

    def visit_expression_stmt(self, stmt: Stmt.Expression):
        self.evaluate(stmt.expression)
        return None

    def visit_print_stmt(self, stmt: Stmt.Print):
        value = self.evaluate(stmt.expression)
        print(self.stringify(value))
        return None

    def visit_var_stmt(self, stmt: Stmt.Var):
        value = None
        if stmt.initializer != None:
            value = self.evaluate(stmt.initializer)

        self.environment.define(stmt.name.lexeme, value)
        return None

    def visit_function(self, stmt: Stmt.Function):
        function = InterpreterFunction(stmt)
        self.environment.define(stmt.name.lexeme, function)
        return None

    def execute_block(self, statements: List[Stmt.Stmt], environment: Environment):
        previous: Environment = self.environment
        try:
            self.environment = environment
            for statement in statements:
                self.execute(statement)
        finally:
            self.environment = previous

    def visit_block(self, stmt: Stmt.Block):
        self.execute_block(stmt.statements, Environment(self.environment))
        return None

    def visit_if(self, stmt: Stmt.If):
        if self.is_truth(self.evaluate(stmt.condition)):
            self.execute(stmt.then_branch)
        elif stmt.else_branch:
            self.execute(stmt.else_branch)

        return None

    def visit_while(self, stmt: Stmt.While):
        while self.is_truth(self.evaluate(stmt.condition)):
            self.execute(stmt.body)

        return None

    def visit_return(self, stmt: Stmt.Return):
        value = None
        if stmt.value != None:
            value = self.evaluate(stmt.value)

        raise Return(value)


def get_args():
    parser = argparse.ArgumentParser()
    run_option = parser.add_mutually_exclusive_group(required = True)
    run_option.add_argument("-f", "--file", help = "Run source code")
    run_option.add_argument("-p", "--prompt", action="store_true", help = "Run in repl mode")

    args = parser.parse_args()
    return args

def run(interpreter: Interpreter, source: str) -> None:
    scanner: Scanner = Scanner(source)
    tokens: List[Token] = scanner.scan()

    parser: Parser = Parser(tokens)
    statements: List[Stmt.Stmt] = parser.parse()

    interpreter.interprete(statements)

    if had_error:
        exit(65)
    if had_runtimeerror:
        exit(70)

def run_prompt() -> None:
    print("Run REPL...")
    
    interpreter: Interpreter = Interpreter()
    while (True):
        global had_error
        had_error = False
        global had_runtimeerror
        had_runtimeerror = False
        line: str = input("> ")
        if not line:
            break
        run(interpreter, line)

def run_file(src: str | Path):
    with open(src, 'r') as f:
        source = f.read()
    interpreter: Interpreter = Interpreter()
    global had_error
    global had_runtimeerror
    had_error = False
    had_runtimeerror = False

    run(interpreter, source)

    #print(f"Run source code with file: {src}")

def main() -> None:
    run_args = get_args()
    if run_args.prompt:
        run_prompt()
    else:
        run_file(run_args.file)

