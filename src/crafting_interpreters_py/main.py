import argparse
from pathlib import Path
from enum import Enum, auto
from typing import List, Dict

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
def report(line: int, where: str, message: str) -> None:
    print(f'[line {line}] Error {where}: {message}')
    global had_error
    had_error = True


def error(line: int, message: str) -> None:
    report(line, "", message)

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

    for token in tokens:
        print(token)


def run_prompt() -> None:
    print("Run REPL...")

    while (True):
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

if __name__ == '__main__':
    main()
