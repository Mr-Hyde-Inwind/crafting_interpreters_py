from enum import Enum, auto

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

