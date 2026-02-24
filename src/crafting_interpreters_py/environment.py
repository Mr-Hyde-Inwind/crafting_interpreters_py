from .token import Token
from .error import RuntimeException
class Environment():
    def __init__(self, enclosing = None):
        self.values = dict()
        self.enclosing = enclosing

    def define(self, name: str, value: object):
        self.values[name] = value

    def assign(self, name: Token, value: object):
        if name.lexeme in self.values:
            self.values[name.lexeme] = value
            return

        if self.enclosing:
            self.enclosing.assign(name, value)
            return

        raise RuntimeException(name, f'Undefined variable "{name.lexeme}".')

    def get(self, name_token: Token):
        if name_token.lexeme in self.values:
            return self.values[name_token.lexeme]
        elif self.enclosing:
            return self.enclosing.get(name_token)
        else:
            raise RuntimeException(name_token, f'Undefined variable {name_token.lexeme}.')

