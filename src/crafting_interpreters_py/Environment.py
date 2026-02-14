
class Environment():
    def __init__(self):
        self.values = {}


    def define(self, name: str, value: object):
        self.values[name] = value

    def get(self, name_token):
        if name_token.lexeme in self.values:
            return values[name_token.lexeme]
        else:
            pass
