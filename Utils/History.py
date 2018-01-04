class History:
    def __init__(self,tokens):
        self.tokens=tokens
    def toString(self):
        tmp = [token.toString() for token in self.tokens[1:]]
        return "".join(tmp)+'\n'
