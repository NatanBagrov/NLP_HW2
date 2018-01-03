import os

from History import History
from Token import Token


class MyParser:

    def __init__(self, fileName):
        self.fileName = fileName
        f = open(self.fileName)
        self.lines = f.read()
        f.close()
        self.lines = self.lines.split('\n\n')
        root = Token(0, 'Root', '_', -1)
        self.histories = []
        for sentence in self.lines:
            tokens = [root]
            words = sentence.split('\n')
            for inputWord in words:
                word = inputWord.split('\t')
                token = Token(int(word[0]), word[1], word[3], int(word[6]))
                tokens.append(token)
            self.histories.append(History(tokens))

    def _getHeadTokens(self):
        tokens = []
        for history in self.histories:
            for token in history:
                p_token = history[token.head]
                if p_token.idx != 0:
                    tokens.append(p_token)
        return tokens

    def getTupleOfPosAndWordFromHeadTokens(self):
        heads = self._getHeadTokens()
        res = set()
        for head in heads:
            tpl = head.token, head.pos
            res.add(tpl)
        res = list(res)
        res.sort()
        return res

    def getHeadWordsFromHeadTokens(self):
        heads = self.getTupleOfPosAndWordFromHeadTokens()
        res = set()
        for tpl in heads:
            res.add(tpl[0])
        res = list(res)
        res.sort()
        return res

    def getHeadPosFromHeadToken(self):
        pos = self.getTupleOfPosAndWordFromHeadTokens()
        res = set()
        for tpl in pos:
            res.add(tpl[1])
        res = list(res)
        res.sort()
        return res

    def getModifierTokens(self):
        tokens = []
        for history in self.histories:
            for token in history:
                if token.idx == 0:
                    continue
                tokens.append(token)
        return tokens

    def getTupleOfPosAndWordFromModifierTokens(self):
        modifiers = self.getModifierTokens()
        res = set()
        for modifier in modifiers:
            tpl = modifier.token, modifier.pos
            res.add(tpl)
        res = list(res)
        res.sort()
        return res

    def getModifierWordsFromModifierTokens(self):
        modifiers = self.getTupleOfPosAndWordFromModifierTokens()
        res = set()
        for tpl in modifiers:
            res.add(tpl[0])
        res = list(res)
        res.sort()
        return res

    def getModifierPosFromModifierToken(self):
        pos = self.getTupleOfPosAndWordFromModifierTokens()
        res = set()
        for tpl in pos:
            res.add(tpl[1])
        res = list(res)
        res.sort()
        return res