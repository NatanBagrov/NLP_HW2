from Utils.History import History
from Utils.Token import Token


class MyParser:

    def __init__(self, fileName):
        self.fileName = fileName
        f = open(self.fileName)
        self.lines = f.read()
        f.close()
        self.lines = self.lines.split('\n\n')
        root = Token(0, 'ROOTWORD', 'ROOTPOS', -1)
        self.histories = []
        for sentence in self.lines:
            tokens = [root]
            words = sentence.split('\n')
            for inputWord in words:
                word = inputWord.split('\t')
                p = -1
                if word[6] != '_':
                    p = int(word[6])
                token = Token(int(word[0]), word[1], word[3], p)
                tokens.append(token)

            self.histories.append(History(tokens))

    def getTupleOfHeadAndModifier(self):
        tuple_tokens = []
        for history in self.histories:
            for modifier_token in history.tokens[1:]:
                head_token = history.tokens[modifier_token.head]
                tuple_tokens.append((head_token, modifier_token))
        return tuple_tokens

    def _getHeadTokens(self):
        return [tpl[0] for tpl in self.getTupleOfHeadAndModifier()]

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
        return [tpl[1] for tpl in self.getTupleOfHeadAndModifier()]

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

    def getAllHistoriesWithParseTrees(self):
        return [(h, self._getParseTreeFromHistory(h)) for h in self.histories]

    def _getParseTreeFromHistory(self, history: History):
        res = []
        for token in history.tokens[1:]:
            res.append((token.head, token.idx))
        return res

    def getHeadAndModifierTokensAndDistance(self):
        return [(h, m, h.idx - m.idx) for h, m in (self.getTupleOfHeadAndModifier())]

    def getHeadAndModifierPosClassAndDistance(self):
        return sorted(set([(self.getPosClass(h.pos), self.getPosClass(m.pos), d) for (h, m, d) in
                           self.getHeadAndModifierTokensAndDistance()]))

    def getPosClass(self, pos: str):
        if pos.startswith('NN'):
            return '_NN'
        elif pos.startswith('VB'):
            return '_VB'
        elif pos.startswith('JJ'):
            return '_JJ'
        elif pos.startswith('RB'):
            return '_RB'
        elif pos.startswith('PRP'):
            return '_PRP'
        else:
            return '_' + pos

    def getTupleOfHeadAndModifierAndTokensInBetween(self):
        tuple_tokens = []
        for history in self.histories:
            for modifier_token in history.tokens[1:]:
                head_token = history.tokens[modifier_token.head]
                start, end = min(head_token.idx, modifier_token.idx) + 1, max(head_token.idx, modifier_token.idx)
                inbetween = [token for token in history.tokens[start:end]]
                tuple_tokens.append((head_token, modifier_token, inbetween))
        return tuple_tokens

    def getTupleOfHeadAndModifierAndNeighbors(self):
        tuple_tokens = []
        for history in self.histories:
            for modifier_token in history.tokens[1:]:
                head_token = history.tokens[modifier_token.head]
                head_nm1 = Token(-1, "____Before Root", "_", -1)
                head_n1 = modifier_n1 = Token(-1, "____STOP", "_", -1)
                if head_token.idx > 0:
                    head_nm1 = history.tokens[head_token.idx - 1]
                modifier_nm1 = history.tokens[modifier_token.idx - 1]
                size = len(history.tokens)
                if head_token.idx != size - 1:
                    head_n1 = history.tokens[head_token.idx + 1]
                if modifier_token.idx != size - 1:
                    modifier_n1 = history.tokens[modifier_token.idx + 1]
                tuple_tokens.append((head_nm1, head_token, head_n1, modifier_nm1, modifier_token, modifier_n1))
        return tuple_tokens
