class Token():
    def __init__(self, idx, token, pos, head):
        self.idx = idx
        self.token = token
        self.pos = pos
        self.head = head

    def __str__(self) -> str:
        return self.token

    def toString(self):
        return str(self.idx) + '\t' + self.token + '\t_\t' + self.pos + '\t_\t_\t'+str(self.head) + '\t_\t_\t_\n'
