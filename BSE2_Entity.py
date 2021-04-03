class Entity:

    def __init__(self, id, init_balance, init_reputation, traders):
        self.lei = id  # LEI = legal entity identifier
        self.balance = init_balance
        self.reputation = init_reputation
        self.traders = traders

    def __str__(self):
        s = '[%s $%d R=%d %s]' % (self.lei, self.balance, self.reputation, str(self.traders))
        return s