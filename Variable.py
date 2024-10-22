class Variable :

    var_integer = 10

    var_float = 1.5

    @classmethod
    def get(cls):
        return cls.var_integer, cls.var_float