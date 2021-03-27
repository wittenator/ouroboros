from itertools import chain, repeat
import functools

def Loop(steps, iterations=1):
    return chain.from_iterable(repeat(steps, iterations))

class Process(object):
    def __init__(self, process):
        self.process = process

    def compose(*functions):
        return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)

    def run(self, message=None):
        return self.compose(self.process)(message)
