import pandas as pd


class DPTask:

    def __init__(self):
        raise NotImplementedError

    def execute(self):
        raise

    def evaluate(self, results, *, dev_name, test_name, **kwargs):
        raise NotImplementedError

    def plot(self, results, *, dev_name, test_name, **kwargs):
        raise NotImplementedError
