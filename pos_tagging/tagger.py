import nltk
from nltk.tag.hmm import HiddenMarkovModelTrainer, HiddenMarkovModelTagger
from nltk.tag.brill_trainer import BrillTaggerTrainer, BrillTagger

class Tagger(object):
    def __init__(self):
        self.trainer = None
        self.tagger = None

    @staticmethod
    def factory(tagger_name):
        if tagger_name == "hmm":
            return HMMTagger()
        elif tagger_name == "brill":
            return BrillTagger()

    def train(self, train_data):
        pass

    def test(self, test_data):
        pass

class HMMTagger(Tagger):
    def __init__(self):
        self.trainer = HiddenMarkovModelTrainer()

    def train(self, train_data):
        self.tagger = self.trainer.train_supervised(train_data)

    def test(self, test_data):
        self.tagger.test(test_data)

# TODO
# - complete Brill tagger with baseline
class BrillTagger(Tagger):
    def __init__(self):
        self.trainer = BrillTaggerTrainer()

    def train(self, train_data):
        self.tagger = self.trainer.train(train_data)

    def test(self, test_data):
        pass
