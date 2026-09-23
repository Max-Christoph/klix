# Minimal example
from klix import DecisionEngine, Choice

engine = DecisionEngine()
engine.add_head(Choice(name="route", options={"yes": ["sure", "absolutely"], "no": ["rather not", "probably not"]}))
engine.compile()
print(engine.decide("clear case"))