# Minimalbeispiel
from klix import DecisionEngine, Choice

engine = DecisionEngine()
engine.add_head(Choice(name="route", options={"ja": ["klar", "auf jeden"], "nein": ["lieber nicht"]}))
engine.compile()
print(engine.decide("klarer Fall"))