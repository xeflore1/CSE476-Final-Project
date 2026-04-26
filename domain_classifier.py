import re


_CODING = re.compile(
    r"\b(?:write|implement|complete)\b.{0,40}"
    r"\b(?:code|program|script|function|method|class)\b"
    r"|\bdef\s+\w+\s*\("
    r"|\bimport\s+\w+"
    r"|```python|```\s*\n"
    r"|\bself_contained\b|task_func\(",
    re.IGNORECASE,
)

_MATH = re.compile(
    r"\$[^$]+\$"
    r"|\\(?:frac|sqrt|boxed|sum|prod|int|cdot|times|leq|geq|neq)"
    r"|\b(?:AMC|AIME|IMO)\b"
    r"|\b\d+\s*[\+\-\*/×÷=]\s*\d+"
    r"|\b(?:find|solve|compute|evaluate|determine|calculate|what is the largest|what is the smallest|how many)\b.{0,80}"
    r"\b(?:value|sum|product|probability|equation|integer|integers|root|area|volume|angle|length|ratio|percent|percentage|fraction|digits|remainder|divisible|prime|matches|number of)\b"
    r"|\b(?:ratio|percentage|remainder|divisible|prime)\b.{0,60}\b(?:is|was|will|equal)\b"
    r"|\bwin(?:\s+)?ratio\b"
    r"|\bmod(?:ulo)?\s+\d+\b",
    re.IGNORECASE,
)

_PLANNING = re.compile(
    r"\b(?:plan|sequence of (?:actions|steps)|action sequence)\b"
    r"|\b(?:transport|ship|move|carry|distribute|deliver)\b"
    r"|\b(?:pick up|put down|stack|unstack|on top of|on the table|hand is empty)\b"
    r"|\[STATEMENT\]|\[PLAN\]|\[PLAN END\]",
    re.IGNORECASE,
)

_FUTURE = re.compile(
    r"\bagent that can predict\b"
    r"|\bpredict\b.{0,40}\b(?:future|will happen|outcome|will be|is going to)\b"
    r"|\\boxed\{YOUR_PREDICTION\}"
    r"|\bevent to be predicted\b"
    r"|\b\d{4}-\d{2}-\d{2}\b.{0,120}\b(?:predict|forecast|estimate|anticipate)\b",
    re.IGNORECASE,
)


def classify_domain(input_text: str) -> str:
    if not input_text:
        return "common_sense"
    t = input_text
    if _FUTURE.search(t):
        return "future_prediction"
    if _PLANNING.search(t):
        return "planning"
    if _MATH.search(t):
        return "math"
    if _CODING.search(t):
        return "coding"
    return "common_sense"