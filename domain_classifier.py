import re


_CODING = re.compile(
    r"("
    r"|write.*code|write.*program|write.*script|write.*function"
    r"|implement.*code|implement.*program|implement.*script|implement.*function"
    r"|what does this code do|explain this code|fix this code"
    r")",
    re.IGNORECASE,
)

_MATH = re.compile(
    r"("
    r"\$[^$]+\$"
    r"(\d+\s*[\+\-\*/ร—รท=]\s*\d+)"
    r"\\frac|\\sqrt|\\boxed"
    r"\b(find|solve|how many|value of|compute|calculate|amount)\b.{0,80}"
    r"(\d|\$|equation|value|probability|sum|different|product|dividend|divisor|integer)"
    r"(\d|\$|division|addition|subtraction|multiplication|plus|minus|times)"
    r"|\b(AMC|AIME|probability|chance|square|triangle|quadrilateral)\b"
    r")",
    re.IGNORECASE,
)

_PLANNING = re.compile(
    r"\b("
    r"plan|"
    r"transporting|"
    r"shipping|"
    r"move|carry|distribute|logistics"
    r"sell|distributor|hoist|seller|buyer|customer"
    r")\b",
    re.IGNORECASE,
)

_FUTURE = re.compile(
    r"("
    r"predict.{0,40}(future|will happen|would happen|will be|outcome)"
    r"agent that can predict|going to happen"
    r"\bforecast\b|\bexpect\b|\bguess\b|\bprediction\b|\bprediction\b"
    r"\\boxed\{YOUR_PREDICTION\}"
    r"|\b\d{4}-\d{2}-\d{2}\b.{0,120}\b(prediction|forecast|estimate|anticipate|expect|guess)\b"
    r")",
    re.IGNORECASE,
)


def classify_domain(input_text: str) -> str:
    if not input_text:
        return "common_sense"
    t = input_text
    if _MATH.search(t):
        return "math"
    if _CODING.search(t):
        return "coding"
    if _PLANNING.search(t):
        return "planning"
    if _FUTURE.search(t):
        return "future_prediction"
    return "common_sense"