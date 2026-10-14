"""Tokenisation shared by BM25 and the hashing embedder."""
import re

STOPWORDS = frozenset(
    """a an and are as at be but by for from has have how i in is it its of on or
    that the their there this to was were what when where which who why will with
    you your can do does did not no if than then so such into about over under""".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def stem(token: str) -> str:
    """Very light suffix stripping (plurals, -ed, -ing) so 'deployments' matches 'deployment'."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return [stem(t) for t in tokens]
