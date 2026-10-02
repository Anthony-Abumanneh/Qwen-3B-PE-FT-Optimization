"""
Answer checker for math eval runs.
Handles MCQ (letter match) and free-form (symbolic/numeric equivalence via sympy).

Free-form answers in the dataset are lists of strings (one per [ANS] blank).
ALL blanks must match for the item to count correct.

Usage (drop-in compatible with the old notebook):
    from judger import Judger
    judger = Judger(strict_extract=False)
    correct = judger.auto_judge(pred=response_text, gold=gold_list, options=[[]]*len(gold_list))
"""
import re
from sympy import simplify, sympify, N, Rational
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor,
)

_TRANSFORMS = standard_transformations + (
    implicit_multiplication_application, convert_xor,
)

NUM_TOL = 1e-4  # numeric tolerance for decimal comparisons


def _extract_boxed(text):
    """Return the content of the LAST \\boxed{...} in text, handling nested braces."""
    idx = text.rfind(r"\boxed")
    if idx == -1:
        return None
    i = text.find("{", idx)
    if i == -1:
        return None
    depth = 0
    out = []
    for ch in text[i:]:
        if ch == "{":
            depth += 1
            if depth == 1:
                continue
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
    return "".join(out).strip()


def _clean_latex(s):
    """Strip common LaTeX wrappers so sympy can parse the expression."""
    if s is None:
        return ""
    s = s.strip()
    s = s.replace("$", "").replace(r"\left", "").replace(r"\right", "")
    s = s.replace(r"\!", "").replace(r"\,", "").replace(r"\;", "").replace(r"\ ", " ")
    s = s.replace(r"\%", "").replace("%", "")
    s = s.replace(r"\times", "*").replace(r"\cdot", "*").replace(r"\div", "/")
    s = s.replace("^{\\circ}", "").replace(r"\circ", "")
    # \frac{a}{b} -> (a)/(b)
    s = re.sub(r"\\d?frac\s*{([^{}]*)}\s*{([^{}]*)}", r"(\1)/(\2)", s)
    s = re.sub(r"\\d?frac\s*{([^{}]*)}\s*{([^{}]*)}", r"(\1)/(\2)", s)  # nested pass
    s = s.replace(r"\sqrt", "sqrt")
    s = re.sub(r"sqrt\s*{([^{}]*)}", r"sqrt(\1)", s)
    s = s.replace(r"\pi", "pi").replace(r"\infty", "oo")
    s = s.replace(r"\{", "").replace(r"\}", "")
    s = s.replace("{", "(").replace("}", ")")
    s = s.replace("\\", "")
    # trailing '=' or leading 'x=' style
    if "=" in s:
        s = s.split("=")[-1]
    return s.strip()


def _to_expr(s):
    cleaned = _clean_latex(s)
    if cleaned == "":
        return None
    try:
        return parse_expr(cleaned, transformations=_TRANSFORMS, evaluate=True)
    except Exception:
        try:
            return sympify(cleaned)
        except Exception:
            return None


def _equiv(pred_str, gold_str):
    """True if pred_str and gold_str are symbolically or numerically equal."""
    if pred_str is None:
        return False
    p_raw = pred_str.strip()
    g_raw = str(gold_str).strip()
    if p_raw == g_raw:
        return True

    pe = _to_expr(p_raw)
    ge = _to_expr(g_raw)
    if pe is None or ge is None:
        return False

    # symbolic equality
    try:
        if simplify(pe - ge) == 0:
            return True
    except Exception:
        pass

    # numeric equality
    try:
        pv = float(N(pe))
        gv = float(N(ge))
        if abs(pv - gv) <= NUM_TOL * max(1.0, abs(gv)):
            return True
    except Exception:
        pass
    return False


class Judger:
    def __init__(self, strict_extract=False):
        self.strict_extract = strict_extract

    def _split_pred_blanks(self, boxed):
        """A boxed answer with commas -> one entry per blank."""
        if boxed is None:
            return []
        # split top-level commas only
        parts, depth, cur = [], 0, ""
        for ch in boxed:
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            if ch == "," and depth == 0:
                parts.append(cur)
                cur = ""
            else:
                cur += ch
        parts.append(cur)
        return [p.strip() for p in parts if p.strip() != ""]

    def auto_judge(self, pred, gold, options=None):
        """
        pred: full model output string
        gold: list of gold answer strings (one per blank)
        options: unused shim for signature compatibility
        Returns True only if every blank matches.
        """
        if not isinstance(gold, list):
            gold = [gold]
        boxed = _extract_boxed(pred)
        if boxed is None:
            return False
        pred_blanks = self._split_pred_blanks(boxed)

        # If counts differ but gold is single, compare whole boxed to it.
        if len(gold) == 1 and len(pred_blanks) != 1:
            pred_blanks = [boxed.strip()]

        if len(pred_blanks) != len(gold):
            return False

        for p, g in zip(pred_blanks, gold):
            if not _equiv(p, g):
                return False
        return True
