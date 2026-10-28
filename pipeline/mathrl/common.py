"""Shared code for every stage: config, prompts, data split, answer extraction, grading.

Every notebook writes this exact file, so prompts / split / grader are identical
across PE, PE+SFT and PE+SFT+RL.
"""
import json, os, random, re, signal, math
from contextlib import contextmanager

# ----------------------------------------------------------------------------
# Frozen evaluation protocol (identical for every model we compare)
# ----------------------------------------------------------------------------
BASE_MODEL = "unsloth/Qwen3-4B-Thinking-2507"   # bf16 copy of Qwen/Qwen3-4B-Thinking-2507
EVAL_MAX_TOKENS = 16384                         # Qwen card recommends >=32k; 16k fits our budget
EVAL_SAMPLING = dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0)  # Qwen-recommended
SPLIT_SEED = 151
N_TEST = 250
N_DEV = 125
LORA_RANK = 32

# ----------------------------------------------------------------------------
# Prompt engineering (reused verbatim from the 47.25% run)
# ----------------------------------------------------------------------------
SYSTEM_PROMPT_MCQ = (
    "You are a world-class mathematician. "
    "This is a multiple choice question - select the best answer. "
    "Reason briefly. "
    "You MUST ALWAYS end your response with \\boxed{X} where X is your chosen letter. "
    "Always attempt the problem even if you are unsure. Never say you cannot solve it. "
    "After reaching a conclusion, immediately write \\boxed{} without additional commentary."
)
SYSTEM_PROMPT_MATH = (
    "You are a world-class mathematician. "
    "Solve the problem step by step. "
    "Count the number of [ANS] placeholders in the question - you must provide exactly that many answers. "
    "Separate multiple answers by commas inside a single \\boxed{}, e.g. \\boxed{3, 7}. "
    "You MUST ALWAYS end your response with \\boxed{}. "
    "Always attempt the problem even if you are unsure. Never say you cannot solve it. "
    "After reaching a conclusion, immediately write \\boxed{} without additional commentary."
)
SYSTEM_PROMPT_OTHER = (
    "You are a world-class mathematician. "
    "This question has answer choices embedded in the text. "
    "Reason briefly. "
    "Select the correct letter(s) and put them inside \\boxed{}. "
    "If multiple answers are correct write them together like \\boxed{BC}. "
    "You MUST ALWAYS end your response with \\boxed{}. "
    "Always attempt the problem even if you are unsure. Never say you cannot solve it. "
    "After reaching a conclusion, immediately write \\boxed{} without additional commentary."
)
LETTERS = "ABCDEFGHIJKLMNOP"


# The original baseline's ("starter") prompts, copied verbatim from the baseline notebook (Main1-2.ipynb),
# including its missing spaces between sentences. Only used to re-run the baseline at 16k for a fair comparison.
STARTER_PROMPT_MATH = (
    "You are a world-class mathematician."
    "Put your final answer inside \\boxed{}. "
    "Solve the problem. Then independently verify your answer by checking it from a different angle or plugging it back in."
    "Be very concise"
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{3, 7}."
)
STARTER_PROMPT_MCQ = (
    "You are a world-class mathematician."
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
    "Be very concise"
)
STARTER_PROMPT_OTHER = (
    "You are a world-class mathematician."
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
    "Be very concise"
    "you can pick multiple answers if they exist"
)
PROMPT_SETS = {
    "pe": (SYSTEM_PROMPT_MCQ, SYSTEM_PROMPT_OTHER, SYSTEM_PROMPT_MATH),
    "starter": (STARTER_PROMPT_MCQ, STARTER_PROMPT_OTHER, STARTER_PROMPT_MATH),
}


def build_messages(question, options=None, prompt_set="pe"):
    """Same routing as the PE notebook (and the baseline notebook): MCQ / embedded choices / math."""
    mcq, other, math_ = PROMPT_SETS[prompt_set]
    if options:
        opts = "\n".join(f"{LETTERS[i]}. {str(o).strip()}" for i, o in enumerate(options))
        return [{"role": "system", "content": mcq},
                {"role": "user", "content": f"{question}\n\nOptions:\n{opts}"}]
    if "A." in question or "B." in question:
        return [{"role": "system", "content": other},
                {"role": "user", "content": question}]
    return [{"role": "system", "content": math_},
            {"role": "user", "content": question}]


def prompt_text(tokenizer, item, prompt_set="pe"):
    """The exact text fed to the model (chat template + generation prompt, ends with '<think>\\n')."""
    return tokenizer.apply_chat_template(build_messages(item["question"], item.get("options"), prompt_set),
                                         tokenize=False, add_generation_prompt=True)


# ----------------------------------------------------------------------------
# Data + split
# ----------------------------------------------------------------------------
def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def gold_list(item):
    a = item["answer"]
    return [str(x) for x in a] if isinstance(a, list) else [str(a)]


def qtype(item):
    if item.get("options"):
        return "mcq"
    return "free_multi" if len(gold_list(item)) > 1 else "free_single"


def make_split(items, seed=SPLIT_SEED, n_test=N_TEST, n_dev=N_DEV):
    """Stratified by question type. Deterministic. Returns dict id -> 'test'|'dev'|'train'."""
    by_type = {}
    for it in sorted(items, key=lambda x: str(x["id"])):
        by_type.setdefault(qtype(it), []).append(it)
    rng = random.Random(seed)
    total = len(items)
    split = {}
    for t, group in sorted(by_type.items()):
        rng.shuffle(group)
        k_test = round(n_test * len(group) / total)
        k_dev = round(n_dev * len(group) / total)
        for i, it in enumerate(group):
            split[it["id"]] = "test" if i < k_test else ("dev" if i < k_test + k_dev else "train")
    return split


def norm_question(q):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(q).lower())).strip()


class NearDupIndex:
    """Flags outside questions that are near-copies of ours (reworded, re-spaced, options reordered), not only exact
    matches, so MMLU-Pro / OpenR1 training data can't smuggle in a test question.
      * question text: shares >= half of its 6-word shingles with one of our questions
      * MCQ options: >= 5 identical non-trivial options (>= 4 chars) with one of our questions"""

    def __init__(self, items, n=6):
        self.n, self.idx, self.size, self.opt_idx = n, {}, {}, {}
        for it in items:
            sh = self._shingles(it["question"])
            self.size[it["id"]] = len(sh)
            for x in sh:
                self.idx.setdefault(x, set()).add(it["id"])
            for o in it.get("options") or []:
                k = norm_question(o)
                if len(k) >= 4:
                    self.opt_idx.setdefault(k, set()).add(it["id"])

    def _shingles(self, text):
        w = norm_question(text).split()
        return {" ".join(w[i:i + self.n]) for i in range(max(1, len(w) - self.n + 1))} if w else set()

    def is_dup(self, question, options=None):
        sh = self._shingles(question)
        hits = {}
        for x in sh:
            for i in self.idx.get(x, ()):
                hits[i] = hits.get(i, 0) + 1
        if any(h >= 0.5 * min(len(sh), self.size[i]) for i, h in hits.items()):
            return True
        oh = {}
        for o in options or []:
            k = norm_question(o)
            if len(k) >= 4:
                for i in self.opt_idx.get(k, ()):
                    oh[i] = oh.get(i, 0) + 1
        return any(v >= 5 for v in oh.values())


# ----------------------------------------------------------------------------
# Resumable JSONL output
# ----------------------------------------------------------------------------
def done_keys(path, key="id"):
    if not os.path.exists(path):
        return set()
    out = set()
    with open(path) as f:
        for l in f:
            try:
                out.add(json.loads(l)[key])
            except Exception:
                pass  # a half-written last line after a crash is skipped and redone
    return out


def append_jsonl(path, rows):
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ----------------------------------------------------------------------------
# Answer extraction
# ----------------------------------------------------------------------------
def all_boxed(text):
    """Contents of every \\boxed{...} / \\fbox{...} (brace-matched), in order."""
    out = []
    for m in re.finditer(r"\\(?:boxed|fbox)\s*\{", text or ""):
        i = m.end(); depth = 1; j = i
        while j < len(text) and depth:
            if text[j] == "{": depth += 1
            elif text[j] == "}": depth -= 1
            j += 1
        if depth == 0:
            out.append(text[i:j - 1].strip())
    return out


def last_boxed(text):
    b = all_boxed(text)
    return b[-1] if b else None


def final_section(text):
    """Text after the reasoning block (if the model closed it)."""
    return text.split("</think>", 1)[1] if "</think>" in (text or "") else None


def split_top_level(s):
    """Split on top-level ';' if present, else top-level ','. Respects (), [], {}."""
    def _split(sep):
        parts, depth, cur, k = [], 0, [], 0
        while k < len(s):
            ch = s[k]
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth = max(0, depth - 1)
            if ch == sep and depth == 0:
                parts.append("".join(cur)); cur = []
            else:
                cur.append(ch)
            k += 1
        parts.append("".join(cur))
        return [p.strip() for p in parts if p.strip() != ""]
    semi = _split(";")
    return semi if len(semi) > 1 else _split(",")


# ----------------------------------------------------------------------------
# Grading
# ----------------------------------------------------------------------------
REL_TOL = 1e-3      # WeBWorK-style default relative tolerance (0.1%)
ABS_TOL = 1e-8


class _Timeout(Exception):
    pass


@contextmanager
def time_limit(seconds):
    """SIGALRM timeout; silently disabled outside the main thread."""
    try:
        old = signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(_Timeout()))
        signal.setitimer(signal.ITIMER_REAL, seconds)
        active = True
    except Exception:
        active = False
    try:
        yield
    finally:
        if active:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)


def _strip_latex(s):
    s = str(s).strip()
    s = re.sub(r"\\(text|mathrm|textbf|mathbf|operatorname)\s*\{([^{}]*)\}", r"\2", s)
    for a, b in [("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""), ("\\;", ""), ("\\:", ""),
                 ("\\ ", " "), ("$", ""), ("\\displaystyle", ""), ("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"),
                 ("\\cdot", "*"), ("\\times", "*"), ("\\div", "/"), ("\\pi", " pi "), ("\\infty", " oo "),
                 ("\\leq", "<="), ("\\geq", ">="), ("\\le", "<="), ("\\ge", ">="),
                 ("^\\circ", ""), ("^{\\circ}", ""), ("\\circ", ""),
                 ("\\%", ""), ("%", ""), ("\\{", "("), ("\\}", ")")]:
        s = s.replace(a, b)
    s = s.strip().rstrip(".").strip()
    return s


def _latex_to_expr(s):
    # Every conversion inserts spaces/'*' so 'x\sqrt{2}' never becomes the symbol run 'xsqrt'.
    for _ in range(3):  # nested fractions / roots
        s = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r" ((\1)/(\2)) ", s)
        s = re.sub(r"\\sqrt\s*\[([^\]]*)\]\s*\{([^{}]*)\}", r" ((\2)**(1/(\1))) ", s)
        s = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r" sqrt(\1) ", s)
    s = re.sub(r"\\sqrt\s*(\d+(\.\d+)?)", r" sqrt(\1) ", s)
    fn = r"(sin|cos|tan|ln|log|exp|arcsin|arccos|arctan|sec|csc|cot|sinh|cosh|tanh)"
    s = re.sub(r"\\" + fn + r"\s*\^\s*\{?(\d+)\}?\s*\(([^()]*)\)", r" (\1(\3))**\2 ", s)  # \cos^3(x)
    s = re.sub(r"\\" + fn + r"\b", r" \1", s)
    s = s.replace("\\", " ").replace("{", "(").replace("}", ")").replace("[", "(").replace("]", ")")
    s = s.replace("^", "**")
    s = re.sub(r"\binfinity\b|\binf\b", "oo", s, flags=re.I)
    s = re.sub(r"(?<=\d),(?=\d{3}(\D|$))", "", s)  # 1,000 -> 1000 (only inside a single value)
    return s.strip()


_SYMPY = None


def _sympy():
    global _SYMPY
    if _SYMPY is None:
        import sympy
        from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
                                                implicit_multiplication_application, convert_xor)
        tr = standard_transformations + (implicit_multiplication_application, convert_xor)
        local = {n: sympy.Symbol(n) for n in ["N", "O", "S", "Q", "C", "I", "i", "E", "beta", "gamma", "lambda", "x", "y", "z", "t"]}
        local["i"] = sympy.Symbol("i"); local["I"] = sympy.Symbol("i")   # imaginary unit: compare symbolically
        local["e"] = sympy.E; local["pi"] = sympy.pi; local["oo"] = sympy.oo
        local["ln"] = sympy.log
        _SYMPY = (sympy, parse_expr, tr, local)
    return _SYMPY


def _parse(s):
    sympy, parse_expr, tr, local = _sympy()
    s = _latex_to_expr(s)
    if not s or len(s) > 400:
        return None
    try:
        with time_limit(2):
            e = parse_expr(s, transformations=tr, local_dict=dict(local), evaluate=True)
        return e if isinstance(e, sympy.Basic) else None
    except BaseException:
        return None


def _num_close(a, b):
    return abs(a - b) <= max(REL_TOL * abs(b), ABS_TOL)


def _expr_equal(p, g):
    sympy, *_ = _sympy()
    try:
        with time_limit(3):
            if p == g:
                return True
            free = sorted(p.free_symbols | g.free_symbols, key=str)
            if not free:
                pv, gv = complex(sympy.N(p)), complex(sympy.N(g))
                if any(map(math.isnan, [pv.real, gv.real])):
                    return False
                return _num_close(pv.real, gv.real) and _num_close(pv.imag, gv.imag)
            # symbolic: random numeric spot checks, then simplify as tie-breaker
            rng = random.Random(0)
            ok = 0
            for _ in range(4):
                sub = {s: rng.uniform(0.5, 2.5) for s in free}
                pv, gv = complex(sympy.N(p.subs(sub))), complex(sympy.N(g.subs(sub)))
                if not (_num_close(pv.real, gv.real) and _num_close(pv.imag, gv.imag)):
                    return False
                ok += 1
            return ok == 4
    except BaseException:
        try:
            with time_limit(3):
                return sympy.simplify(p - g) == 0
        except BaseException:
            return False


def _is_letters(s):
    return re.fullmatch(r"[A-P]+", s) is not None


def _clean_letters(s):
    t = re.sub(r"\\text\{|\}|[\s\(\)\.\,\*\$]", "", str(s)).upper()
    return t if _is_letters(t) else None


def _is_word(s):
    return re.fullmatch(r"[A-Za-z][A-Za-z \-']*", s.strip()) is not None and len(s.strip()) > 1


def _bracketed(s):
    s = s.strip()
    return len(s) >= 2 and s[0] in "([" and s[-1] in ")]"


def answer_equal(pred, gold):
    """One blank: predicted string vs gold string."""
    if pred is None:
        return False
    p, g = _strip_latex(pred), _strip_latex(gold)
    inf = lambda t: re.sub(r"(?i)\binfinity\b|\binf\b", "oo", t)
    p, g = inf(p), inf(g)
    if p == g or p.replace(" ", "").lower() == g.replace(" ", "").lower():
        return True
    if g in ("T", "F") and p.lower() in ("true", "false"):   # True/False questions
        return p[0].upper() == g
    gl = g.strip() if re.fullmatch(r"[A-P]{1,4}", g.strip()) else None
    if gl is not None:                                    # letter answer(s), uppercase golds only
        pl = _clean_letters(p)
        return pl is not None and sorted(set(pl)) == sorted(set(gl))
    if _is_word(g) and not re.search(r"\d", g) and len(g) > 2 and not re.fullmatch(r"[a-z]", g):
        if _is_word(p):
            return p.strip().lower() == g.strip().lower()
    # equations: compare right-hand sides
    if "=" in g and not any(op in g for op in ["<=", ">="]):
        g = g.split("=")[-1]
        if "=" in p:
            p = p.split("=")[-1]
    elif "=" in p and not any(op in p for op in ["<=", ">="]):
        p = p.split("=")[-1]
    # tuples / intervals: bracket types must match, then element-wise
    if _bracketed(g) and "," in g:
        if not _bracketed(p) or p.strip()[0] != g.strip()[0] or p.strip()[-1] != g.strip()[-1]:
            return False
        gp, pp = split_top_level(g.strip()[1:-1]), split_top_level(p.strip()[1:-1])
        return len(gp) == len(pp) and all(answer_equal(a, b) for a, b in zip(pp, gp))
    pe, ge = _parse(p), _parse(g)
    if pe is None or ge is None:
        return False
    return _expr_equal(pe, ge)


def extract_mcq_letter(text):
    b = last_boxed(text)
    if b is not None:
        m = re.search(r"\b([A-P])\b", _strip_latex(b).upper()) or re.fullmatch(r"\(?([A-P])\)?\.?", _strip_latex(b).upper())
        if m:
            return m.group(1)
    tail = final_section(text) or ""
    m = re.findall(r"answer\s*(?:is|:)\s*\(?\**\s*([A-P])\b", tail, flags=re.I)
    return m[-1].upper() if m else None


def pred_parts(text, n_gold):
    """Split the final boxed answer into one string per blank."""
    b = last_boxed(text)
    if b is None:
        return None
    if n_gold == 1:
        return [b]
    parts = split_top_level(b)
    # the OTHER prompt tells the model to write letters together (\boxed{BC}) -> one letter per blank
    if len(parts) == 1 and _clean_letters(parts[0]) and len(_clean_letters(parts[0])) == n_gold:
        parts = list(_clean_letters(parts[0]))
    return parts


def grade(text, item):
    """Returns (correct: bool, frac: float in [0,1]). Mirrors 'last \\boxed{}' extraction."""
    golds = gold_list(item)
    if item.get("options"):
        ok = extract_mcq_letter(text) == golds[0].strip().upper()
        return ok, float(ok)
    parts = pred_parts(text, len(golds))
    if parts is None or len(parts) != len(golds):
        return False, 0.0
    hits = sum(answer_equal(p, g) for p, g in zip(parts, golds))
    return hits == len(golds), hits / len(golds)


# ----------------------------------------------------------------------------
# RL reward (uses the same grader; strict about finishing and formatting)
# ----------------------------------------------------------------------------
R_CORRECT, R_PARTIAL_MAX, R_WRONG, R_NO_ANSWER = 1.0, 0.5, 0.0, -0.5


def reward_for(text, item, truncated):
    final = final_section(text)
    if truncated or final is None or last_boxed(final) is None:
        return R_NO_ANSWER
    ok, frac = grade(final, item)
    if ok:
        return R_CORRECT
    if len(gold_list(item)) > 1 and frac > 0:
        return R_PARTIAL_MAX * frac
    return R_WRONG


def summarize(rows, label=""):
    by = {}
    for r in rows:
        by.setdefault(r["qtype"], []).append(r["correct"])
    allc = [r["correct"] for r in rows]
    out = {"label": label, "n": len(rows), "overall": 100 * sum(allc) / max(1, len(allc))}
    for t in ["mcq", "free_single", "free_multi"]:
        v = by.get(t, [])
        out[t] = 100 * sum(v) / len(v) if v else None
    out["truncated_pct"] = 100 * sum(r.get("finish_reason") == "length" for r in rows) / max(1, len(rows))
    return out


# ----------------------------------------------------------------------------
# Comparing runs honestly: paired differences on the same questions
# ----------------------------------------------------------------------------
def per_question(results_path, items, which):
    """{id: correct} for one results file, re-graded with the current grader, restricted to one split."""
    split, by = make_split(items), {it["id"]: it for it in items}
    out = {}
    for r in load_jsonl(results_path):
        it = by.get(r["id"])
        if it is not None and split[it["id"]] == which:
            out[r["id"]] = bool(grade(r["response"], it)[0])
    return out


def paired_diff(a, b, iters=4000, seed=0):
    """Accuracy of b minus a (points) on questions both answered, with a 95% bootstrap CI over questions.
    With 125-250 questions, single-sample scores move several points from sampling luck alone; this says
    whether a difference is bigger than that."""
    ids = sorted(set(a) & set(b), key=str)
    n = len(ids)
    if n == 0:
        return None
    d = [float(b[i]) - float(a[i]) for i in ids]   # per-question accuracy (0/1, or a mean over samples)
    rng = random.Random(seed)
    boots = sorted(100.0 * sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return {"n": n, "diff": 100.0 * sum(d) / n, "lo": boots[int(0.025 * iters)], "hi": boots[int(0.975 * iters) - 1],
            "fixed": sum(x > 0 for x in d), "broke": sum(x < 0 for x in d)}


# ----------------------------------------------------------------------------
# Compute-unit guard for long steps (survives disconnects: time used is stored next to the outputs)
# ----------------------------------------------------------------------------
class TimeBudget:
    def __init__(self, path, max_hours):
        self.path, self.max_s = path, (max_hours or 0) * 3600
        self.used0 = 0.0
        if os.path.exists(path):
            try:
                self.used0 = float(json.load(open(path))["seconds"])
            except Exception:
                pass
        import time
        self._time, self.t0 = time.time, time.time()

    def used(self):
        return self.used0 + self._time() - self.t0

    def save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"seconds": self.used()}, f)
        os.replace(tmp, self.path)

    def would_exceed(self, next_s):
        """True if doing one more unit of work (estimated next_s seconds) would pass the limit."""
        return self.max_s > 0 and self.used() + next_s > self.max_s
