import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mathrl.common import answer_equal, grade, reward_for, split_top_level, last_boxed, extract_mcq_letter, R_CORRECT, R_NO_ANSWER, R_WRONG

SHOULD_MATCH = [
    ("\\dfrac{x\\sqrt{36 - x^2}}{x^2 - 18}", "2*x*sqrt(6^2-x^2)/(2*x^2-6^2)"),
    ("0.928", "2.947*1.26/[sqrt(16)]"),
    ("8\\cos^3(x) + 6\\cos^2(x) - 6", "8*[cos(x)]^3+6*[cos(x)]^2-6"),
    ("True", "T"), ("false", "F"),
    ("\\frac{\\sqrt{3}}{3}", "1/sqrt(3)"),
    ("12.08", "12.0814"),                 # 0.01% off -> within 0.1%
    ("(-1, -3)", "(-1,-3)"),
    ("negative", "NEGATIVE"),
    ("\\text{neither}", "neither"),
    ("0.076x", "7.6/100*x"),
    ("5\\left(\\frac{7}{4}\\right)^x", "5*1.75**x"),
    ("y = 2x+1", "2*x+1"),
    ("2x+1", "y=2*x+1"),
    ("1,000", "1000"),
    ("(-8, \\infty)", "(-8,infinity)"),
    ("\\frac{\\pi}{2}", "1.5707963"),
    ("e^{2}", "7.389056"),
    ("BC", "BC"), ("C,F", "CF"),
]
SHOULD_NOT_MATCH = [
    ("2.33", "2.32624773420025"),         # 0.16% off -> outside tolerance
    ("(1, 3)", "[1,3]"),                   # interval bracket types differ
    ("-0.770", "ln(0.462)"),
    ("x+1", "x-1"),
    ("B", "C"),
    ("increasing", "decreasing"),
    ("\\frac{7}{4}, -\\frac{7}{4}", "(1.75, -1.75)"),   # point needs parentheses
    ("", "5"),
    ("x", "A"),                            # lowercase variable is not an option letter
]

def check():
    bad = []
    for p, g in SHOULD_MATCH:
        if not answer_equal(p, g): bad.append(("expected MATCH", p, g))
    for p, g in SHOULD_NOT_MATCH:
        if answer_equal(p, g): bad.append(("expected NO match", p, g))
    # splitting
    assert split_top_level("(1, 2), 3, [4,5]") == ["(1, 2)", "3", "[4,5]"], split_top_level("(1, 2), 3, [4,5]")
    assert split_top_level("a; b, c") == ["a", "b, c"]
    assert last_boxed("x \\boxed{\\frac{1}{2}} y \\boxed{3, \\{4\\}}") == "3, \\{4\\}"
    # full responses
    mcq = {"question": "q", "options": ["1"] * 10, "answer": "C"}
    assert grade("...</think>\n\nSo \\boxed{C}", mcq) == (True, 1.0)
    assert grade("...</think>\n\n\\boxed{\\text{(C)}}", mcq)[0]
    assert grade("I think the answer is D maybe", mcq) == (False, 0.0)      # no box, no 'answer is' in final part
    multi = {"question": "a [ANS] b [ANS]", "answer": ["3", "(1,2)"]}
    assert grade("</think> \\boxed{3, (1, 2)}", multi) == (True, 1.0)
    assert grade("</think> \\boxed{3, (1, 5)}", multi) == (False, 0.5)
    assert grade("</think> \\boxed{3}", multi) == (False, 0.0)               # wrong count -> no credit
    letters = {"question": "A. x B. y [ANS] [ANS]", "answer": ["G", "B"]}
    assert grade("</think> \\boxed{GB}", letters)[0] and grade("</think> \\boxed{G, B}", letters)[0]
    # reward
    assert reward_for("reasoning</think>\n\n\\boxed{3, (1,2)}", multi, truncated=False) == R_CORRECT
    assert reward_for("reasoning \\boxed{3, (1,2)}", multi, truncated=False) == R_NO_ANSWER   # never closed </think>
    assert reward_for("reasoning</think>\\boxed{3, (1,2)}", multi, truncated=True) == R_NO_ANSWER
    assert reward_for("reasoning</think>\\boxed{4, (9,9)}", multi, truncated=False) == R_WRONG
    assert 0 < reward_for("r</think>\\boxed{3, (9,9)}", multi, truncated=False) < R_CORRECT
    return bad

if __name__ == "__main__":
    bad = check()
    for b in bad: print("FAIL", b)
    print("grader tests:", "ALL PASS" if not bad else f"{len(bad)} FAIL")
    sys.exit(1 if bad else 0)
