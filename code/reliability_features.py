"""Input-only format and bounded arithmetic diagnostics. Never execute candidate code.

These signals do not change the official parser, and are not proofs of correctness.
Symbolic expressions and unsupported LaTeX are deliberately skipped.
"""
import ast
from fractions import Fraction
import re
import numpy as np
from common import parse_answer

FORMAT_NAMES = ['hash_present', 'hash_count', 'last_hash_heading_like',
                'boxed_present', 'hash_box_disagree', 'multiple_boxes_disagree',
                'box_after_hash', 'official_parse_missing']
ARITH_NAMES = ['equations_checked', 'equations_good', 'equations_bad', 'bad_fraction',
               'any_bad', 'last_checked_bad', 'mean_relative_error', 'max_relative_error']

def format_features(text):
    official = parse_answer(text)
    hashes = list(re.finditer(r'####', text))
    boxes = list(re.finditer(r'\\boxed\{([^{}]+)\}', text))
    box_values = [parse_answer('#### '+m.group(1)) for m in boxes]
    last_box = box_values[-1] if box_values else None
    tail = text[hashes[-1].end():].splitlines()[0] if hashes and text[hashes[-1].end():].splitlines() else ''
    heading_like = bool(hashes and re.search(r'\b(?:job|step|case|option|part|example|method)\s*\d', tail, re.I))
    return [bool(hashes), np.log1p(len(hashes)), heading_like, bool(boxes),
            official is not None and last_box is not None and official != last_box,
            len(set(v for v in box_values if v is not None)) > 1,
            bool(boxes and hashes and boxes[-1].start() > hashes[-1].start()), official is None]

def normalize_math(text):
    s = text.replace('−', '-').replace('–', '-').replace('×', '*').replace('÷', '/')
    s = s.replace('\\times', '*').replace('\\cdot', '*').replace('\\div', '/')
    s = s.replace('\\left', '').replace('\\right', '').replace('\\%', '%')
    s = re.sub(r'\\(?:,|;|!|quad\b|qquad\b)', ' ', s)
    # Keep words as barriers: deleting units could join unrelated numbers.
    s = re.sub(r'\\(?:text|mathrm|operatorname)\{[^{}]*\}', ' | ', s)
    for _ in range(8):
        updated = re.sub(r'\\(?:dfrac|tfrac|frac)\{([^{}]+)\}\{([^{}]+)\}', r'((\1)/(\2))', s)
        if updated == s: break
        s = updated
    s = re.sub(r'\\boxed\{([^{}]+)\}', r'(\1)', s)
    s = re.sub(r'\^\{([^{}]+)\}', r'^(\1)', s)
    s = re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))', '', s)
    # Preserve math delimiters as boundaries, so following prose is not a variable.
    s = s.replace('**', '').replace('$', ':')
    return s

def numeric_value(expression):
    expression = expression.strip()
    if len(expression) > 240 or not re.fullmatch(r'[\d\s.()+*/^%\-]+', expression):
        raise ValueError('Unsupported expression')
    expression = re.sub(r'(\d+(?:\.\d+)?)\s*%', r'(\1/100)', expression).replace('^', '**')
    tree = ast.parse(expression, mode='eval')
    if sum(1 for _ in ast.walk(tree)) > 100: raise ValueError('Expression too large')
    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = Fraction(str(node.value))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp):
            a, b = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add): value = a+b
            elif isinstance(node.op, ast.Sub): value = a-b
            elif isinstance(node.op, ast.Mult): value = a*b
            elif isinstance(node.op, ast.Div): value = a/b
            elif isinstance(node.op, ast.Pow) and b.denominator == 1 and abs(b) <= 6: value = a**int(b)
            else: raise ValueError('Unsupported operator')
        else: raise ValueError('Unsupported syntax')
        if abs(value.numerator) > 10**30 or value.denominator > 10**30: raise ValueError('Out of bounds')
        return value
    return visit(tree.body)

_CHAIN = re.compile(r'[\d .()+*/^%\-]+(?:=[\d .()+*/^%\-]+)+')

def arithmetic_checks(text):
    """Return numeric equalities with evidence; no symbolic algebra or unit reasoning."""
    findings = []
    for line in normalize_math(text).splitlines():
        # A leading spaced hyphen may be a Markdown bullet or a sign: skip ambiguity.
        if re.match(r'^\s*-\s+', line):
            continue
        for match in _CHAIN.finditer(line):
            # Reject numeric suffixes of symbolic expressions and partial RHS values.
            before = line[:match.start()].rstrip()
            after = line[match.end():].lstrip()
            unsafe_first = bool(before and (before[-1].isalnum() or before[-1] in '_\\}|'))
            unsafe_last = bool(after and (after[0].isalnum() or after[0] in '_\\{'))
            pieces = match.group().strip().split('=')
            # "8/25*100=32%" is conventional percentage notation; avoid treating it
            # as an ordinary numeric equality. Decimal multipliers remain supported.
            if '%' in match.group():
                continue
            for pair_index, (left, right) in enumerate(zip(pieces, pieces[1:])):
                # A symbolic/label boundary invalidates only that adjacent pair;
                # fully numeric interior pairs of an equality chain remain checkable.
                if (unsafe_first and pair_index == 0) or (unsafe_last and pair_index == len(pieces)-2):
                    continue
                try:
                    a, b = numeric_value(left), numeric_value(right)
                except (ValueError, SyntaxError, ZeroDivisionError, OverflowError, RecursionError):
                    continue
                error = float(abs(a-b))
                tolerance = 1e-7
                # Allow ordinary rounding to the explicitly printed decimal precision.
                decimal = re.fullmatch(r'\s*[-+]?\d*\.(\d+)\s*', right)
                if decimal: tolerance = max(tolerance, .5*10**(-len(decimal.group(1))))
                findings.append({'left': left.strip(), 'right': right.strip(),
                    'correct': error <= tolerance+1e-12,
                    'relative_error': min(10., error/max(1., abs(float(a)), abs(float(b))))})
    return findings

def arithmetic_features(text):
    checks = arithmetic_checks(text)
    n = len(checks); bad = sum(not c['correct'] for c in checks)
    errors = [c['relative_error'] for c in checks if not c['correct']]
    return [np.log1p(n), np.log1p(n-bad), np.log1p(bad), bad/max(n, 1), bad > 0,
            bool(checks and not checks[-1]['correct']), np.mean(errors) if errors else 0., max(errors, default=0.)]

def reliability_features(bank, mode):
    rows = []
    for row in bank:
        group = []
        for candidate in row['candidates']:
            text = candidate['text']
            fs = format_features(text) if mode in ('format', 'both') else []
            if mode in ('arithmetic', 'both'): fs += arithmetic_features(text)
            group.append(fs)
        rows.append(group)
    return np.asarray(rows, dtype=np.float32)
