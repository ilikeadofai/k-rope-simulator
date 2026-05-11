"""Pure math core for the K-RoPE Streamlit simulator.

This module intentionally has no Streamlit dependency.  It contains only the
educational simulation logic used by the app and by the original report:

- 2D rotary transformations with sin/cos.
- Stable row-wise softmax.
- Three comparison models:
  1. RoPE: position-only rotation.
  2. Additive K-RoPE: position angle + grammar-role angle.
  3. Dual-axis K-RoPE: separate position and grammar-role rotation axes.

Important limitation: this is not a trained language model.  The vectors below
are hand-designed educational vectors so that the algebraic effect of rotation
and Korean word-order changes can be inspected directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from html import escape
import math
from typing import Callable, Iterable, Mapping, Sequence

DEFAULT_THETA = 0.85
DEFAULT_PHI = 0.70
FREQUENCY_BASE = 10000.0

MODEL_ROPE = "RoPE"
MODEL_ADDITIVE = "Additive K-RoPE"
MODEL_DUAL = "Dual-axis K-RoPE"
MODEL_ORDER = [MODEL_ROPE, MODEL_ADDITIVE, MODEL_DUAL]

ROLE_ID = {
    "SUBJ": 0,       # 주어
    "OBJ": 1,        # 목적어
    "PRED": 2,       # 주 서술어
    "MOD": 3,        # 관형어
    "ADV": 4,        # 부사어
    "TIME": 5,       # 부사어(시간)
    "LOC": 6,        # 부사어(장소)
    "EMB_SUBJ": 7,   # 안긴문장의 주어
    "OTHER": 8,
}

ROLE_KO = {
    "SUBJ": "주어",
    "OBJ": "목적어",
    "PRED": "서술어",
    "MOD": "관형어",
    "ADV": "부사어",
    "TIME": "부사어(시간)",
    "LOC": "부사어(장소)",
    "EMB_SUBJ": "안긴문장의 주어",
    "OTHER": "기타",
}

# Simplified content vectors by grammatical role.  The vectors are deliberately
# low-dimensional so the report can explain every rotation with high-school
# algebra.  They are not learned embeddings and should not be interpreted as a
# real LLM representation.
ROLE_VEC = {
    "SUBJ":     [1.00, 0.20, 0.90, 0.10],
    "OBJ":      [0.20, 1.00, 0.10, 0.90],
    "PRED":     [0.80, 0.80, 0.75, 0.75],
    "MOD":      [0.55, 0.45, 0.45, 0.35],
    "ADV":      [0.35, 0.55, 0.20, 0.45],
    "TIME":     [0.30, 0.25, 0.20, 0.25],
    "LOC":      [0.25, 0.30, 0.25, 0.20],
    "EMB_SUBJ": [0.90, 0.25, 0.80, 0.15],
    "OTHER":    [0.30, 0.30, 0.30, 0.30],
}

Token = tuple[str, str, str]
Relation = tuple[str, str, str]

SENTENCE_PAIRS: list[dict[str, object]] = [
    {
        "name": "단순문장 1",
        "a_text": "철수가 영희를 좋아한다",
        "a": [("cheolsu", "철수가", "SUBJ"), ("younghee", "영희를", "OBJ"), ("like", "좋아한다", "PRED")],
        "b_text": "영희를 철수가 좋아한다",
        "b": [("younghee", "영희를", "OBJ"), ("cheolsu", "철수가", "SUBJ"), ("like", "좋아한다", "PRED")],
        "relations": [("cheolsu", "like", "주어→서술어"), ("younghee", "like", "목적어→서술어")],
    },
    {
        "name": "단순문장 2",
        "a_text": "학생이 책을 읽었다",
        "a": [("student", "학생이", "SUBJ"), ("book", "책을", "OBJ"), ("read", "읽었다", "PRED")],
        "b_text": "책을 학생이 읽었다",
        "b": [("book", "책을", "OBJ"), ("student", "학생이", "SUBJ"), ("read", "읽었다", "PRED")],
        "relations": [("student", "read", "주어→서술어"), ("book", "read", "목적어→서술어")],
    },
    {
        "name": "단순문장 3",
        "a_text": "고양이가 생선을 먹었다",
        "a": [("cat", "고양이가", "SUBJ"), ("fish", "생선을", "OBJ"), ("eat", "먹었다", "PRED")],
        "b_text": "생선을 고양이가 먹었다",
        "b": [("fish", "생선을", "OBJ"), ("cat", "고양이가", "SUBJ"), ("eat", "먹었다", "PRED")],
        "relations": [("cat", "eat", "주어→서술어"), ("fish", "eat", "목적어→서술어")],
    },
    {
        "name": "복잡문장 1: 관형어와 부사어",
        "a_text": "민수가 어제 도서관에서 빌린 책을 읽었다",
        "a": [
            ("minsu", "민수가", "SUBJ"),
            ("yesterday", "어제", "TIME"),
            ("library", "도서관에서", "LOC"),
            ("borrowed", "빌린", "MOD"),
            ("book", "책을", "OBJ"),
            ("read", "읽었다", "PRED"),
        ],
        "b_text": "어제 도서관에서 빌린 책을 민수가 읽었다",
        "b": [
            ("yesterday", "어제", "TIME"),
            ("library", "도서관에서", "LOC"),
            ("borrowed", "빌린", "MOD"),
            ("book", "책을", "OBJ"),
            ("minsu", "민수가", "SUBJ"),
            ("read", "읽었다", "PRED"),
        ],
        "relations": [
            ("minsu", "read", "주어→서술어"),
            ("book", "read", "목적어→서술어"),
            ("library", "borrowed", "부사어(장소)→관형어"),
            ("borrowed", "book", "관형어→체언"),
        ],
    },
    {
        "name": "복잡문장 2: 안긴문장과 부사어",
        "a_text": "영희가 직접 만든 케이크를 철수가 맛있게 먹었다",
        "a": [
            ("younghee", "영희가", "EMB_SUBJ"),
            ("directly", "직접", "ADV"),
            ("made", "만든", "MOD"),
            ("cake", "케이크를", "OBJ"),
            ("cheolsu", "철수가", "SUBJ"),
            ("deliciously", "맛있게", "ADV"),
            ("eat", "먹었다", "PRED"),
        ],
        "b_text": "철수가 영희가 직접 만든 케이크를 맛있게 먹었다",
        "b": [
            ("cheolsu", "철수가", "SUBJ"),
            ("younghee", "영희가", "EMB_SUBJ"),
            ("directly", "직접", "ADV"),
            ("made", "만든", "MOD"),
            ("cake", "케이크를", "OBJ"),
            ("deliciously", "맛있게", "ADV"),
            ("eat", "먹었다", "PRED"),
        ],
        "relations": [
            ("cheolsu", "eat", "주어→서술어"),
            ("cake", "eat", "목적어→서술어"),
            ("younghee", "made", "안긴문장의 주어→관형어"),
            ("made", "cake", "관형어→체언"),
            ("deliciously", "eat", "부사어→서술어"),
        ],
    },
    {
        "name": "복잡문장 3: 안긴문장의 목적어 이동",
        "a_text": "내가 어제 만난 친구를 선생님이 칭찬했다",
        "a": [
            ("i", "내가", "EMB_SUBJ"),
            ("yesterday", "어제", "TIME"),
            ("met", "만난", "MOD"),
            ("friend", "친구를", "OBJ"),
            ("teacher", "선생님이", "SUBJ"),
            ("praise", "칭찬했다", "PRED"),
        ],
        "b_text": "선생님이 내가 어제 만난 친구를 칭찬했다",
        "b": [
            ("teacher", "선생님이", "SUBJ"),
            ("i", "내가", "EMB_SUBJ"),
            ("yesterday", "어제", "TIME"),
            ("met", "만난", "MOD"),
            ("friend", "친구를", "OBJ"),
            ("praise", "칭찬했다", "PRED"),
        ],
        "relations": [
            ("teacher", "praise", "주어→서술어"),
            ("friend", "praise", "목적어→서술어"),
            ("i", "met", "안긴문장의 주어→관형어"),
            ("met", "friend", "관형어→체언"),
        ],
    },
]


@dataclass(frozen=True)
class RelationScore:
    label: str
    source_id: str
    target_id: str
    score_a: float
    score_b: float
    abs_diff: float


@dataclass(frozen=True)
class ModelResult:
    name: str
    attention_a: list[list[float]]
    attention_b: list[list[float]]
    relations: list[RelationScore]
    mean_delta: float


@dataclass(frozen=True)
class PairResult:
    pair_name: str
    a_text: str
    b_text: str
    tokens_a: list[Token]
    tokens_b: list[Token]
    relations: list[Relation]
    models: dict[str, ModelResult]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    """Componentwise product-sum, i.e. a finite-dimensional dot product."""
    if len(a) != len(b):
        raise ValueError(f"dot requires equal lengths, got {len(a)} and {len(b)}")
    return sum(x * y for x, y in zip(a, b))


def rotate_pair(pair: Sequence[float], angle: float) -> list[float]:
    """Rotate a 2D pair by `angle` radians using the standard rotation matrix."""
    if len(pair) != 2:
        raise ValueError(f"rotate_pair requires length 2, got {len(pair)}")
    c, s = math.cos(angle), math.sin(angle)
    x, y = pair
    return [x * c - y * s, x * s + y * c]


def role_id(role: str) -> int:
    return ROLE_ID.get(role, ROLE_ID["OTHER"])


def role_ko(role: str) -> str:
    return ROLE_KO.get(role, ROLE_KO["OTHER"])


def base_vec(role: str) -> list[float]:
    return list(ROLE_VEC.get(role, ROLE_VEC["OTHER"]))


def vector_pairs(vec: Sequence[float]) -> list[list[float]]:
    if len(vec) % 2:
        raise ValueError("RoPE requires an even-dimensional vector")
    return [list(vec[i:i + 2]) for i in range(0, len(vec), 2)]


def rope_frequencies(dim: int, base: float = FREQUENCY_BASE) -> list[float]:
    """Return RoPE frequency bands θ_t = base^{-2(t-1)/d} for t=1..d/2.

    Python indexes the returned list from 0, so result[0] is θ_1.
    For d=8 and base=10000, this gives [1, 0.1, 0.01, 0.001].
    """
    if dim <= 0 or dim % 2:
        raise ValueError("dim must be a positive even integer")
    if base <= 1:
        raise ValueError("base must be greater than 1")
    return [base ** (-2 * pair_index / dim) for pair_index in range(dim // 2)]


def _position_angle(position: int, pair_index: int, dim: int, theta: float, use_frequency_bands: bool) -> float:
    if not use_frequency_bands:
        return position * theta
    return position * theta * rope_frequencies(dim)[pair_index]


def encode_rope(tokens: Sequence[Token], theta: float = DEFAULT_THETA, *, use_frequency_bands: bool = False) -> list[list[float]]:
    """Position-only RoPE.

    In the default educational mode, every 2D pair rotates by iθ so it exactly
    matches the original report simulator.  In frequency-band mode, the t-th
    pair rotates by iθθ_t, with θ_t following the RoPE geometric sequence.
    """
    encoded: list[list[float]] = []
    for position, (_tid, _text, role) in enumerate(tokens):
        base = base_vec(role)
        dim = len(base)
        pairs = vector_pairs(base)
        rotated: list[float] = []
        for pair_index, pair in enumerate(pairs):
            angle = _position_angle(position, pair_index, dim, theta, use_frequency_bands)
            rotated.extend(rotate_pair(pair, angle))
        encoded.append(rotated)
    return encoded


def encode_additive_krope(
    tokens: Sequence[Token],
    theta: float = DEFAULT_THETA,
    phi: float = DEFAULT_PHI,
    *,
    use_frequency_bands: bool = False,
) -> list[list[float]]:
    """Additive K-RoPE: θ_i^K = iθ + r_iφ.

    With frequency bands enabled, the position part becomes iθθ_t for each 2D
    pair, while the role part remains r_iφ.  This keeps the role term explicit
    and avoids pretending that grammar roles are real RoPE frequencies.
    """
    encoded: list[list[float]] = []
    for position, (_tid, _text, role) in enumerate(tokens):
        role_angle = role_id(role) * phi
        base = base_vec(role)
        dim = len(base)
        pairs = vector_pairs(base)
        rotated: list[float] = []
        for pair_index, pair in enumerate(pairs):
            angle = _position_angle(position, pair_index, dim, theta, use_frequency_bands) + role_angle
            rotated.extend(rotate_pair(pair, angle))
        encoded.append(rotated)
    return encoded


def encode_dual_krope(
    tokens: Sequence[Token],
    theta: float = DEFAULT_THETA,
    phi: float = DEFAULT_PHI,
    *,
    use_frequency_bands: bool = False,
) -> list[list[float]]:
    """Dual-axis K-RoPE: R^K(i)=R(iθ)⊕R(r_iφ) for 4D vectors.

    The first 2D pair is the surface-position axis.  The second 2D pair is the
    grammatical-role axis.  This implements the report's educational model
    directly and keeps the two sources of information disentangled.
    """
    encoded: list[list[float]] = []
    for position, (_tid, _text, role) in enumerate(tokens):
        base = base_vec(role)
        if len(base) != 4:
            raise ValueError("dual-axis K-RoPE currently expects 4D educational vectors")
        position_angle = _position_angle(position, 0, len(base), theta, use_frequency_bands)
        role_angle = role_id(role) * phi
        encoded.append(rotate_pair(base[0:2], position_angle) + rotate_pair(base[2:4], role_angle))
    return encoded


def score_matrix(vecs: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[dot(row_vec, col_vec) for col_vec in vecs] for row_vec in vecs]


def softmax_rows(scores: Sequence[Sequence[float]]) -> list[list[float]]:
    """Numerically stable row-wise softmax."""
    result: list[list[float]] = []
    for row in scores:
        if not row:
            result.append([])
            continue
        row_max = max(row)
        exps = [math.exp(value - row_max) for value in row]
        denom = sum(exps)
        if denom == 0.0 or not math.isfinite(denom):
            raise ArithmeticError("softmax denominator became invalid")
        result.append([value / denom for value in exps])
    return result


def attention_matrix(
    tokens: Sequence[Token],
    model_name: str,
    theta: float = DEFAULT_THETA,
    phi: float = DEFAULT_PHI,
    *,
    use_frequency_bands: bool = False,
) -> list[list[float]]:
    encoder = encoder_for_model(model_name)
    encoded = encoder(tokens, theta, phi, use_frequency_bands=use_frequency_bands) if model_name != MODEL_ROPE else encoder(tokens, theta, use_frequency_bands=use_frequency_bands)
    return softmax_rows(score_matrix(encoded))


def encoder_for_model(model_name: str) -> Callable[..., list[list[float]]]:
    encoders: dict[str, Callable[..., list[list[float]]]] = {
        MODEL_ROPE: encode_rope,
        MODEL_ADDITIVE: encode_additive_krope,
        MODEL_DUAL: encode_dual_krope,
    }
    try:
        return encoders[model_name]
    except KeyError as exc:
        raise ValueError(f"unknown model: {model_name}") from exc


def relation_scores(tokens: Sequence[Token], weights: Sequence[Sequence[float]], relations: Sequence[Relation]) -> dict[str, float]:
    idx = {tid: i for i, (tid, _text, _role) in enumerate(tokens)}
    out: dict[str, float] = {}
    for source_id, target_id, label in relations:
        if source_id not in idx or target_id not in idx:
            raise KeyError(f"relation references missing token: {source_id!r}->{target_id!r}")
        out[label] = weights[idx[source_id]][idx[target_id]]
    return out


def _coerce_tokens(value: object) -> list[Token]:
    return [(str(tid), str(text), str(role)) for tid, text, role in value]  # type: ignore[misc]


def _coerce_relations(value: object) -> list[Relation]:
    return [(str(src), str(dst), str(label)) for src, dst, label in value]  # type: ignore[misc]


def compute_pair(
    pair: Mapping[str, object],
    theta: float = DEFAULT_THETA,
    phi: float = DEFAULT_PHI,
    *,
    use_frequency_bands: bool = False,
) -> PairResult:
    tokens_a = _coerce_tokens(pair["a"])
    tokens_b = _coerce_tokens(pair["b"])
    relations = _coerce_relations(pair["relations"])
    models: dict[str, ModelResult] = {}
    for model_name in MODEL_ORDER:
        weights_a = attention_matrix(tokens_a, model_name, theta, phi, use_frequency_bands=use_frequency_bands)
        weights_b = attention_matrix(tokens_b, model_name, theta, phi, use_frequency_bands=use_frequency_bands)
        scores_a = relation_scores(tokens_a, weights_a, relations)
        scores_b = relation_scores(tokens_b, weights_b, relations)
        relation_rows = [
            RelationScore(
                label=label,
                source_id=source,
                target_id=target,
                score_a=scores_a[label],
                score_b=scores_b[label],
                abs_diff=abs(scores_a[label] - scores_b[label]),
            )
            for source, target, label in relations
        ]
        mean_delta = sum(row.abs_diff for row in relation_rows) / len(relation_rows) if relation_rows else 0.0
        models[model_name] = ModelResult(model_name, weights_a, weights_b, relation_rows, mean_delta)
    return PairResult(
        pair_name=str(pair["name"]),
        a_text=str(pair["a_text"]),
        b_text=str(pair["b_text"]),
        tokens_a=tokens_a,
        tokens_b=tokens_b,
        relations=relations,
        models=models,
    )


def aggregate_model_means(
    pairs: Sequence[Mapping[str, object]],
    theta: float = DEFAULT_THETA,
    phi: float = DEFAULT_PHI,
    *,
    use_frequency_bands: bool = False,
) -> dict[str, float]:
    values: dict[str, list[float]] = {name: [] for name in MODEL_ORDER}
    for pair in pairs:
        result = compute_pair(pair, theta, phi, use_frequency_bands=use_frequency_bands)
        for model_name, model in result.models.items():
            values[model_name].extend(row.abs_diff for row in model.relations)
    return {model_name: sum(rows) / len(rows) if rows else 0.0 for model_name, rows in values.items()}


def relation_rows_for_display(model: ModelResult) -> list[dict[str, object]]:
    return [
        {
            "관계": row.label,
            "문장 A attention": round(row.score_a, 6),
            "문장 B attention": round(row.score_b, 6),
            "Δ=|A-B|": round(row.abs_diff, 6),
        }
        for row in model.relations
    ]


def token_rows_for_display(tokens: Sequence[Token]) -> list[dict[str, object]]:
    return [
        {
            "위치 i": i,
            "token_id": tid,
            "어절": text,
            "역할": role_ko(role),
            "역할 번호 r_i": role_id(role),
        }
        for i, (tid, text, role) in enumerate(tokens)
    ]


def color_for(value: float, vmin: float = 0.0, vmax: float = 1.0) -> str:
    """Color scale from dark purple through teal to yellow."""
    t = max(0.0, min(1.0, (value - vmin) / (vmax - vmin if vmax != vmin else 1.0)))
    if t < 0.5:
        k = t / 0.5
        r = int(68 + (33 - 68) * k)
        g = int(1 + (145 - 1) * k)
        b = int(84 + (140 - 84) * k)
    else:
        k = (t - 0.5) / 0.5
        r = int(33 + (253 - 33) * k)
        g = int(145 + (231 - 145) * k)
        b = int(140 + (231 - 140) * k)
    return f"#{r:02x}{g:02x}{b:02x}"


def attention_heatmap_svg(matrix: Sequence[Sequence[float]], labels: Sequence[str], title: str) -> str:
    n = len(labels)
    cell = 76 if n <= 5 else 66
    left = 150
    top = 98
    width = left + n * cell + 42
    height = top + n * cell + 82
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>.title{font:700 15px sans-serif}.lab{font:12px sans-serif}.num{font:12px monospace}</style>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" class="title">{escape(title)}</text>',
        f'<text x="{width / 2}" y="50" text-anchor="middle" class="lab" fill="#555">행: 참고하는 토큰(query), 열: 참고되는 토큰(key), 각 행의 합은 1</text>',
    ]
    for col, label in enumerate(labels):
        x = left + col * cell + cell / 2
        parts.append(f'<text x="{x}" y="82" text-anchor="middle" class="lab" transform="rotate(-24 {x} 82)">{escape(label)}</text>')
    for row, label in enumerate(labels):
        y = top + row * cell + cell / 2 + 5
        parts.append(f'<text x="{left - 12}" y="{y}" text-anchor="end" class="lab">{escape(label)}</text>')
    for row, values in enumerate(matrix):
        for col, value in enumerate(values):
            x = left + col * cell
            y = top + row * cell
            fill = color_for(float(value))
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="white"/>')
            parts.append(f'<text x="{x + cell / 2}" y="{y + cell / 2 + 5}" text-anchor="middle" class="num" fill="white">{float(value):.2f}</text>')
    parts.append('</svg>')
    return "\n".join(parts)


def mean_delta_bar_svg(means: Mapping[str, float], title: str = "모형별 평균 attention 차이 Δ") -> str:
    width, height = 760, 420
    left, bottom = 90, 330
    chart_height = 230
    max_value = max(max(means.values()) * 1.25, 0.01)
    colors = {MODEL_ROPE: "#7c3aed", MODEL_ADDITIVE: "#059669", MODEL_DUAL: "#f59e0b"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>.title{font:700 18px sans-serif}.lab{font:13px sans-serif}.num{font:13px monospace}</style>',
        f'<text x="{width / 2}" y="34" text-anchor="middle" class="title">{escape(title)}</text>',
        f'<text x="{width / 2}" y="58" text-anchor="middle" class="lab" fill="#555">값이 작을수록 어순 변화 후 같은 문법 관계 attention이 안정적</text>',
        f'<line x1="{left}" y1="{bottom}" x2="{width - 60}" y2="{bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{bottom - chart_height}" x2="{left}" y2="{bottom}" stroke="#333"/>',
    ]
    bar_width = 150
    gap = 55
    for idx, model_name in enumerate(MODEL_ORDER):
        value = float(means[model_name])
        x = left + 60 + idx * (bar_width + gap)
        bar_height = value / max_value * chart_height
        y = bottom - bar_height
        parts.append(f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" fill="{colors[model_name]}"/>')
        parts.append(f'<text x="{x + bar_width / 2}" y="{y - 8}" text-anchor="middle" class="num">{value:.6f}</text>')
        parts.append(f'<text x="{x + bar_width / 2}" y="{bottom + 24}" text-anchor="middle" class="lab">{escape(model_name)}</text>')
    parts.append('</svg>')
    return "\n".join(parts)


def frequency_bands_svg(dim: int = 8, max_position: int = 40, base: float = FREQUENCY_BASE) -> str:
    freqs = rope_frequencies(dim, base=base)
    width, height = 860, 520
    left, right, top, bottom = 90, 80, 105, 90
    plot_w = width - left - right
    plot_h = height - top - bottom

    def sx(position: float) -> float:
        return left + (position / max_position) * plot_w

    def sy(value: float) -> float:
        return top + ((1 - value) / 2) * plot_h

    palette = ["#dc2626", "#2563eb", "#16a34a", "#7c3aed", "#f59e0b", "#0891b2"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#111827"/></marker></defs>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>.title{font:700 22px sans-serif}.label{font:14px sans-serif}.small{font:12px sans-serif}.axis{stroke:#111827;stroke-width:1.5;marker-end:url(#arrow)}.grid{stroke:#e5e7eb;stroke-width:1}</style>',
        '<text x="30" y="40" class="title">RoPE 회전 주파수 비교: 위치 m에 따른 cos(mθ_t)</text>',
        f'<text x="{left}" y="76" class="label">cos(mθ_t)</text>',
    ]
    for value in [-1, -0.5, 0, 0.5, 1]:
        y = sy(value)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{left - 55}" y="{y + 4:.1f}" class="small">{value:g}</text>')
    step = 5 if max_position <= 50 else 10
    for position in range(0, max_position + 1, step):
        x = sx(position)
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" class="grid"/>')
        parts.append(f'<text x="{x - 7:.1f}" y="{top + plot_h + 23}" class="small">{position}</text>')
    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w + 25}" y2="{top + plot_h}" class="axis"/>')
    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left}" y2="{top - 25}" class="axis"/>')
    parts.append(f'<text x="{left + plot_w - 12}" y="{top + plot_h + 50}" class="label">위치 m</text>')

    show_count = min(len(freqs), 6)
    for idx, theta_t in enumerate(freqs[:show_count]):
        pts = []
        for step_index in range(max_position * 10 + 1):
            m = step_index / 10
            pts.append(f'{sx(m):.1f},{sy(math.cos(m * theta_t)):.1f}')
        parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{palette[idx % len(palette)]}" stroke-width="2.5"/>')

    legend_x, legend_y = 615, 120
    legend_h = 44 + show_count * 22
    parts.append(f'<rect x="{legend_x}" y="{legend_y}" width="205" height="{legend_h}" rx="12" fill="white" stroke="#d1d5db"/>')
    parts.append(f'<text x="{legend_x + 14}" y="{legend_y + 24}" class="label">성분쌍별 회전 주파수</text>')
    for idx, theta_t in enumerate(freqs[:show_count]):
        y = legend_y + 48 + idx * 22
        parts.append(f'<line x1="{legend_x + 16}" y1="{y}" x2="{legend_x + 48}" y2="{y}" stroke="{palette[idx % len(palette)]}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x + 58}" y="{y + 4}" class="small">θ{idx + 1}={theta_t:.4g}</text>')
    parts.append('<text x="90" y="495" class="small">θ가 클수록 위치가 조금만 바뀌어도 곡선이 빠르게 흔들리고, θ가 작을수록 먼 위치까지 천천히 변한다.</text>')
    parts.append('</svg>')
    return "\n".join(parts)


def make_custom_pair(tokens_a: Sequence[Token], tokens_b: Sequence[Token], relations: Sequence[Relation], name: str = "사용자 정의") -> dict[str, object]:
    return {
        "name": name,
        "a_text": " ".join(text for _tid, text, _role in tokens_a),
        "b_text": " ".join(text for _tid, text, _role in tokens_b),
        "a": list(tokens_a),
        "b": list(tokens_b),
        "relations": list(relations),
    }


def role_options() -> list[tuple[str, str]]:
    return [(role, ROLE_KO[role]) for role in ROLE_ID]
