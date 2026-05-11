from __future__ import annotations

import streamlit as st

from krope_core import (
    DEFAULT_PHI,
    DEFAULT_THETA,
    MODEL_ADDITIVE,
    MODEL_DUAL,
    MODEL_ORDER,
    MODEL_ROPE,
    ROLE_ID,
    ROLE_KO,
    SENTENCE_PAIRS,
    aggregate_model_means,
    attention_heatmap_svg,
    compute_pair,
    frequency_bands_svg,
    make_custom_pair,
    mean_delta_bar_svg,
    relation_rows_for_display,
    role_ko,
    token_rows_for_display,
)


st.set_page_config(
    page_title="K-RoPE Mini Simulator",
    page_icon="🧭",
    layout="wide",
)


MODEL_KO = {
    MODEL_ROPE: "일반 RoPE",
    MODEL_ADDITIVE: "가산형 K-RoPE",
    MODEL_DUAL: "이중축 K-RoPE",
}

DEFAULT_CUSTOM_A = "cheolsu,철수가,SUBJ\nyounghee,영희를,OBJ\nlike,좋아한다,PRED"
DEFAULT_CUSTOM_B = "younghee,영희를,OBJ\ncheolsu,철수가,SUBJ\nlike,좋아한다,PRED"
DEFAULT_CUSTOM_REL = "cheolsu,like,주어→서술어\nyounghee,like,목적어→서술어"


def parse_csv_lines(text: str, expected_fields: int, label: str) -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = []
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = tuple(part.strip() for part in line.split(","))
        if len(parts) != expected_fields:
            raise ValueError(f"{label} {line_no}행: 쉼표로 구분된 {expected_fields}개 값이 필요합니다. 입력={raw_line!r}")
        rows.append(parts)
    if not rows:
        raise ValueError(f"{label}: 최소 1개 행이 필요합니다.")
    return rows


def parse_custom_pair() -> dict[str, object] | None:
    st.markdown("#### 사용자 정의 문장")
    st.caption("자동 형태소/문장성분 분석기가 아닙니다. 어절과 문법 역할은 직접 지정합니다.")
    role_help = ", ".join(f"{code}={name}" for code, name in ROLE_KO.items())
    st.code(role_help, language="text")

    col_a, col_b, col_r = st.columns([1, 1, 1])
    with col_a:
        tokens_a_text = st.text_area("문장 A 토큰: token_id,어절,역할코드", DEFAULT_CUSTOM_A, height=150)
    with col_b:
        tokens_b_text = st.text_area("문장 B 토큰: token_id,어절,역할코드", DEFAULT_CUSTOM_B, height=150)
    with col_r:
        relations_text = st.text_area("비교 관계: source_id,target_id,관계명", DEFAULT_CUSTOM_REL, height=150)

    try:
        tokens_a_raw = parse_csv_lines(tokens_a_text, 3, "문장 A")
        tokens_b_raw = parse_csv_lines(tokens_b_text, 3, "문장 B")
        relations = parse_csv_lines(relations_text, 3, "관계")
        tokens_a = [(tid, surface, role if role in ROLE_ID else "OTHER") for tid, surface, role in tokens_a_raw]
        tokens_b = [(tid, surface, role if role in ROLE_ID else "OTHER") for tid, surface, role in tokens_b_raw]
        return make_custom_pair(tokens_a, tokens_b, relations, name="사용자 정의 문장쌍")
    except Exception as exc:
        st.error(str(exc))
        return None


def heatmap_height(token_count: int) -> int:
    cell = 76 if token_count <= 5 else 66
    return 98 + token_count * cell + 110


def render_svg(svg: str, height: int) -> None:
    # Streamlit 1.57+ deprecates components.html in favor of st.iframe.
    # Passing the SVG string directly embeds it as raw HTML in an iframe.
    st.iframe(svg, width="stretch", height=height)


st.title("K-RoPE Mini Simulator")
st.caption("한국어 어순 변화와 격조사 기반 문법 역할을 반영한 RoPE 변형 모형 교육용 시뮬레이터")

with st.sidebar:
    st.header("실험 설정")
    pair_names = [str(pair["name"]) for pair in SENTENCE_PAIRS] + ["사용자 정의"]
    selected_name = st.selectbox("문장쌍", pair_names, index=0)

    theta = st.slider(
        "위치 회전 단위 θ",
        min_value=0.0,
        max_value=2.0,
        value=float(DEFAULT_THETA),
        step=0.01,
        help="위치가 한 칸 증가할 때 적용되는 기본 회전량입니다. 기본값은 보고서 실험값입니다.",
    )
    phi = st.slider(
        "문법 역할 회전 단위 φ",
        min_value=0.0,
        max_value=2.0,
        value=float(DEFAULT_PHI),
        step=0.01,
        help="문법 역할 번호 r_i에 곱해지는 회전량입니다. K-RoPE 모형에서만 사용됩니다.",
    )
    use_frequency_bands = st.checkbox(
        "RoPE 주파수 밴드 사용",
        value=False,
        help="끄면 보고서의 4차원 교육용 단일 θ 모드와 수치가 일치합니다. 켜면 θ_t=10000^{-2(t-1)/d}를 위치축에 반영합니다.",
    )
    st.divider()
    st.markdown("**해석 기준**")
    st.markdown("Δ가 작을수록 어순이 바뀌어도 같은 문법 관계의 attention이 안정적으로 유지됩니다.")

if selected_name == "사용자 정의":
    pair = parse_custom_pair()
    if pair is None:
        st.stop()
else:
    pair = next(pair for pair in SENTENCE_PAIRS if pair["name"] == selected_name)

result = compute_pair(pair, theta=theta, phi=phi, use_frequency_bands=use_frequency_bands)
all_means = aggregate_model_means(SENTENCE_PAIRS, theta=theta, phi=phi, use_frequency_bands=use_frequency_bands)

st.markdown(
    """
이 앱은 실제 LLM을 학습시키는 도구가 아니라, RoPE와 K-RoPE의 수학적 차이를 보기 위한 **교육용 미니 self-attention 시뮬레이터**입니다.  
query와 key는 같은 교육용 역할 벡터에서 만들어진다고 단순화하고, attention score는 성분별 곱의 합과 softmax로 계산합니다.
"""
)

metric_cols = st.columns(3)
for col, model_name in zip(metric_cols, MODEL_ORDER):
    with col:
        st.metric(MODEL_KO[model_name], f"Δ={result.models[model_name].mean_delta:.6f}")

main_tab, heatmap_tab, formula_tab, token_tab, custom_note_tab = st.tabs([
    "모형 비교",
    "Attention heatmap",
    "수식·주파수",
    "토큰·관계",
    "사용 방법",
])

with main_tab:
    st.subheader("선택한 문장쌍의 안정성 비교")
    st.markdown(f"**문장 A**: {result.a_text}")
    st.markdown(f"**문장 B**: {result.b_text}")
    selected_means = {model_name: result.models[model_name].mean_delta for model_name in MODEL_ORDER}
    render_svg(mean_delta_bar_svg(selected_means, title="선택한 문장쌍: 모형별 평균 Δ"), height=430)

    st.subheader("전체 기본 문장쌍 평균")
    st.caption("사용자 정의 문장은 제외하고, 보고서에 사용한 기본 문장쌍 전체에 대해 계산합니다.")
    render_svg(mean_delta_bar_svg(all_means, title="기본 문장쌍 전체: 모형별 평균 Δ"), height=430)

    st.markdown("#### 관계쌍별 수치")
    selected_model_for_table = st.selectbox("표로 볼 모형", MODEL_ORDER, format_func=lambda name: MODEL_KO[name])
    st.dataframe(relation_rows_for_display(result.models[selected_model_for_table]), width="stretch")

with heatmap_tab:
    st.subheader("Attention heatmap")
    model_name = st.radio("시각화할 모형", MODEL_ORDER, format_func=lambda name: MODEL_KO[name], horizontal=True)
    model = result.models[model_name]
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("##### 문장 A")
        labels_a = [text for _tid, text, _role in result.tokens_a]
        render_svg(
            attention_heatmap_svg(model.attention_a, labels_a, f"{MODEL_KO[model_name]}: 문장 A"),
            height=heatmap_height(len(labels_a)),
        )
    with col_b:
        st.markdown("##### 문장 B")
        labels_b = [text for _tid, text, _role in result.tokens_b]
        render_svg(
            attention_heatmap_svg(model.attention_b, labels_b, f"{MODEL_KO[model_name]}: 문장 B"),
            height=heatmap_height(len(labels_b)),
        )
    st.caption("각 행은 한 토큰이 다른 토큰들을 참고하는 비율입니다. softmax를 적용했기 때문에 각 행의 합은 1입니다.")

with formula_tab:
    st.subheader("핵심 수식")
    st.markdown("RoPE는 2차원 성분쌍을 위치에 따라 회전시킵니다.")
    st.latex(r"x'=x\cos(i\theta)-y\sin(i\theta)")
    st.latex(r"y'=x\sin(i\theta)+y\cos(i\theta)")

    st.markdown("두 위치 $i,j$를 비교하면 회전각의 차이 때문에 상대 위치가 나타납니다.")
    st.latex(r"R(i\theta)^T R(j\theta)=R((j-i)\theta)")
    st.latex(r"\operatorname{score}(i,j)=\mathbf{q}^T R((j-i)\theta)\mathbf{k}")

    st.markdown("가산형 K-RoPE와 이중축 K-RoPE는 다음처럼 구분합니다.")
    st.latex(r"\theta_i^K=i\theta+r_i\phi")
    st.latex(r"R^K(i)=R(i\theta)\oplus R(r_i\phi)")

    st.subheader("회전 주파수 밴드")
    st.latex(r"\theta_t=10000^{-2(t-1)/d}")
    st.latex(r"\alpha_{m,t}=m\theta_t")
    st.latex(r"\Delta\alpha_t=(n-m)\theta_t")
    dim = st.select_slider("주파수 그래프 차원 d", options=[4, 8, 16, 32], value=8)
    max_position = st.slider("그래프 최대 위치 m", 20, 100, 40, 5)
    render_svg(frequency_bands_svg(dim=dim, max_position=max_position), height=545)

with token_tab:
    st.subheader("토큰과 문법 역할")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("##### 문장 A")
        st.dataframe(token_rows_for_display(result.tokens_a), width="stretch")
    with col_b:
        st.markdown("##### 문장 B")
        st.dataframe(token_rows_for_display(result.tokens_b), width="stretch")

    st.subheader("비교하는 문법 관계")
    relation_display = [
        {"source_id": src, "target_id": dst, "관계": label}
        for src, dst, label in result.relations
    ]
    st.dataframe(relation_display, width="stretch")

with custom_note_tab:
    st.subheader("앱 사용 방법")
    st.markdown(
        r"""
1. 왼쪽 사이드바에서 문장쌍을 고릅니다.  
2. $\theta$와 $\phi$를 조절합니다.  
3. `모형 비교` 탭에서 평균 $\Delta$를 확인합니다.  
4. `Attention heatmap` 탭에서 어순 변화 전후의 attention 분포를 비교합니다.  
5. `수식·주파수` 탭에서 RoPE 회전식과 주파수 밴드를 확인합니다.

주의: 사용자 정의 모드는 한국어 문장을 자동 분석하지 않습니다. 어절과 문법 역할을 직접 입력해야 합니다. 이 점을 명시해야 탐구 결과를 과장하지 않게 됩니다.
"""
    )
    st.markdown("#### 역할 코드")
    role_rows = [{"역할 코드": code, "의미": name, "역할 번호": ROLE_ID[code]} for code, name in ROLE_KO.items()]
    st.dataframe(role_rows, width="stretch")
