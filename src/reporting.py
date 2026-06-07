from typing import Dict, List

import pandas as pd
import streamlit as st


def grade_label(score: int) -> str:
    if score >= 90:
        return "S"
    if score >= 80:
        return "A"
    if score >= 70:
        return "B"
    if score >= 60:
        return "C"
    return "D"


def render_score_card(result: Dict):
    overall = int(result.get("overall_score", 0) or 0)
    grade = result.get("grade") or grade_label(overall)

    c1, c2, c3 = st.columns(3)
    c1.metric("종합 광고 효과 점수", f"{overall}/100")
    c2.metric("등급", str(grade))
    c3.metric("사용 모델", result.get("_model_used", "-"))

    st.progress(max(0, min(overall, 100)) / 100)
    st.subheader("핵심 총평")
    st.write(result.get("summary", "-"))


def render_detail_report(result: Dict):
    st.subheader("항목별 점수")
    scores = result.get("scores", {}) or {}
    rows = []
    labels = {
        "visual_impact": "시각적 임팩트",
        "message_clarity": "메시지 명확성",
        "emotional_appeal": "감성 공명도",
        "cta_effectiveness": "CTA 효과성",
        "brand_consistency": "브랜드 일관성",
        "production_quality": "제작 품질",
    }
    for key, label in labels.items():
        item = scores.get(key, {}) or {}
        rows.append({
            "항목": label,
            "점수": item.get("score", 0),
            "평가 근거": item.get("comment", ""),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("플랫폼 적합도")
    pf = result.get("platform_fit", {}) or {}
    rows = []
    for platform, item in pf.items():
        rows.append({
            "플랫폼": platform,
            "점수": (item or {}).get("score", 0),
            "근거": (item or {}).get("reason", ""),
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("타깃 오디언스")
    ta = result.get("target_audience", {}) or {}
    st.write(f"**Primary:** {ta.get('primary', '-')}")
    st.write(f"**Resonance:** {ta.get('resonance', '-')}")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("강점")
        for x in result.get("strengths", []) or []:
            st.write(f"- {x}")
    with c2:
        st.subheader("약점")
        for x in result.get("weaknesses", []) or []:
            st.write(f"- {x}")

    st.subheader("개선 제안")
    imps = result.get("improvements", []) or []
    if imps:
        st.dataframe(pd.DataFrame(imps), use_container_width=True, hide_index=True)

    st.subheader("ROI/성과 예측")
    roi = result.get("roi_prediction", {}) or {}
    st.json(roi, expanded=True)


def batch_summary_dataframe(batch_results: List[Dict]) -> pd.DataFrame:
    rows = []
    for item in batch_results:
        res = item.get("analysis", {}) or {}
        roi = res.get("roi_prediction", {}) or {}
        rows.append({
            "파일명": item.get("filename", ""),
            "종합점수": res.get("overall_score", 0),
            "등급": res.get("grade", ""),
            "사용모델": res.get("_model_used", ""),
            "예상CTR": roi.get("ctr_estimate", ""),
            "전환가능성": roi.get("conversion_potential", ""),
            "바이럴가능성": roi.get("viral_potential", ""),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("종합점수", ascending=False)
