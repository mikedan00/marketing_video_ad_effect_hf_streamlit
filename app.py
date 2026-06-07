import json
import os
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import HFSettings
from src.hf_client import chat_json_with_fallback
from src.prompts import build_messages
from src.reporting import batch_summary_dataframe, render_detail_report, render_score_card
from src.video_utils import compute_video_stats, extract_keyframes, save_uploaded_file


st.set_page_config(
    page_title="마케팅 영상 광고효과 예측 에이전트",
    page_icon="🎬",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def _cached_video_stats(video_path: str):
    return compute_video_stats(video_path)


def analyze_video(video_path: str, filename: str, context: str, n_frames: int, settings: HFSettings, selected_model: str):
    stats = compute_video_stats(video_path)
    frames, meta = extract_keyframes(video_path, n_frames=n_frames)
    messages = build_messages(frames, stats, meta, context)

    result = chat_json_with_fallback(
        hf_token=settings.hf_token,
        selected_model=selected_model,
        model_candidates=settings.model_candidates,
        messages=messages,
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
        timeout=(settings.timeout_connect, settings.timeout_read),
        max_retries=settings.max_retries,
    )
    return {
        "filename": filename,
        "stats": stats,
        "frames_meta": meta,
        "analysis": result,
    }


def download_youtube_video(url: str) -> str:
    tmpdir = tempfile.mkdtemp()
    out_template = str(Path(tmpdir) / "yt_video.%(ext)s")
    cmd = [
        "yt-dlp",
        "-f",
        "mp4[height<=720]/best[height<=720]",
        "-o",
        out_template,
        "--no-playlist",
        url,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(res.stderr[-1000:] or "yt-dlp 다운로드 실패")
    files = list(Path(tmpdir).glob("yt_video.*"))
    if not files:
        raise RuntimeError("다운로드된 파일을 찾을 수 없습니다.")
    return str(files[0])


settings = HFSettings.load()

st.title("🎬 마케팅 영상 광고효과 예측 에이전트")
st.caption("영상 업로드 → 핵심 프레임 추출 → Hugging Face Router LLM 분석 → 광고 효과 리포트 생성")

with st.sidebar:
    st.header("LLM 엔진 설정")
    st.write(f"LLM_ENGINE: `{settings.llm_engine}`")
    token_input = st.text_input(
        "HF_TOKEN",
        value=settings.hf_token if settings.hf_token.startswith("hf_") else "",
        type="password",
        help="로컬은 .env, Streamlit Cloud는 Secrets 사용을 권장합니다.",
    )
    if token_input:
        settings.hf_token = token_input

    selected_model = st.selectbox(
        "HF Router 모델 선택",
        settings.model_candidates,
        index=settings.model_candidates.index(settings.router_model) if settings.router_model in settings.model_candidates else 0,
    )

    settings.max_tokens = st.number_input("HF_MAX_TOKENS", 256, 8192, int(settings.max_tokens), step=128)
    settings.temperature = st.slider("HF_TEMPERATURE", 0.0, 1.5, float(settings.temperature), 0.05)
    settings.max_retries = st.number_input("HF_MAX_RETRIES", 1, 10, int(settings.max_retries), step=1)

    st.divider()
    n_frames = st.slider("추출 프레임 수", 4, 16, 8, 1)
    st.info("영상/이미지 입력을 지원하지 않는 모델은 실패할 수 있으며, 이 경우 후보 모델을 순차 재시도합니다.")

tab1, tab2, tab3, tab4 = st.tabs(["단일 영상 분석", "YouTube URL 분석", "A/B 배치 비교", "설정/배포 가이드"])

default_context = (
    "예시: 20~35세 여성 대상 뷰티 브랜드 신제품 런칭 광고. "
    "핵심 메시지: 자연스러운 아름다움. 게재 채널: 인스타그램 릴스, 유튜브 쇼츠."
)

with tab1:
    st.subheader("단일 영상 업로드 분석")
    context = st.text_area("광고 컨텍스트", value=default_context, height=110)
    uploaded = st.file_uploader("분석할 영상 파일 업로드", type=["mp4", "mov", "avi", "mkv", "webm"])

    if uploaded and st.button("영상 분석 실행", type="primary"):
        suffix = Path(uploaded.name).suffix or ".mp4"
        video_path = save_uploaded_file(uploaded, suffix=suffix)

        with st.spinner("영상 통계 계산, 프레임 추출, HF LLM 분석 중..."):
            try:
                result_pack = analyze_video(video_path, uploaded.name, context, n_frames, settings, selected_model)
                st.session_state["last_result"] = result_pack
            except Exception as exc:
                st.error(f"분석 실패: {exc}")
                st.stop()

        st.success("분석 완료")
        st.video(video_path)
        render_score_card(result_pack["analysis"])
        render_detail_report(result_pack["analysis"])

        st.download_button(
            "JSON 결과 다운로드",
            data=json.dumps(result_pack, ensure_ascii=False, indent=2),
            file_name=f"ad_analysis_{Path(uploaded.name).stem}.json",
            mime="application/json",
        )

with tab2:
    st.subheader("YouTube URL 분석")
    st.warning("YouTube 다운로드는 영상 소유권/이용약관/저작권을 확인한 뒤 사용하세요.")
    url = st.text_input("YouTube URL")
    yt_context = st.text_area("광고 컨텍스트", value=default_context, height=110, key="yt_context")

    if url and st.button("YouTube 영상 다운로드 및 분석", type="primary"):
        with st.spinner("YouTube 영상 다운로드 중..."):
            try:
                video_path = download_youtube_video(url)
            except Exception as exc:
                st.error(f"YouTube 다운로드 실패: {exc}")
                st.stop()

        with st.spinner("영상 분석 중..."):
            try:
                result_pack = analyze_video(video_path, Path(video_path).name, yt_context, n_frames, settings, selected_model)
            except Exception as exc:
                st.error(f"분석 실패: {exc}")
                st.stop()

        st.success("분석 완료")
        st.video(video_path)
        render_score_card(result_pack["analysis"])
        render_detail_report(result_pack["analysis"])

        st.download_button(
            "JSON 결과 다운로드",
            data=json.dumps(result_pack, ensure_ascii=False, indent=2),
            file_name="ad_analysis_youtube.json",
            mime="application/json",
        )

with tab3:
    st.subheader("여러 영상 A/B 배치 비교")
    batch_context = st.text_area(
        "배치 분석 컨텍스트",
        value="동일 캠페인의 A/B 테스트용 광고 소재들. 타겟: 25~40세 직장인.",
        height=100,
    )
    uploads = st.file_uploader(
        "비교할 영상 파일 여러 개 업로드",
        type=["mp4", "mov", "avi", "mkv", "webm"],
        accept_multiple_files=True,
    )

    if uploads and st.button("배치 분석 실행", type="primary"):
        batch_results = []
        progress = st.progress(0)
        for i, uploaded_file in enumerate(uploads, start=1):
            st.write(f"분석 중: {uploaded_file.name}")
            suffix = Path(uploaded_file.name).suffix or ".mp4"
            video_path = save_uploaded_file(uploaded_file, suffix=suffix)
            try:
                result_pack = analyze_video(video_path, uploaded_file.name, batch_context, max(4, min(n_frames, 8)), settings, selected_model)
                batch_results.append(result_pack)
            except Exception as exc:
                st.error(f"{uploaded_file.name} 분석 실패: {exc}")
            progress.progress(i / len(uploads))

        if batch_results:
            df = batch_summary_dataframe(batch_results)
            st.subheader("A/B 비교 요약")
            st.dataframe(df, use_container_width=True, hide_index=True)

            best = batch_results[0]
            best_score = -1
            for item in batch_results:
                score = int(item["analysis"].get("overall_score", 0) or 0)
                if score > best_score:
                    best = item
                    best_score = score

            st.subheader("최고 점수 소재 상세")
            st.write(f"**{best['filename']}**")
            render_score_card(best["analysis"])
            render_detail_report(best["analysis"])

            st.download_button(
                "배치 JSON 결과 다운로드",
                data=json.dumps(batch_results, ensure_ascii=False, indent=2),
                file_name="ad_analysis_batch_results.json",
                mime="application/json",
            )

with tab4:
    st.subheader("로컬/배포 설정")
    st.markdown(
        """
### VS Code 로컬 실행
```powershell
python -m venv .venv
.\\.venv\\Scripts\\activate
pip install -r requirements.txt
copy .env.example .env
notepad .env
streamlit run app.py
```

### Streamlit Cloud Secrets
```toml
LLM_ENGINE = "hf_api"
HF_TOKEN = "hf_새로_발급한_토큰"
HF_ROUTER_MODEL = "google/gemma-4-26B-A4B-it:deepinfra"
HF_MODEL_CANDIDATES = "google/gemma-4-26B-A4B-it:deepinfra,google/gemma-4-26B-A4B-it:novita,google/gemma-4-31B-it:deepinfra,google/gemma-4-31B-it:together,Qwen/Qwen3.5-9B:together,Qwen/Qwen2.5-7B-Instruct:together"
HF_MAX_TOKENS = 1400
HF_TEMPERATURE = 0.2
HF_TIMEOUT_CONNECT = 10
HF_TIMEOUT_READ = 120
HF_MAX_RETRIES = 3
```
"""
    )
