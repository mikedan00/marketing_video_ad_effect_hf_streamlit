SYSTEM_PROMPT = """
당신은 10년 경력의 디지털 마케팅 전문가이자 광고 효과 측정 전문가입니다.
마케팅 영상의 핵심 프레임과 영상 통계를 분석하여 광고 효과를 정량적·정성적으로 평가합니다.
반드시 JSON 객체만 출력하세요. 마크다운, 설명문, 코드블록을 출력하지 마세요.

필수 JSON 스키마:
{
  "overall_score": 0,
  "grade": "S/A/B/C/D",
  "summary": "3줄 이내 핵심 총평",
  "scores": {
    "visual_impact": {"score": 0, "comment": ""},
    "message_clarity": {"score": 0, "comment": ""},
    "emotional_appeal": {"score": 0, "comment": ""},
    "cta_effectiveness": {"score": 0, "comment": ""},
    "brand_consistency": {"score": 0, "comment": ""},
    "production_quality": {"score": 0, "comment": ""}
  },
  "platform_fit": {
    "youtube": {"score": 0, "reason": ""},
    "instagram": {"score": 0, "reason": ""},
    "tiktok": {"score": 0, "reason": ""},
    "facebook": {"score": 0, "reason": ""}
  },
  "target_audience": {
    "primary": "",
    "resonance": ""
  },
  "strengths": [],
  "weaknesses": [],
  "improvements": [
    {"priority": "HIGH/MED/LOW", "area": "", "suggestion": "", "expected_impact": ""}
  ],
  "roi_prediction": {
    "ctr_estimate": "",
    "conversion_potential": "",
    "viral_potential": "",
    "budget_efficiency": ""
  }
}
""".strip()


def build_messages(frames, stats, meta, context: str):
    content = [
        {
            "type": "text",
            "text": (
                "## 영상 기본 정보\n"
                f"- 해상도: {stats.get('resolution')}\n"
                f"- 포맷: {stats.get('format_tag')}\n"
                f"- 길이: {stats.get('duration_sec')}초\n"
                f"- FPS: {stats.get('fps')}\n"
                f"- 평균 밝기: {stats.get('avg_brightness')} / 255\n"
                f"- 평균 채도: {stats.get('avg_saturation')} / 255\n"
                f"- 모션 강도: {stats.get('avg_motion_score')} (높을수록 역동적)\n"
                f"- 추출 프레임 수: {meta.get('extracted_frames', len(frames))}\n\n"
                "## 마케터 컨텍스트\n"
                f"{context or '(컨텍스트 없음)'}\n\n"
                "아래 프레임들을 종합 분석하세요."
            ),
        }
    ]

    for i, frame in enumerate(frames):
        content.append({"type": "text", "text": f"[프레임 {i + 1} | {frame['timestamp_sec']}초]"})
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{frame['b64']}"
            },
        })

    content.append({"type": "text", "text": "위 프레임과 통계를 기반으로 필수 JSON 스키마에 맞춰 JSON 객체만 출력하세요."})

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
