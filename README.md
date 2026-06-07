# 마케팅 영상 광고효과 예측 에이전트 - HF Router Streamlit 버전

첨부 Colab 노트북을 VS Code 로컬 실행 및 Streamlit Cloud 배포가 가능하도록 재구성한 버전입니다.

## 주요 변경
- Colab 전용 `google.colab`, `ipywidgets`, `IPython.display` 제거
- Anthropic/OpenAI 직접 호출 제거
- Hugging Face Inference Providers Router의 OpenAI-compatible Chat Completions API 사용
- 사용자가 UI에서 HF 모델 후보를 선택 가능
- 영상 업로드, YouTube URL 분석, A/B 배치 비교 지원
- `.env` 로컬 설정과 Streamlit Secrets 배포 설정 모두 지원
- LLM 실패 시 후보 모델 순차 fallback 및 JSON 파싱 보정

## 로컬 VS Code 실행

```powershell
cd marketing_video_ad_effect_hf_streamlit
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
notepad .env
streamlit run app.py
```

`.env`에서 `HF_TOKEN=hf_새로_발급한_토큰`을 실제 토큰으로 바꾸세요.


## YouTube URL 입력 방식

YouTube 분석 탭에서는 아래 입력을 지원합니다.

- 일반 URL: `https://www.youtube.com/watch?v=dQw4w9WgXcQ`
- Shorts URL: `https://www.youtube.com/shorts/dQw4w9WgXcQ`
- 단축 URL: `https://youtu.be/dQw4w9WgXcQ`
- 영상 ID만 입력: `dQw4w9WgXcQ`

`https://www.youtube.com/watch?v=여기에_유튜브_ID_입력` 같은 예시 placeholder는 자동 차단됩니다.

## Streamlit Cloud 배포

1. GitHub 저장소에 이 폴더 전체를 업로드합니다.
2. Streamlit Cloud에서 New app을 만들고 `app.py`를 entry file로 지정합니다.
3. App Settings > Secrets에 `.streamlit/secrets.toml.example` 내용을 붙여넣고 `HF_TOKEN`만 실제 토큰으로 교체합니다.
4. Deploy를 누릅니다.

## Hugging Face Token 권한

Fine-grained token을 만들 때 `Make calls to Inference Providers` 권한이 필요합니다.

## 주의
- 영상 프레임을 base64 image data URL로 LLM에 전달합니다.
- 선택한 모델/Provider가 VLM(image input)을 지원하지 않으면 실패할 수 있습니다. 이때 후보 모델을 순차 재시도합니다.
- Qwen 후보는 Provider/모델 상태에 따라 이미지 입력을 지원하지 않을 수 있어 fallback 대상으로만 두었습니다.
