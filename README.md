# Meoning-Reosin-ai

워크크 프로젝트의 AI 레포입니다. 건물 인식, 추천, RAG 설명, 혼잡도 예측 코드를 둡니다.


## AI 기능

| 기능 | Azure 서비스 | 코드 위치 (AI 레포) | 코드 위치 (백엔드) | 상태 |
| --- | --- | --- | --- | --- |
| 건물 인식 | Azure AI Vision (멀티모달 임베딩) | `recognition/` | `app/recognitions/client.py` | 진행 중 |
| 추천 | <!-- TODO --> | `recommend/` | <!-- TODO --> | 예정 |
| RAG 설명 | <!-- TODO: Azure OpenAI, AI Search 등 --> | `description/` | <!-- TODO --> | 예정 |
| 혼잡도 ML | <!-- TODO: Azure Machine Learning 등 --> | `congestion/` | <!-- TODO --> | 예정 |

## 폴더 구조

```
Meoning-Reosin-ai/
├── .env.example        # 환경 변수 이름 (값 없음)
├── requirements.txt
└── README.md
```

- `recognition/`, `recommend/`, `description/`, `congestion/`은 각 기능 브랜치에서 만듭니다.
- `common/`은 같은 코드가 두 곳에서 필요해질 때 만듭니다.


## 실행 방법

Python 3.12

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # .env에 실제 값을 채웁니다
```

conda를 쓴다면 앞의 두 줄 대신 아래를 실행합니다.

```bash
conda create -n wcc-ai python=3.12
conda activate wcc-ai
```

`.env`는 커밋하지 않습니다. 새 변수가 생기면 `.env.example`에 이름만 추가합니다.

## 데이터 규칙

- 사진과 데이터는 레포에 올리지 않고 Azure Blob Storage에 둡니다.
- 정제된 사진은 `curated/{place_id}/` 경로에 장소별로 저장합니다.

## 건물 인식 Model Contract

| 항목 | 내용 |
| --- | --- |
| 입력 | 사진 (필수), `lat`/`lng` (선택) |
| 출력 | `Prediction` |
| 모델 버전 | Azure AI Vision `2023-04-15` (`VISION_MODEL_VERSION`) |
| 기준값 | `confidence >= 0.85`일 때 인식 성공 |
| 백엔드 호출 위치 | `app/recognitions/client.py` |

`Prediction` 필드

| 필드 | 타입 | 설명 |
| --- | --- | --- |
| `place_id` | int | 인식된 장소 ID |
| `confidence` | float | 사진 임베딩과 장소 임베딩의 유사도 |

> `confidence`는 확률이 아니라 유사도입니다. 화면에 `%`로 표시하지 않습니다.

모델 버전을 바꿀 때는 이 표와 백엔드 `app/recognitions/client.py`를 같이 수정합니다.
기준값은 백엔드 `app/recognitions/config.py`의 `RECOGNITION_CONFIDENCE_THRESHOLD`와 같은 값으로 맞춥니다.
