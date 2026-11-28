### 건물 인식을 위한 이미지 임베딩 및 참조 벡터 만드는 코드 ###
# 사진 추가·모델 버전 변경 때만 다시 실행

'''
Blob 구조
  curated 컨테이너 / {place_id}/{파일 이름}   (place_images.blob_path = 컨테이너 기준 경로)

접속
  - DB: .env의 DATABASE_URL (place_image_embeddings 쓰기 권한 필요)
  - Blob: Azure 로그인(Entra ID) 계정으로 접속 (연결 문자열·키 없음)
          로컬에서는 먼저 `az login`, 계정에 Storage Blob Data Reader 역할 필요
  - Vision: .env의 AZURE_VISION_ENDPOINT, AZURE_VISION_KEY

  같은 모델 버전의 벡터가 이미 있는 사진은 처음부터 목록에서 빠짐 

실행
  python recognition/build_index.py            # 실제 실행
  python recognition/build_index.py --dry-run  # 대상 사진 목록만 출력 (Vision 호출·DB 저장 없음)
'''

import argparse
import math
import os
import sys
import time

import httpx
import psycopg
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv


## 설정
API_VERSION = "2024-02-01"     # 공식문서 기준, 수정 금지
CALL_INTERVAL = 3              # 3초 간격
MAX_RETRY = 5                  # 429 -> 다시 시도할 횟수
RETRY_WAIT = 10                # 429 -> retry까지 쉬는 시간
EMBEDDING_DIM = 1024           # vector(1024)
DEFAULT_CONTAINER = "curated"  # 참조 사진이 들어 있는 Blob 컨테이너


# 1. 벡터가 아직 없는 참조 사진 불러오기
SELECT_TODO = """
    SELECT pi.image_id, pi.place_id, pi.blob_path
    FROM place_images AS pi
    WHERE pi.is_reference = true
      AND NOT EXISTS (
          SELECT 1 FROM place_image_embeddings AS e
          WHERE e.image_id = pi.image_id AND e.model_version = %s
      )
    ORDER BY pi.place_id, pi.image_id
"""

def load_todo(conn, model_version):
    '''
    → [(image_id, place_id, blob_path), ...]
    같은 모델 버전의 벡터가 이미 저장된 사진은 빠짐
    '''
    with conn.cursor() as cur:
        cur.execute(SELECT_TODO, (model_version,))
        return cur.fetchall()


# 벡터를 길이 1로 정규화
def normalize(vector):
    '''
    각 값을 벡터 길이로 나누기
    '''
    length = math.sqrt(sum(x * x for x in vector))
    if length == 0:
        raise ValueError()
    return [x / length for x in vector]


# pgvector 입력 형식으로 바꾸기: [0.1, 0.2] → "[0.1,0.2]"
def to_pgvector(vector):
    return "[" + ",".join(str(x) for x in vector) + "]"


# 레코드 저장 (DB 저장 방식이 바뀌면 이 함수만 교체)
INSERT_EMBEDDING = """
    INSERT INTO place_image_embeddings (image_id, place_id, model_version, embedding)
    VALUES (%s, %s, %s, %s::vector)
    ON CONFLICT (image_id, model_version) DO NOTHING
"""


def save_embeddings(conn, records):
    '''
    records: [{"image_id", "place_id", "model_version", "embedding"}, ...]
    같은 (image_id, model_version)이 이미 있으면 건너뜀 (ON CONFLICT DO NOTHING)
    호출마다 commit → 중간에 멈춰도 저장한 만큼은 남음
    '''
    with conn.cursor() as cur:
        for r in records:
            cur.execute(INSERT_EMBEDDING, (r["image_id"], r["place_id"], r["model_version"], to_pgvector(r["embedding"])))
    conn.commit()


# DB 접속
def connect_db():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        sys.exit("DATABASE_URL이 .env에 없음")
    return psycopg.connect(database_url)


# Blob 컨테이너 접속 (접속 방식이 바뀌면 이 함수만 교체)
def connect_container():
    '''
    DefaultAzureCredential: 로컬에서는 az login 계정, Azure 안에서는 관리 ID를 자동으로 사용
    '''
    account_url = os.getenv("AZURE_STORAGE_ACCOUNT_URL")
    container_name = os.getenv("AZURE_STORAGE_CONTAINER", DEFAULT_CONTAINER)
    if not account_url:
        sys.exit("AZURE_STORAGE_ACCOUNT_URL이 .env에 없음 (예: https://<스토리지 계정>.blob.core.windows.net)")
    service = BlobServiceClient(account_url=account_url, credential=DefaultAzureCredential())
    return service.get_container_client(container_name)


# 사진 한 장 내려받기 (사진을 어디서 읽는지가 바뀌면 이 함수만 교체)
def read_image(container, blob_path):
    return container.download_blob(blob_path).readall()


# 2. 사진 -> 벡터
def vectorize_image(client, image_bytes, model_version):
    params = {"api-version": API_VERSION, "model-version": model_version}

    for _ in range(MAX_RETRY):
        try:
            response = client.post("/computervision/retrieval:vectorizeImage", params=params, content=image_bytes)
        except httpx.RequestError as e:   # 연결 끊김, 시간 초과
            print(f"   연결 실패 ({type(e).__name__}), {RETRY_WAIT}초 뒤 다시")
            time.sleep(RETRY_WAIT)
            continue

        #너무 많은 요청(429)면 쉬고 다시
        if response.status_code == 429:
            time.sleep(RETRY_WAIT)
            continue
        if response.status_code == 200:
            return response.json()["vector"]
        #그 외 실패
        print(f"   실패 ({response.status_code}): {response.text[:150]}")
        return None

    return None


# Azure AI Vision 클라이언트
def make_vision_client():
    endpoint = os.getenv("AZURE_VISION_ENDPOINT")
    key = os.getenv("AZURE_VISION_KEY")
    if not endpoint or not key:
        sys.exit("AZURE_VISION_ENDPOINT, AZURE_VISION_KEY가 .env에 없음")

    headers = {"Ocp-Apim-Subscription-Key": key, "Content-Type": "application/octet-stream"}
    return httpx.Client(base_url=endpoint.rstrip("/"), headers=headers, timeout=30)


def parse_args():
    parser = argparse.ArgumentParser(description="place_images 참조 사진 → Azure 임베딩 → 정규화 → place_image_embeddings")
    parser.add_argument("--dry-run", action="store_true", help="대상 사진 목록만 출력 (Vision 호출·DB 저장 없음)")
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()
    model_version = os.getenv("VISION_MODEL_VERSION", "2023-04-15")

    #1. 벡터가 아직 없는 참조 사진 불러오기
    conn = connect_db()
    todo = load_todo(conn, model_version)
    print(f"참조 사진: {len(todo)}장 (모델 버전 {model_version})")

    #--dry--run이면 목록만 출력
    if args.dry_run:
        for image_id, place_id, blob_path in todo:
            print(f"  image_id={image_id} place_id={place_id} {blob_path}")
        conn.close()
        return

    container = connect_container()
    client = make_vision_client()
    new_count = 0     # 이번 실행에서 새로 저장한 장수
    fail_count = 0    # 실패해서 건너뛴 장수

    for image_id, place_id, blob_path in todo:
        #2. 사진 -> 벡터
        try:
            image_bytes = read_image(container, blob_path)
        except ResourceNotFoundError:   # place_images에는 있는데 Blob에 사진이 없음
            print(f"[건너뜀] Blob에 사진 없음: image_id={image_id} {blob_path}")
            fail_count += 1
            continue

        time.sleep(CALL_INTERVAL)
        vector = vectorize_image(client, image_bytes, model_version)
        if vector is None or len(vector) != EMBEDDING_DIM:
            print(f"[건너뜀] image_id={image_id} {blob_path}")
            fail_count += 1
            continue

        #3. 정규화해서 한 장씩 저장
        record = {
            "image_id": image_id,
            "place_id": place_id,
            "model_version": model_version,
            "embedding": normalize(vector),
        }
        save_embeddings(conn, [record])
        new_count += 1
        print(f"[새로 {new_count}] image_id={image_id} {blob_path}")

    client.close()
    conn.close()
    print(f"끝: 새로 저장 {new_count}장, 건너뜀 {fail_count}장")


## 실행
if __name__ == "__main__":
    main()
