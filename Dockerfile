FROM python:3.10-slim

# 작업 디렉터리
WORKDIR /srv

# 1) core 폴더 전체를 /srv/core 로 복사
COPY core/ core/

# 2) requirements.txt 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3) adapter 코드+의존성
COPY adapters/fastapi/ adapters/fastapi/
RUN pip install --no-cache-dir -r adapters/fastapi/requirements.txt

# # 3) 헬스체크용
# CMD ["python", "-c", "from core.predictor import IDX2LBL; print('model ok →', IDX2LBL)"]

# 4) API 서버 실행
EXPOSE 8000
CMD ["uvicorn", "adapters.fastapi.main:app", "--host", "0.0.0.0", "--port", "8000"]



