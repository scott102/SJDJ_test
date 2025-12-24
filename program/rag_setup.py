import os
# 🚨 .env 파일을 읽기 위한 라이브러리 추가
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter

# 🚨 .env 파일 로드 (이 코드가 있어야 API 키를 인식합니다)
load_dotenv()

# API 키 확인 (디버깅용 - 실제 키가 출력되면 안 되므로 일부만 확인하거나 존재 여부만 체크)
if not os.getenv("OPENAI_API_KEY"):
    print("❌ 오류: OPENAI_API_KEY가 환경 변수에 설정되지 않았습니다. .env 파일을 확인해주세요.")
    exit(1) # 스크립트 중단

# 1. 문서 로드
print("1. 문서 로드 중...")
try:
    loader = TextLoader("website_content.txt", encoding="utf-8")
    documents = loader.load()
except FileNotFoundError:
    print("❌ 오류: 'website_content.txt' 파일을 찾을 수 없습니다.")
    exit(1)

# 2. 문서 분할 (청크 나누기)
# 긴 문서를 AI가 처리하기 쉬운 작은 조각으로 나눕니다.
print("2. 문서 분할 중...")
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# 3. 임베딩 모델 준비 (GPT-4o-mini와 연동하기 위해 OpenAI 모델 사용)
print("3. 임베딩 모델 준비 중...")
try:
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
except Exception as e:
    print(f"❌ 임베딩 모델 초기화 실패: {e}")
    exit(1)

# 4. 벡터 데이터베이스에 저장 (ChromaDB 사용)
# 저장된 경로는 './chroma_db' 폴더입니다.
print("4. ChromaDB에 벡터 저장 중...")
try:
    db = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")
    print(f"✔️ RAG 색인 완료. 총 {len(texts)}개의 문서 조각이 저장되었습니다.")
except Exception as e:
    print(f"❌ 데이터베이스 저장 실패: {e}")