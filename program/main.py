# -*- coding: utf-8 -*-
from fastapi import FastAPI, Request, Form, Depends, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session, relationship
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
# 🚨 database.py 파일에서 필요한 객체 임포트
from database import SessionLocal, engine, Base
from passlib.hash import bcrypt
from starlette.middleware.sessions import SessionMiddleware
from datetime import datetime
from pydantic import BaseModel
import uuid
import os
import requests
import re
from bs4 import BeautifulSoup
import urllib3
import json # Function Calling에 필요
import logging

# --- RAG/OpenAI 관련 임포트 ---
from dotenv import load_dotenv
import openai
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
# 🚨 .env 파일 로드 (OPENAI_API_KEY 로드)
load_dotenv()
# InsecureRequestWarning 경고 비활성화 (크롤링 경고)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
# 🚨 수정: logging.basicConfig 함수 호출을 명확히 하고, 인코딩 지정
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="your-secret-key")

# 정적 파일 (CSS, 이미지, GLB 등) 서빙
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = "static/news"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ----------------------------------------------------
# RAG/GPT 시스템 초기화 (서버 시작 시 한 번만 실행)
# ----------------------------------------------------
openai_client = None
retriever = None
available_tools = {}
tools = []

crop_data = {
    "토마토": {
        "온도": {"day": (25, 27), "night": (18, 20)},
        "습도": (60, 83),
        "광량": (200, 500), # µmol·m2/s
        "pH": (6.0, 6.4),
        "EC": (1.0, 1.5)
    },
    "상추": {
        "온도": (15, 20),
        "습도": (60, 70),
        "광량": (1500, 25000), # Lux
        "pH": (5.8, 6.6),
        "EC": (1.5, 2.0)
    },
    "딸기": {
        "온도": {"day": (17, 24), "night": (8, 15)},
        "습도": (60, 70),
        "광량": (3000, 4000), # Lux
        "pH": (5.5, 6.5),
        "EC": (1.0, 1.5)
    }
}

def get_unit(item, crop):
    """항목과 작물에 따라 적절한 단위를 반환합니다."""
    unit_map = {
        "온도": "°C",
        "습도": "%",
        "pH": "",
        "EC": "dS/m"
    }
    # 광량은 작물별로 단위가 다름
    if item == "광량":
        return "µmol·m²/s" if crop == "토마토" else "Lux"
    return unit_map.get(item, "")



@app.on_event("startup")
def initialize_chatbot_system():
    global openai_client, retriever, available_tools, tools


    # Function Calling 도구 함수 정의
    def navigate_to_page(page_name: str) -> str:
        """
        사용자가 요청한 웹사이트의 특정 페이지로 이동할 수 있는 URL을 제공합니다.
        페이지 이름(page_name)에 따라 미리 정의된 URL을 반환합니다.
        """
        url_map = {
            "실증단지 소개": "http://127.0.0.1:8000/about",
            "온실 3D 모델링": "http://127.0.0.1:8000/datas",
            "실시간 데이터": "http://127.0.0.1:8000/participate",
            "의견 게시판": "http://127.0.0.1:8000/sns",
            "AI 챗봇": "http://127.0.0.1:8000/aichat",
            "입주 공고": "https://innovalley.smartfarmkorea.net/gimje/Demonstration/prv_application",
            "문의하기": "http://127.0.0.1:8000/contact",
            "공지·뉴스": "http://127.0.0.1:8000/news",
            "장비실 3D 뷰어": "http://127.0.0.1:8000/equipment_viewer",
        }
        page_url = url_map.get(page_name)

        if page_url:
            return f"[{page_name}]({page_url})"
        else:
            return f"'{page_name}' 페이지를 찾을 수 없습니다."

    global available_tools
    available_tools = {
        "navigate_to_page": navigate_to_page,
    }

    global tools
    tools.append({
        "type": "function",
        "function": {
            "name": "navigate_to_page",
            "description": "사용자가 특정 페이지로 이동하고 싶다고 요청했을 때, 해당 페이지의 URL을 제공합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "page_name": {
                        "type": "string",
                        "description": "사용자가 요청한 페이지의 이름입니다. (예: '교육 안내', '시설 안내' 등)"
                    }
                },
                "required": ["page_name"]
            }
        }
    })

    try:
        # OpenAI 클라이언트 초기화
        openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # RAG 시스템 초기화 (rag_setup.py 실행 결과물 사용)
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})

        print("✔️ RAG 시스템 초기화 완료.")

    except Exception as e:
        print(f"🚨 챗봇 초기화 오류: OpenAI/Chroma DB 로드 실패. 환경 변수와 rag_setup.py 실행을 확인하세요. 오류: {e}")
        openai_client = None
        retriever = None


# ----------------------------------------------------
# 🚨 챗봇 API 엔드포인트 (FastAPI) 🚨
# ----------------------------------------------------
class ChatRequest(BaseModel):
    message: str


@app.post('/chat')
async def chat(data: ChatRequest):
    if not openai_client or not retriever:
        return JSONResponse(
            {"response": "서버 설정 오류: 챗봇 시스템을 초기화할 수 없습니다."},
            status_code=500
        )

    user_message = data.message
    if not user_message:
        return JSONResponse({"response": "메시지를 입력해 주세요."}, status_code=400)

    try:
        # RAG - 1단계: 검색 (Retrieval)
        docs = retriever.invoke(user_message)
        context = "\n---\n".join([doc.page_content for doc in docs])

        # 🚨 디버깅: 검색된 문서 내용을 콘솔에 출력하여 확인 (배포 시 제거 가능)
        print(f"🔍 검색된 문서 내용 (Top 4):\n{context[:200]}...")

        # --- 웹사이트 소개 정보 추가 ---
        website_intro = """
            김제 스마트팜 혁신밸리 실증단지 웹사이트는 첨단 농업 기술과 관련된 다양한 정보 및 서비스를 제공하는 통합 플랫폼입니다.
            핵심 기능은 다음과 같습니다:

            1.  **실증단지 소개:** 단지의 비전, 시설 정보, 운영 방식 등 종합적인 소개.
            2.  **온실 3D 모델링:** 실증 온실의 구조 및 내부 시설을 시각적으로 탐색할 수 있는 3D 모델링 정보 제공.
            3.  **실시간 데이터:** 실증 재배 환경(온도, 습도, CO2 등)의 실시간 환경 및 생육 데이터 모니터링.
            4.  **의견 게시판 (SNS):** 사용자들이 자유롭게 소통하고 정보를 공유하는 커뮤니티 공간.
            5.  **AI 챗봇:** 사용자 질문에 답변하고 필요한 페이지로 안내하는 AI 기반 상담 서비스.
            """
        # ---

        system_prompt = f"""
                    당신은 '김제 스마트팜 혁신밸리 실증단지' 상담 챗봇입니다.
                    주어진 정보와 도구를 사용하여 사용자 질문에 답변하세요.

                    #중요 지침
                    1. **웹사이트 소개 요청 시**: 사용자가 **'웹사이트', '홈페이지', '소개', '기능'** 등에 대해 질문하면, 아래 **'--- 웹사이트 소개 ---'** 내용을 기반으로 5가지 핵심 기능(소개, 3D 모델링, 데이터, 게시판, 챗봇)을 포함하여 종합적으로 설명해야 합니다.
                    2. **지식 질문**: 다음 '제공된 문서'를 기반으로 답변하세요. 문서에 없으면 모른다고 하세요.
                    3. **기능 요청**: 페이지 이동 등 기능을 요청하면, 제공된 **도구(Tools)**를 사용하여 ID/이름을 추출하고 함수를 호출하세요.
                    4. **가격 및 수치**: 가격, 이용료 등 수치 정보 답변 시, 제공된 문서에 포함된 **단위**를 생략하지 말고 **완전한 문장**으로 답변에 포함해야 합니다.
                    5. **정보의 활용**: 검색된 문서를 그대로 인용하되, 사용자 친화적인 설명 형태로 포장하여 전달해야 합니다.
                    6. **링크 표시**: navigate_to_page 도구의 결과는 [페이지명](URL) 형식으로 반환됩니다. 
                       이를 자연스러운 문장에 포함하세요.
                       예시: "요청하신 [온실 3D 모델링](/datas) 페이지에서 확인하실 수 있습니다."

                    --- 웹사이트 소개 ---
                    {website_intro} # 이 변수는 이미 위에서 5가지 기능을 포함하고 있습니다.
                    --- 제공된 문서 ---
                    {context}
                    ---
                    """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        # 2. 1차 GPT 호출 (도구 사용 여부 결정)
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        response_message = response.choices[0].message

        # 3. Function Calling 실행 로직
        if response_message.tool_calls:
            tool_calls = response_message.tool_calls
            messages.append(response_message)
            tool_outputs = []

            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_tools.get(function_name)

                if function_to_call:
                    function_args = json.loads(tool_call.function.arguments)

                    function_response = function_to_call(
                        page_name=function_args.get("page_name")
                    )

                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    })

            # 2차 GPT 호출 (함수 실행 결과를 기반으로 최종 답변 생성)
            messages.extend(tool_outputs)

            second_response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
            )
            bot_response = second_response.choices[0].message.content

        else:
            # GPT가 일반적인 텍스트 답변을 했을 경우 (RAG 기반 답변)
            bot_response = response_message.content

        return JSONResponse({"response": bot_response})

    except Exception as e:
        print(f"🚨 챗봇 처리 중 오류 발생: {e}")
        return JSONResponse({"response": "죄송합니다. 챗봇 처리 중 내부 오류가 발생했습니다."}, status_code=500)


# -----------------------------------------------------------
# 기존 웹사이트 라우터 및 DB 모델 (여기에 계속 이어집니다.)
# -----------------------------------------------------------

GIMJE_NEWS_URL = "https://innovalley.smartfarmkorea.net/gimje/bbsArticle/list.do?bbsId=notice"
VIEW_BASE_URL = "https://innovalley.smartfarmkorea.net/gimje/bbsArticle/view.do"


# 🚨 모델 정의 (database.py의 Base를 사용)
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True)
    email = Column(String(100), unique=True, index=True)
    password = Column(String(255))
    role = Column(String(50))


class Post(Base):
    __tablename__ = "posts"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    username = Column(String(50), nullable=False)
    role = Column(String(50))
    created_at = Column(DateTime, default=datetime.now)

    comments = relationship("Comment", back_populates="post", cascade="all, delete-orphan")


class Comment(Base):
    __tablename__ = "comments"

    id = Column(Integer, primary_key=True, index=True)
    post_id = Column(Integer, ForeignKey("posts.id"), nullable=False)
    username = Column(String(50), nullable=False)
    role = Column(String(50))
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.now)

    post = relationship("Post", back_populates="comments")


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# -----------------------------------------------------------
# 홈 및 정적 페이지
# -----------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    user = request.session.get("user")
    gimje_news = get_gimje_news()
    return templates.TemplateResponse("profile.html", {"request": request, "user": user,
                                                       "gimje_news": gimje_news,
                                                       })


@app.get("/about", response_class=HTMLResponse)
async def read_about(request: Request):
    user = request.session.get("user")
    return templates.TemplateResponse("about.html", {"request": request, "user": user})


@app.get("/participate", response_class=HTMLResponse)
async def read_participate_form(request: Request):
    """작물 생육 진단 폼 GET"""
    crops = list(crop_data.keys())
    items = list(crop_data["토마토"].keys())  # 토마토 기준으로 항목 로드

    # 초기 폼을 렌더링
    return templates.TemplateResponse("participate.html", {
        "request": request,
        "crops": crops,
        "items": items,
        "result": None,
        "selected_crop": "토마토",  # 초기 선택값 설정
        "selected_item": "온도",
        "selected_value": "",
        "selected_temp_type": "day",
        "user": request.session.get("user")
    })


@app.post("/participate", response_class=HTMLResponse)
async def diagnose_crop(
        request: Request,
        crop: str = Form(...),
        item: str = Form(...),
        user_value: str = Form(None),
        temp_type: str = Form(None)  # 주간/야간 온도 타입
):
    """작물 생육 진단 결과 POST"""
    crops = list(crop_data.keys())
    # 🚨 상추를 선택했을 경우 항목 리스트를 상추 데이터 기준으로 로드 (HTML의 JS 로직과 맞춤)
    items = list(crop_data.get(crop, crop_data["토마토"]).keys())

    result_feedback = ""
    # 🚨 수정: min_v, max_v, unit은 반드시 초기화해야 함
    min_v, max_v, unit = None, None, get_unit(item, crop)

    # 1. 필수 값 검사
    if not all([crop, item, user_value is not None]):
        result_feedback = "🚨 작물, 항목, 값을 모두 선택/입력해 주세요."
        return templates.TemplateResponse('participate.html', {"request": request, "crops": crops, "items": items,
                                                               "result": result_feedback,
                                                               "user": request.session.get("user")})

    # 2. 숫자 변환 검사
    try:
        user_float_value = float(user_value.strip())
    except ValueError:
        result_feedback = "🚨 값은 숫자로 입력해야 합니다."
        return templates.TemplateResponse('participate.html', {"request": request, "crops": crops, "items": items,
                                                               "result": result_feedback,
                                                               "user": request.session.get("user")})

    # 3. 최적 범위 결정 로직

    if item == "온도" and crop in ["토마토", "딸기"]:
        if not temp_type or temp_type not in ['day', 'night']:
            result_feedback = "🚨 토마토/딸기 온도 진단 시 주간/야간을 선택해야 합니다."
            return templates.TemplateResponse('participate.html', {
                "request": request, "crops": crops, "items": items, "result": result_feedback,
                "selected_crop": crop, "selected_item": item, "selected_value": user_value,
                "user": request.session.get("user")
            })

        # 주간/야간 온도 범위 로드
        min_v, max_v = crop_data[crop]["온도"][temp_type]

    else:  # 상추 온도 및 온도 외 항목 처리
        # 해당 항목의 데이터가 crop_data에 정의되어 있는지 확인
        if item in crop_data[crop]:
            # 만약 상추 온도인데 값이 딕셔너리 형태가 아닌 튜플인 경우
            if item == "온도" and type(crop_data[crop]["온도"]) is tuple:
                min_v, max_v = crop_data[crop]["온도"]
            else:
                # 기타 항목 또는 상추 외의 단일 온도 항목
                min_v, max_v = crop_data[crop][item]
        else:
            result_feedback = f"🚨 {crop}에 대한 {item} 데이터가 정의되지 않았습니다."
            return templates.TemplateResponse('participate.html', {"request": request, "crops": crops, "items": items,
                                                                   "result": result_feedback,
                                                                   "user": request.session.get("user")})

    # 🚨 최종 판정 전에 min_v와 max_v가 할당되었는지 확인
    if min_v is None or max_v is None:
        result_feedback = "🚨 내부 오류: 최적 범위를 결정하지 못했습니다."
        return templates.TemplateResponse('participate.html', {"request": request, "crops": crops, "items": items,
                                                               "result": result_feedback,
                                                               "user": request.session.get("user")})

    # 4. 진단 결과 판정 (unit 변수는 이미 get_unit으로 정의됨)
    if user_float_value < min_v:
        result_feedback = f"✅ 진단 결과: **{item}** 값이 최적 범위 **({min_v} {unit} ~ {max_v} {unit})**보다 **낮습니다.** 온도를 높이거나 관수량을 조절하세요."
        color = "red"
    elif user_float_value > max_v:
        result_feedback = f"✅ 진단 결과: **{item}** 값이 최적 범위 **({min_v} {unit} ~ {max_v} {unit})**보다 **높습니다.** 환기를 시키거나 차광을 고려하세요."
        color = "red"
    else:
        result_feedback = f"✅ 진단 결과: **{item}** 값 **{user_float_value} {unit}**은 최적 범위 **({min_v} {unit} ~ {max_v} {unit})** 내에 있습니다. 현재 상태가 좋습니다."
        color = "green"

    # 5. 템플릿 렌더링
    return templates.TemplateResponse('participate.html', {
        "request": request,
        "crops": crops,
        "items": items,
        "result": result_feedback,
        "result_color": color,
        "selected_crop": crop,
        "selected_item": item,
        "selected_value": user_value,
        "selected_temp_type": temp_type,
        "user": request.session.get("user")
    })

# 🚨 수정: /datas 라우터를 3D 전체 뷰어 페이지로 연결
@app.get("/datas", response_class=HTMLResponse)
async def read_datas(request: Request):
    user = request.session.get("user")
    # wholeview.html 템플릿 렌더링
    return templates.TemplateResponse("wholeview.html", {"request": request, "user": user})


@app.get("/contact", response_class=HTMLResponse)
async def contact_form(request: Request):
    return templates.TemplateResponse("contact.html", {"request": request})


@app.get("/aichat", response_class=HTMLResponse)
async def aichat_page(request: Request):
    user = request.session.get("user")
    return templates.TemplateResponse("aichat.html", {"request": request, "user": user})


@app.get("/imdae_sf", response_class=HTMLResponse)
async def contact_form(request: Request):
    user = request.session.get("user")
    return templates.TemplateResponse("imdae_sf.html", {"request": request, "user": user})


@app.post("/contact", response_class=HTMLResponse)
async def submit_contact(request: Request, name: str = Form(...), email: str = Form(...), message: str = Form(...)):
    print(f"문의 도착: {name} | {email} | {message}")
    return templates.TemplateResponse("contact.html", {
        "request": request,
        "submitted": True,
        "name": name
    })


# -----------------------------------------------------------
# 🚨 3D/데이터 시각화 페이지 라우터 추가 🚨
# -----------------------------------------------------------

# @app.get("/data_visualization", response_class=HTMLResponse)
# async def data_visualization(request: Request):
#     """실시간 환경 데이터 시각화 페이지 (greenhouse_data_visualization.html)"""
#     user = request.session.get("user")
#     return templates.TemplateResponse("greenhouse_data_visualization.html", {"request": request, "user": user})


@app.get("/data_visualization1", response_class=HTMLResponse)
async def data_visualization(request: Request):
    """실시간 환경 데이터 시각화 페이지 (greenhouse_data_visualization.html)"""
    user = request.session.get("user")
    return templates.TemplateResponse("greenhouse_data_visualization_1.html", {"request": request, "user": user})


@app.get("/data_visualization2", response_class=HTMLResponse)
async def data_visualization(request: Request):
    """실시간 환경 데이터 시각화 페이지 (greenhouse_data_visualization.html)"""
    user = request.session.get("user")
    return templates.TemplateResponse("greenhouse_data_visualization_2.html", {"request": request, "user": user})


@app.get("/data_visualization5", response_class=HTMLResponse)
async def data_visualization(request: Request):
    """실시간 환경 데이터 시각화 페이지 (greenhouse_data_visualization.html)"""
    user = request.session.get("user")
    return templates.TemplateResponse("greenhouse_data_visualization_5.html", {"request": request, "user": user})


@app.get("/data_visualization6", response_class=HTMLResponse)
async def data_visualization(request: Request):
    """실시간 환경 데이터 시각화 페이지 (greenhouse_data_visualization.html)"""
    user = request.session.get("user")
    return templates.TemplateResponse("greenhouse_data_visualization_6.html", {"request": request, "user": user})


@app.get("/data_visualization8", response_class=HTMLResponse)
async def data_visualization(request: Request):
    """실시간 환경 데이터 시각화 페이지 (greenhouse_data_visualization.html)"""
    user = request.session.get("user")
    return templates.TemplateResponse("greenhouse_data_visualization_8.html", {"request": request, "user": user})


@app.get("/data_visualization11", response_class=HTMLResponse)
async def data_visualization(request: Request):
    """실시간 환경 데이터 시각화 페이지 (greenhouse_data_visualization.html)"""
    user = request.session.get("user")
    return templates.TemplateResponse("greenhouse_data_visualization_11.html", {"request": request, "user": user})


@app.get("/data_visualization12", response_class=HTMLResponse)
async def data_visualization(request: Request):
    """실시간 환경 데이터 시각화 페이지 (greenhouse_data_visualization.html)"""
    user = request.session.get("user")
    return templates.TemplateResponse("greenhouse_data_visualization_12.html", {"request": request, "user": user})


@app.get("/data_visualization14", response_class=HTMLResponse)
async def data_visualization(request: Request):
    """실시간 환경 데이터 시각화 페이지 (greenhouse_data_visualization.html)"""
    user = request.session.get("user")
    return templates.TemplateResponse("greenhouse_data_visualization_14.html", {"request": request, "user": user})


@app.get("/data_visualization16", response_class=HTMLResponse)
async def data_visualization(request: Request):
    """실시간 환경 데이터 시각화 페이지 (greenhouse_data_visualization.html)"""
    user = request.session.get("user")
    return templates.TemplateResponse("greenhouse_data_visualization_16.html", {"request": request, "user": user})


@app.get("/equipment_viewer", response_class=HTMLResponse)
async def equipment_viewer(request: Request):
    """장비실 3D 뷰어 페이지 (equipment_room_viewer_final.html)"""
    user = request.session.get("user")
    return templates.TemplateResponse("equipment_room_viewer_final.html", {"request": request, "user": user})


# -----------------------------------------------------------
# 인증 (Authentication)
# -----------------------------------------------------------

@app.get("/register", response_class=HTMLResponse)
def register_form(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})


@app.post("/register", response_class=HTMLResponse)
def register_user(
        request: Request,
        username: str = Form(...),
        email: str = Form(...),
        password: str = Form(...),
        role: str = Form(...),
        db: Session = Depends(get_db)
):
    existing = db.query(User).filter((User.username == username) | (User.email == email)).first()
    if existing:
        return templates.TemplateResponse(
            "register.html",
            {"request": request, "error": "이미 존재하는 아이디 또는 이메일입니다."}
        )

    hashed_password = bcrypt.hash(password)

    new_user = User(
        username=username,
        password=hashed_password,
        email=email,
        role=role
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return RedirectResponse(url="/login", status_code=302)


@app.get("/login", response_class=HTMLResponse)
def login_form(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/login", response_class=HTMLResponse)
def login_user(request: Request, email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == email).first()

    if not user or not bcrypt.verify(password, user.password):
        return templates.TemplateResponse("login.html", {"request": request, "error": "이메일 또는 비밀번호가 틀렸습니다."})

    request.session["user"] = {
        "username": user.username,
        "email": user.email,
        "role": user.role
    }
    return RedirectResponse(url="/", status_code=303)


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/", status_code=303)


# -----------------------------------------------------------
# SNS (게시판)
# -----------------------------------------------------------

@app.get("/write", response_class=HTMLResponse)
def write_form(request: Request):
    user = request.session.get("user")
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse("write.html", {"request": request, "user": user})


@app.post("/write", response_class=HTMLResponse)
def write_post(
        request: Request,
        title: str = Form(...),
        content: str = Form(...),
        db: Session = Depends(get_db)
):
    user = request.session.get("user")
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    new_post = Post(
        title=title,
        content=content,
        username=user["username"],
        role=user["role"],
        created_at=datetime.now()
    )
    db.add(new_post)
    db.commit()
    db.refresh(new_post)

    return RedirectResponse(url="/sns", status_code=303)


@app.get("/sns", response_class=HTMLResponse)
def board_page(request: Request, db: Session = Depends(get_db)):
    user = request.session.get("user")

    posts = db.query(Post).order_by(Post.created_at.desc()).all()

    posts_data = []
    for post in posts:
        posts_data.append({
            "id": post.id,
            "title": post.title,
            "content": post.content,
            "username": post.username,
            "role": post.role,
            "created_at": post.created_at,
            "comment_count": len(post.comments)
        })

    return templates.TemplateResponse("sns.html", {
        "request": request,
        "user": user,
        "posts": posts_data
    })


@app.get("/post/{post_id}", response_class=HTMLResponse)
def read_post(request: Request, post_id: int, db: Session = Depends(get_db)):
    user = request.session.get("user")

    post = db.query(Post).filter(Post.id == post_id).first()

    if not post:
        return HTMLResponse("게시글을 찾을 수 없습니다.", status_code=404)

    comments = db.query(Comment).filter(Comment.post_id == post_id).order_by(Comment.created_at.asc()).all()

    return templates.TemplateResponse("post_detail.html", {
        "request": request,
        "user": user,
        "post": post,
        "comments": comments
    })


@app.post("/comment/{post_id}", response_class=HTMLResponse)
def write_comment(request: Request, post_id: int, content: str = Form(...), db: Session = Depends(get_db)):
    user = request.session.get("user")
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    new_comment = Comment(
        post_id=post_id,
        username=user["username"],
        role=user["role"],
        content=content,
        created_at=datetime.now()
    )

    db.add(new_comment)
    db.commit()

    return RedirectResponse(url=f"/post/{post_id}", status_code=303)


@app.get("/delete/post/{post_id}")
def delete_post(request: Request, post_id: int, db: Session = Depends(get_db)):
    user = request.session.get("user")
    if not user:
        return RedirectResponse("/login", status_code=303)

    post = db.query(Post).filter(Post.id == post_id).first()

    if not post:
        return HTMLResponse("게시글을 찾을 수 없습니다.", status_code=404)

    if post.username != user["username"]:
        return HTMLResponse("권한이 없습니다.", status_code=403)

    db.delete(post)
    db.commit()

    return RedirectResponse("/sns", status_code=303)


@app.get("/delete/comment/{post_id}/{comment_id}")
def delete_comment(request: Request, post_id: int, comment_id: int, db: Session = Depends(get_db)):
    user = request.session.get("user")
    if not user:
        return RedirectResponse("/login", status_code=303)

    comment = db.query(Comment).filter(Comment.id == comment_id, Comment.post_id == post_id).first()

    if not comment:
        return HTMLResponse("댓글을 찾을 수 없습니다.", status_code=404)

    if comment.username != user["username"]:
        return HTMLResponse("댓글 삭제 권한이 없습니다.", status_code=403)

    db.delete(comment)
    db.commit()

    return RedirectResponse(f"/post/{post_id}", status_code=303)


# -----------------------------------------------------------
# 뉴스 및 크롤링
# -----------------------------------------------------------

def get_gimje_news():
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        response = requests.get(GIMJE_NEWS_URL, headers=headers, timeout=10, verify=False)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        rows = soup.select('.board_list tbody tr')

        gimje_news_list = []

        VIEW_BASE_URL_NOTICE = "https://innovalley.smartfarmkorea.net/gimje/bbsArticle/list.do?bbsId=notice"

        for row in rows:
            cols = row.find_all('td')

            if len(cols) >= 4:
                title_tag = cols[2].find('a')
                if not title_tag:
                    continue

                title = title_tag.text.strip()
                onclick_value = title_tag.get('onclick')
                full_link = "#"

                if onclick_value:
                    match = re.search(r"fn_view\s*\(\s*(\d+)\s*\)", onclick_value)

                    if match:
                        nttSn = match.group(1)
                        full_link = f"{VIEW_BASE_URL_NOTICE}"

                date = cols[3].text.strip()

                gimje_news_list.append({
                    "title": title,
                    "link": full_link,
                    "date": date
                })

        return gimje_news_list

    except requests.exceptions.RequestException as e:
        print(f"웹 크롤링 요청 오류 발생: {e}")
        return []
    except Exception as e:
        print(f"웹 파싱 오류 발생: {e}")
        return []


@app.get("/news", response_class=HTMLResponse)
def news_page(request: Request):
    user = request.session.get("user")

    gimje_news = get_gimje_news()

    return templates.TemplateResponse("news.html", {
        "request": request,
        "gimje_news": gimje_news,
        "user": user
    })


# uvicorn main:app --reload