# 🤖 Code Interpreter Agent

자연어 질문을 Python 코드로 변환해 실행하거나, CSV 파일을 분석하는 LangChain 기반 멀티 에이전트 시스템입니다. Grand Agent가 질문 유형에 따라 Python Agent와 CSV Agent를 동적으로 라우팅합니다.

## 📋 목차
- [프로젝트 개요](#프로젝트-개요)
- [기술 스택](#기술-스택)
- [프로젝트 구조](#프로젝트-구조)
- [주요 기능](#주요-기능)
- [설치 및 실행](#설치-및-실행)
- [사용 예시](#사용-예시)

---

## 프로젝트 개요

세 개의 Agent가 계층 구조로 동작합니다. 최상위 Grand Agent는 사용자의 질문을 분석해 코드 실행이 필요하면 Python Agent로, CSV 데이터 분석이 필요하면 CSV Agent로 위임합니다.

```
Grand Agent (Router)
├── Python Agent  → PythonREPLTool로 코드 생성 및 실행
└── CSV Agent     → Pandas로 episode_info.csv 분석
```

## 기술 스택

| 구분 | 기술 |
|------|------|
| Language | Python 3.11 |
| LLM Orchestration | LangChain, LangChain-Experimental |
| LLM | OpenAI GPT-3.5-turbo |
| Tools | PythonREPLTool, create_csv_agent |
| Data | Pandas, Tabulate |
| Utility | qrcode, python-dotenv |

## 프로젝트 구조

```
code-interpreter/
├── main.py              # Agent 정의 및 실행 진입점
├── episode_info.csv     # Seinfeld 에피소드 데이터
└── Pipfile
```

## 주요 기능

**Python Agent**
- 자연어 지시를 Python 코드로 변환 후 REPL에서 즉시 실행
- 오류 발생 시 자동으로 디버깅 후 재시도
- QR 코드 생성 등 범용 Python 작업 처리

**CSV Agent**
- `episode_info.csv`에 대한 자연어 질의 처리
- Pandas 연산으로 집계·정렬·필터링 수행

**Grand Agent (Router)**
- 질문 의도를 파악해 적합한 하위 Agent로 자동 라우팅
- ReAct(Reasoning + Acting) 프레임워크 기반

## 설치 및 실행

```bash
cd code-interpreter

# 의존성 설치
pipenv install

# 환경 변수 설정
echo "OPENAI_API_KEY=your_key" > .env

# 실행
python main.py
```

## 사용 예시

```python
# CSV 분석 - 에피소드 수가 가장 많은 시즌은?
grand_agent_executor.invoke({
    "input": "which season has the most episodes?"
})

# Python 실행 - QR 코드 15개 생성
grand_agent_executor.invoke({
    "input": "Generate and save in current working directory 15 qrcodes that point to `www.udemy.com/course/langchain`"
})
```
