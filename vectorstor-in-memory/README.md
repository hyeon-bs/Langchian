# 🗂️ Vector Store In-Memory

FAISS 인메모리 벡터 DB와 OpenAI 임베딩을 활용한 PDF 기반 RAG(Retrieval-Augmented Generation) 실습 프로젝트입니다.

## 📌 프로젝트 개요

PDF 문서를 로드하고 청크로 분할한 뒤, FAISS 벡터 스토어에 저장하여 자연어 질문에 대한 답변을 생성하는 RAG 파이프라인을 구현합니다. ReAct 논문(`2210.03629v3.pdf`)을 예시 문서로 사용합니다.

## 🛠️ 기술 스택

| 항목 | 내용 |
|------|------|
| Language | Python 3.13 |
| LLM | OpenAI (`gpt` / `text-embedding-3-small`) |
| Vector Store | FAISS (CPU) |
| Framework | LangChain, LangChain-OpenAI |
| PDF Loader | PyPDFLoader |
| Package Manager | pipenv |

## 📁 프로젝트 구조

```
vectorstor-in-memory/
├── main.py          # 메인 실행 파일 (RAG 파이프라인)
├── Pipfile          # 의존성 정의
├── Pipfile.lock     # 의존성 잠금 파일
└── faiss_index_react/  # 저장된 FAISS 인덱스 (실행 후 생성)
```

## ⚙️ 설치 및 실행

### 1. 의존성 설치

```bash
pipenv install
pipenv shell
```

### 2. 환경 변수 설정

`.env` 파일을 생성하고 OpenAI API 키를 입력합니다.

```
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. PDF 파일 경로 수정

`main.py` 내 `pdf_path`를 본인의 PDF 파일 경로로 변경합니다.

```python
pdf_path = "/path/to/your/document.pdf"
```

### 4. 실행

```bash
python main.py
```

## 🔄 파이프라인 흐름

```
PDF 로드
  ↓
텍스트 청크 분할 (chunk_size=1000, overlap=30)
  ↓
OpenAI 임베딩 생성 (text-embedding-3-small)
  ↓
FAISS 벡터 스토어 저장 → faiss_index_react/
  ↓
벡터 스토어 로드
  ↓
Retrieval Chain 구성 (langchain-ai/retrieval-qa-chat)
  ↓
질의 응답 생성
```

## 💬 예시 질의

```
"Give me the gist of ReAct in 3 sentences"
```

## 📦 주요 의존성

```
langchain
langchain-openai
langchain-community
langchainhub
faiss-cpu
pypdf
```
