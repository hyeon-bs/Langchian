# 📚 Documentation Helper

LangChain 공식 문서를 기반으로 질문에 답변하는 RAG(Retrieval-Augmented Generation) 챗봇입니다. Pinecone 벡터 스토어와 대화 히스토리 인식 검색을 결합해 문맥을 유지하며 정확한 답변을 제공합니다.

## 📋 목차
- [프로젝트 개요](#프로젝트-개요)
- [기술 스택](#기술-스택)
- [프로젝트 구조](#프로젝트-구조)
- [주요 기능](#주요-기능)
- [설치 및 실행](#설치-및-실행)
- [환경 변수](#환경-변수)

---

## 프로젝트 개요

LangChain 공식 문서를 크롤링·임베딩하여 Pinecone에 저장하고, 사용자 질문에 대해 관련 문서를 검색한 뒤 GPT 모델로 답변을 생성합니다. 멀티턴 대화에서도 이전 컨텍스트를 반영하는 `history_aware_retriever`를 적용했습니다.


![Logo](https://github.com/emarco177/documentation-helper/blob/main/static/banner.gif)


## 기술 스택

| 구분 | 기술 |
|------|------|
| Language | Python 3.11 |
| Framework | Streamlit |
| LLM Orchestration | LangChain |
| LLM | OpenAI GPT-4o-mini |
| Embedding | OpenAI text-embedding-3-small |
| Vector Store | Pinecone |
| Web Scraping | ReadTheDocs Loader, FireCrawl |
| UI | Streamlit, streamlit-chat |

## 프로젝트 구조

```
documentation-helper/
├── backend/
│   ├── __init__.py
│   └── core.py          # RAG 체인 핵심 로직
├── ingestion.py         # 문서 로딩 및 Pinecone 업로드
├── main.py              # Streamlit 웹 UI
└── Pipfile
```

## 주요 기능

**RAG 파이프라인**
```
사용자 질문
  → History-Aware Retriever (이전 대화 반영하여 질문 재구성)
  → Pinecone 벡터 검색
  → Stuff Documents Chain (문서 결합)
  → LLM 답변 생성
```

**History-Aware Retrieval**
- `create_history_aware_retriever`로 이전 대화를 참조해 질문을 재구성
- 멀티턴 대화에서도 정확한 문서 검색 가능

**문서 수집 방식 (두 가지)**
- `ReadTheDocsLoader`: LangChain API 문서 HTML 일괄 다운로드 후 파싱
- `FireCrawlLoader`: 실시간 웹 크롤링으로 최신 문서 수집

**Streamlit 챗 UI**
- 사용자/어시스턴트 메시지 구분 렌더링
- 스피너로 응답 대기 시각화

## 설치 및 실행

```bash
# 저장소 클론
cd documentation-helper

# 의존성 설치
pipenv install

# (선택) LangChain 문서 다운로드
mkdir langchain-docs
wget -r -A.html -P langchain-docs https://api.python.langchain.com/en/latest

# 문서 임베딩 및 Pinecone 업로드
python ingestion.py

# Streamlit 앱 실행
streamlit run main.py
```

## 환경 변수

`.env` 파일에 아래 항목을 설정하세요.

```env
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=langchain-doc-index2
FIRECRAWL_API_KEY=your_firecrawl_api_key  # FireCrawl 사용 시
```

> Pinecone 인덱스는 `text-embedding-3-small` 기준 **1536차원**으로 생성하세요.
