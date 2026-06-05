# 🦜 LangChain Projects

LangChain을 활용한 다양한 AI 애플리케이션 실습 모음입니다.

## 📁 프로젝트 목록

| 프로젝트 | 설명 | 주요 기술 |
|----------|------|-----------|
| [ice_breaker](./ice_breaker/) | LinkedIn/Twitter 프로필 기반 아이스브레이킹 정보 생성 | Flask, LangChain Agent, Tavily, Pydantic |
| [documentation_helper](./documentation-helper/) | LangChain 공식 문서 RAG 챗봇 | Streamlit, Pinecone, History-Aware Retriever |
| [code_interpreter](./code-interpreter/) | 자연어 → Python 코드 실행 / CSV 분석 멀티 에이전트 | PythonREPLTool, CSV Agent, ReAct |
| [intro_to_vector_dbs](./intro-to-vector-dbs/) | Pinecone 벡터 DB 기반 RAG 기초 실습 | Pinecone, OpenAI Embeddings, LCEL |
| [vectorstor-in-memory](./vectorstor-in-memory/) | FAISS 인메모리 벡터 DB 기반 PDF RAG | FAISS, PyPDF, OpenAI Embeddings |

## 🛠️ 공통 사전 준비

- Python 3.10+
- pipenv
- OpenAI API Key

각 프로젝트 디렉토리의 README를 참고해 개별 설치 및 실행하세요.

## 📚 학습 흐름

```
1. intro_to_vector_dbs    →  벡터 DB와 RAG 개념 이해
2. vectorstor-in-memory   →  FAISS 인메모리 벡터 DB 실습
3. documentation_helper   →  실전 RAG + 멀티턴 대화
4. code_interpreter       →  멀티 에이전트 & 도구 사용
5. ice_breaker            →  Agent + 외부 API 통합 + Flask UI
```
