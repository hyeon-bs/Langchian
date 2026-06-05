# 🧊 Ice Breaker

LinkedIn과 Twitter(X) 프로필 데이터를 자동 수집하고, LLM을 활용해 대상 인물의 요약·흥미로운 사실·아이스브레이킹 문구를 생성하는 AI 기반 웹 애플리케이션입니다.

## 📋 목차
- [프로젝트 개요](#프로젝트-개요)
- [기술 스택](#기술-스택)
- [프로젝트 구조](#프로젝트-구조)
- [주요 기능](#주요-기능)
- [설치 및 실행](#설치-및-실행)
- [환경 변수](#환경-변수)
- [한계 및 개선점](#한계-및-개선점)

---

## 프로젝트 개요

사용자가 이름을 입력하면 LangChain Agent가 Tavily 검색을 통해 LinkedIn/Twitter(X) 프로필 URL을 자동 탐색합니다. 수집된 데이터는 LangChain LCEL 파이프라인을 통해 LLM에 전달되어 대화 시작에 활용할 수 있는 개인화된 정보로 변환됩니다.

## 기술 스택

| 구분 | 기술 |
|------|------|
| Language | Python 3.10 |
| Framework | Flask |
| LLM Orchestration | LangChain, LangChain-OpenAI |
| LLM | OpenAI GPT-3.5, Ollama (Mistral) |
| Search | Tavily |
| Social Data | Scrapin API (LinkedIn), Tweepy (Twitter) |
| Config | python-dotenv, Pydantic |

## 프로젝트 구조

```
ice_breaker/
├── agents/
│   ├── linkedin_lookup_agent.py   # LinkedIn URL 탐색 Agent
│   └── twitter_lookup_agent.py    # Twitter username 탐색 Agent
├── third_parties/
│   ├── linkedin.py                # LinkedIn 프로필 스크래핑
│   └── twitter.py                 # Twitter 트윗 수집
├── tools/
│   └── tools.py                   # Tavily 검색 도구
├── templates/
│   └── index.html                 # Flask 웹 UI
├── app.py                         # Flask 애플리케이션 진입점
├── ice_breaker.py                 # LLM 파이프라인 핵심 로직
├── output_parsers.py              # Pydantic 기반 Output Parser
└── Pipfile
```

## 주요 기능

**LangChain ReAct Agent**
- LinkedIn/Twitter(X) 플랫폼별 전용 Agent로 URL 및 username 자동 탐색
- Tavily 검색 도구를 통한 실시간 웹 검색

**LLM 파이프라인 (LCEL)**
```python
chain = summary_prompt_template | llm | summary_parser
```
프롬프트 생성 → LLM 응답 → 구조화 파싱을 단일 파이프라인으로 연결

**구조화된 Output Parser**
- Pydantic `Summary` 모델로 LLM 응답을 구조화
- 인물 요약, 흥미로운 사실 2가지, 아이스브레이킹 문구 항목별 출력

**Mock 데이터 지원**
- 외부 API 비용 없이 전체 파이프라인 테스트 가능

## 설치 및 실행

```bash
# 저장소 클론
git clone <repo-url>
cd ice_breaker

# 의존성 설치
pipenv install

# 환경 변수 설정
cp .env.example .env  # 아래 환경 변수 섹션 참고

# 서버 실행
python app.py
```

브라우저에서 `http://localhost:5000` 접속 후 이름 입력

## 환경 변수

`.env` 파일에 아래 항목을 설정하세요.

```env
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
SCRAPIN_API_KEY=your_scrapin_api_key
TWITTER_BEARER_TOKEN=your_twitter_bearer_token
TWITTER_API_KEY=your_twitter_api_key
TWITTER_API_KEY_SECRET=your_twitter_api_key_secret
TWITTER_ACCESS_TOKEN=your_twitter_access_token
TWITTER_ACCESS_TOKEN_SECRET=your_twitter_access_token_secret
```

> API 키 없이 테스트하려면 `scrape_linkedin_profile(mock=True)`, `scrape_user_tweets(mock=True)` 옵션을 사용하세요.

## 한계 및 개선점

- 외부 플랫폼 API 접근 제한 및 유료 플랜 필요
- OpenAI vs Ollama(Mistral) 간 Output Parser 호환성 차이 존재
- 동명이인으로 인한 URL 정확도 문제 → URL 검증 로직 추가 필요
- Mock 데이터와 실제 API 응답 구조 간 차이 가능성
