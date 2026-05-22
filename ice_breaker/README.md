# 🧊 Ice Breaker

## 📝 프로젝트 개요
LinkedIn과 Twitter(X) 프로필 데이터를 자동으로 수집하고 LLM을 활용하여 대상 인물에 대한 요약, 흥미로운 사실, 아이스브레이킹 문구를 생성하는 AI 기반 웹 애플리케이션입니다.

사용자가 이름을 입력하면 LangChain Agent가 검색 도구를 통해 LinkedIn/Twitter 프로필 URL을 탐색하고 수집된 프로필 데이터를 LLM 파이프라인에 전달하여 대화 시작에 활용할 수 있는 개인화된 정보를 생성합니다.

## 📅 프로젝트 기간
* 2025.08.14 ~

## 💻 기술 스택
* **Language & Framework**: Python 3.10, Flask, LangChain, LangChain-OpenAI, Ollama(Mistral)
* **Library / API**: Tavily, Tweepy, python-dotenv, Scrapin API, Pydantic
  
## 🎯 프로젝트 목적 및 내용

### 1. 소셜 프로필 기반 아이스브레이킹 정보 생성
처음 만나는 사람과의 대화를 쉽게 시작할 수 있도록 LinkedIn과 Twitter(X)에 공개된 프로필 데이터를 수집하고 LLM으로 분석하여 인물 요약, 흥미로운 사실, 아이스브레이킹 문구를 생성합니다.

단순한 프로필 조회가 아니라 상대방의 경력·관심사·활동 내용을 대화 소재로 활용할 수 있도록 정리하는 데 중점을 두었습니다.

### 2. LangChain Agent와 LLM 파이프라인 구현

사용자가 이름을 입력하면 LangChain Agent가 Tavily 검색을 통해 LinkedIn/Twitter(X) URL을 자동 탐색하고 각 플랫폼에서 프로필 데이터를 수집합니다.

수집된 LinkedIn 데이터와 Twitter(X) 게시글은 `PromptTemplate`에 주입되며 LangChain Expression Language(LCEL) 체인 구조를 통해 LLM 응답 생성과 Output Parser 기반 구조화까지 이어지도록 구현했습니다.

#### 주요 기능
* LinkedIn 프로필 자동 검색 및 스크래핑
* Twitter(X) 프로필 URL 및 username 자동 탐색
* Twitter/X 트윗 자동 수집
* Mock 데이터를 활용한 외부 API 대체 실행
* LLM 기반 인물 요약 생성
* 흥미로운 사실 2가지 생성
* 아이스브레이킹 문구 생성
* Flask 웹 UI를 통한 결과 시각화
* 프로필 사진 출력

#### 핵심 구현
```python
# ice_breaker.py
chain = summary_prompt_template | llm | summary_parser

res: Summary = chain.invoke(
    input={
        "information": linkedin_data,
        "twitter_posts": tweets
    }
)
```

## 🛠️ 주요 기능 및 구현 상세

### 1. LinkedIn Lookup Agent
`agents/linkedin_lookup_agent.py`에서 LangChain의 create_react_agent와 Tavily 검색 도구를 조합하여 대상 인물의 LinkedIn 프로필 URL을 자동으로 탐색하도록 구현했습니다.

사용자가 입력한 이름만으로 관련 LinkedIn URL을 찾을 수 있도록 Agent가 검색 도구를 호출하고 최종적으로 프로필 URL을 반환하는 구조입니다.

### 2. Twitter Lookup Agent
`agents/twitter_lookup_agent.py`에서 동일한 ReAct 구조를 활용하여 Twitter(X) 프로필 `URL`과 `username`을 탐색하도록 구현했습니다.

LinkedIn과 Twitter(X)를 분리된 Agent로 구성하여 각 플랫폼별 검색 목적과 반환 데이터를 명확하게 나누었습니다.

### 3. 프로필 데이터 수집 구조
LinkedIn 데이터는 Scrapin API 또는 Mock 데이터를 통해 수집하고 Twitter(X) 데이터는 Tweepy 또는 Mock 데이터를 통해 수집하는 방식으로 구성했습니다.

외부 API 비용과 접근 제한이 있는 상황에서도 전체 LLM 파이프라인을 테스트할 수 있도록 Mock 데이터 기반 실행 흐름을 함께 마련했습니다.

### 4. LangChain 기반 LLM 파이프라인
`ice_breaker.py`에서 수집된 LinkedIn/Twitter 데이터를 `PromptTemplate`에 주입하고, LangChain Expression Language(LCEL) 방식의 체인을 구성했습니다.

```Python
chain = summary_prompt_template | llm | summary_parser
```

이 구조를 통해 프롬프트 생성, LLM 응답 생성, 구조화된 결과 파싱 과정을 하나의 파이프라인으로 연결했습니다.

### 5. 구조화된 Output Parser 적용
LLM 응답을 단순 문자열로 처리하지 않고 Summary 모델을 통해 요약 결과를 구조화했습니다.

이를 통해 인물 요약, 흥미로운 사실, 아이스브레이킹 문구와 같은 항목을 웹 화면에서 일정한 형태로 출력할 수 있도록 구성했습니다.

### 6. Flask 웹 UI 구현
Flask 기반 웹 화면에서 사용자가 이름을 입력하면 분석 결과를 확인할 수 있도록 구성했습니다.

결과 화면에서는 인물 요약, 흥미로운 사실, 아이스브레이킹 문구, 프로필 사진 등을 시각적으로 확인할 수 있도록 했습니다.

## 🖥️ 실행 결과

### 
* 사용자가 이름을 입력하면 해당 인물의 LinkedIn/Twitter(X) 프로필 정보를 기반으로 분석을 수행합니다.
<img src="docs/static/.png" height="400" />

* LLM이 생성한 인물 요약, 흥미로운 사실, 아이스브레이킹 문구를 웹 화면에서 확인할 수 있습니다.
* 프로필 사진이 함께 출력되어 대상 인물 정보를 직관적으로 파악할 수 있습니다.
<img src="docs/static/.png" height="400" />

## ⚠️ 프로젝트 문제점 및 한계

### 1. API 비용 및 접근 제한
LinkedIn, Twitter(X) 등 외부 플랫폼의 API는 접근 제한이 있거나 유료 플랜이 필요한 경우가 많습니다.

이로 인해 실제 운영 환경에서는 각 API 키와 사용 가능한 요금제가 필요하며 현재는 Mock 데이터를 함께 사용하여 전체 기능 흐름을 테스트할 수 있도록 구성했습니다.


### 2. LLM 응답 품질 편차
OpenAI GPT 모델과 Ollama(Mistral) 모델 간 응답 형식과 품질에 차이가 있습니다.

특히 로컬 모델인 Mistral을 사용할 경우 응답 형식이 기대한 JSON 또는 Pydantic 구조와 다르게 생성되어 Output Parser 오류가 발생할 수 있습니다.

### 3. 외부 데이터 신뢰성 문제
검색 도구를 통해 탐색한 LinkedIn/Twitter(X) URL이 항상 정확한 인물의 프로필이라고 보장하기 어렵습니다.

동명이인이나 검색 결과 노이즈가 포함될 수 있기 때문에, 실제 서비스 수준에서는 URL 검증 로직이 추가로 필요합니다.

### 4. Mock 데이터와 실제 API 데이터 간 차이
Mock 데이터는 개발 및 테스트에는 유용하지만, 실제 API 응답 구조와 완전히 동일하지 않을 수 있습니다.

따라서 실제 API 연동 시 응답 필드 구조, 예외 상황, 인증 실패 처리 등을 추가로 점검해야 합니다.


## ✅ 수정 및 보완 사항
* Mock 데이터 외 실제 API 연동 테스트 보완
* Twitter API 응답 파싱 로직 개선
* LLM 응답 형식 오류에 대한 예외 처리 추가
* 실행 화면 및 결과 예시 이미지 추가
