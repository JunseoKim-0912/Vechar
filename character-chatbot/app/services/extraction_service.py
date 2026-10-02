import json
import re
from sqlalchemy.orm import Session
from ..llm import generate_text
from ..schemas import CharacterProfileData
from ..models import SourceType

EXTRACTION_SYSTEM_PROMPT_TEMPLATE = """당신은 캐릭터 분석가입니다. 사용자가 제공한 텍스트(단편 소설, 대화 기록, 또는 캐릭터에 대한 직접적인 설명)에서
오직 "{character_name}"이라는 인물 한 명에 대한 정보만 추출합니다.

중요한 규칙:
- 텍스트에 다른 인물이 등장하더라도, 그 인물의 성격/말투/행동/대사를 "{character_name}"의 것으로 착각해서 섞지 마세요.
- 다른 인물에 대한 내용은 "{character_name}"과의 관계를 설명하는 데만 사용하고, relationships 필드에만 반영하세요.
- 만약 텍스트에 "{character_name}"이라는 이름이 전혀 등장하지 않거나 근거를 찾을 수 없다면, 다른 인물의 정보로
  대신 채우지 말고 해당 필드를 빈 문자열/빈 배열로 두세요.

반드시 아래 JSON 스키마와 정확히 일치하는 JSON만 출력하세요. 다른 설명, 인사말, 마크다운 코드블록 없이 순수 JSON만 출력합니다.

{{
  "personality_summary": "2-4문장으로 요약한 성격",
  "speech_style": "말투, 어미, 존댓말/반말, 자주 쓰는 표현 등",
  "background_facts": ["텍스트에서 확인 가능한 배경 사실들"],
  "relationships": ["다른 인물과의 관계 (있다면)"],
  "sample_dialogues": ["말투를 보여주는 원문 그대로의 짧은 대사 몇 개"],
  "do_not_do": []
}}

텍스트에서 근거를 찾을 수 없는 필드는 빈 문자열이나 빈 배열로 두세요."""


def _strip_code_fence(text: str) -> str:
    return re.sub(r"^```json\s*|```\s*$", "", text.strip())


def extract_profile_from_text(
    db: Session, user_id: str, raw_text: str, source_type: SourceType, character_name: str
) -> CharacterProfileData:
    type_hint = {
        SourceType.MANUAL_DESCRIPTION: "이 텍스트는 사용자가 캐릭터에 대해 직접 설명한 내용입니다.",
        SourceType.DIALOGUE: "이 텍스트는 캐릭터가 실제로 말한 대화 기록입니다.",
        SourceType.STORY: "이 텍스트는 캐릭터가 등장하는 단편 소설입니다.",
    }[source_type]

    response_text = generate_text(
        db=db,
        user_id=user_id,
        request_type="character_extraction",
        instructions=EXTRACTION_SYSTEM_PROMPT_TEMPLATE.format(character_name=character_name),
        input_messages=[{"role": "user", "content": f"{type_hint}\n\n---\n{raw_text}\n---"}],
        max_output_tokens=2000,
    )

    raw = _strip_code_fence(response_text)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse extraction JSON: {e}") from e

    return CharacterProfileData.model_validate(parsed)
