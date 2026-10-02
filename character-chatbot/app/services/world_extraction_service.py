import json
import re
from ..llm import client, MODELS, extract_text
from ..schemas import WorldProfileData
from ..models import WorldSourceType

WORLD_EXTRACTION_SYSTEM_PROMPT = """당신은 세계관 분석가입니다. 소설의 한 화, 또는 세계관 설명 텍스트에서
그 세계관 자체에 대한 정보만 추출합니다. 등장인물 개개인의 성격/말투가 아니라, 세계관의 배경/규칙/설정을 추출하세요.

인물을 뽑을 때, 같은 인물이 텍스트 안에서 이름/별명/직함 등 여러 다른 표기로 불린다면, 가장 널리 알려진
표기를 name으로, 나머지를 aliases 배열에 담으세요 (예: 본문에서 "엑시아"라고도, "루미에르 엑시아"라고도
불렸다면 name: "엑시아", aliases: ["루미에르 엑시아"]).

반드시 아래 JSON 스키마와 정확히 일치하는 JSON만 출력하세요. 다른 설명, 인사말, 마크다운 코드블록 없이 순수 JSON만 출력합니다.

{
  "world_summary": "2-4문장으로 요약한 세계관 개요",
  "key_facts": ["장소, 규칙, 기술/마법 수준, 사회 구조 등 세계관 자체에 대한 사실들"],
  "timeline_notes": ["시간/역사적 사실. 항목마다 어느 시리즈/시기인지 대괄호로 표시 (예: '[오리진 시리즈] 왕국이 세워지기 전')"],
  "mentioned_characters": [{"name": "대표 이름", "aliases": ["다른 이름/별명/직함"]}]
}

텍스트에서 근거를 찾을 수 없는 필드는 빈 문자열이나 빈 배열로 두세요."""


def _strip_code_fence(text: str) -> str:
    return re.sub(r"^```json\s*|```\s*$", "", text.strip())


def extract_world_profile_from_text(
    raw_text: str,
    source_type: WorldSourceType,
    series_name: str | None,
    episode_number: int | None,
) -> WorldProfileData:
    if source_type == WorldSourceType.NOVEL_EPISODE:
        context_hint = (
            f"이 텍스트는 '{series_name or '이름 없는 시리즈'}'의 {episode_number or '?'}번째 이야기입니다. "
            f"다른 시리즈나 다른 화와 시간대가 다를 수 있으니, timeline_notes에 반드시 이 시리즈명을 표시하세요."
        )
    else:
        context_hint = "이 텍스트는 세계관에 대한 사용자의 직접적인 설명입니다. 등장인물 정보는 없을 수 있습니다."

    response = client.messages.create(
        model=MODELS["extraction"],
        max_tokens=2000,
        system=WORLD_EXTRACTION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"{context_hint}\n\n---\n{raw_text}\n---"}],
    )

    raw = _strip_code_fence(extract_text(response))
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse world extraction JSON: {e}") from e

    return WorldProfileData.model_validate(parsed)