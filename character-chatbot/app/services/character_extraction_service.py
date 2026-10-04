from sqlalchemy.orm import Session
from ..llm import generate_structured
from ..schemas import CharacterProfileData

FOCUSED_EXTRACTION_SYSTEM_PROMPT = """당신은 캐릭터 분석가입니다. 여러 개의 텍스트가 주어지고, 그 중 특정 인물 한 명에
대한 정보만 추출해야 합니다. 다른 인물의 정보는 무시하세요. 지목된 인물이 직접 말하거나 행동하거나,
다른 인물이 그 인물에 대해 언급하는 부분만 근거로 삼으세요.

반드시 아래 JSON 스키마와 정확히 일치하는 JSON만 출력하세요. 다른 설명 없이 순수 JSON만 출력합니다.
{
  "personality_summary": "2-4문장으로 요약한 성격",
  "speech_style": "말투, 어미, 존댓말/반말, 자주 쓰는 표현 등",
  "background_facts": ["이 인물에 대한 배경 사실들"],
  "relationships": ["다른 인물과의 관계"],
  "sample_dialogues": ["이 인물이 실제로 한 말 그대로"],
  "do_not_do": []
}

지목된 인물에 대한 근거를 텍스트에서 찾을 수 없는 필드는 빈 문자열이나 빈 배열로 두세요."""

FOCUSED_EXTRACTION_SYSTEM_PROMPT += """

Language policy:
- Understand English, Korean, and mixed English/Korean text, including facts split across languages.
- This creates a new canonical profile, so use the source material's natural dominant language and style while incorporating evidence from every language present.
- Preserve proper names and fictional terms as written. Do not translate names merely to match the output language.
- Keep sample_dialogues in their original quotation language."""


def extract_character_from_world_text(
    db: Session, user_id: str, character_name: str, source_texts: list[str]
) -> CharacterProfileData:
    combined = "\n\n---\n\n".join(source_texts)
    return generate_structured(
        db=db,
        user_id=user_id,
        request_type="world_character_extraction",
        task="analysis",
        instructions=FOCUSED_EXTRACTION_SYSTEM_PROMPT,
        input_messages=[{"role": "user", "content": f"지목된 인물: {character_name}\n\n{combined}"}],
        max_output_tokens=4000,
        response_model=CharacterProfileData,
        input_policy="training",
    )
