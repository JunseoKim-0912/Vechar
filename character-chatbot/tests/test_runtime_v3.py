"""Provider-free invariants for thread transitions and knowledge provenance."""

from types import SimpleNamespace

import pytest

from app.models import MessageRole
from app.schemas import CharacterProfileData, TimelineEvent, TimelineStateChanges
from app.services.character_fidelity import KnowledgeScope, assess_fidelity, build_fidelity_contract
from app.services.conversation_runtime import (
    RUNTIME_VERSION, ChatTurnResult, ConversationRuntimeState, ThreadRecord, ThreadStatus,
    TurnFidelity, TurnProgression, advance_runtime, empty_turn, style_categories, validated_delta,
)
from app.services.knowledge_provenance import KnowledgeProvenance, Provenance
from app.services.response_quality import InvalidConversationTransitionError, select_quality_response


def turn(response: str, **changes) -> ChatTurnResult:
    fields = dict(topic="grete_music", new_development=None, resolved_thread=None,
                  opened_thread=None, action_taken=None, advice_given=None,
                  repeated_point=False)
    fields.update(changes)
    return ChatTurnResult(
        response=response, progression=TurnProgression(**fields),
        fidelity=TurnFidelity(knowledge_scope=KnowledgeScope.UNCERTAIN,
                              persona_preserved=True, assistant_mode=False),
    )


def message(content: str, role: MessageRole, speaker: str | None = None):
    return SimpleNamespace(content=content, role=role, speaker_character_id=speaker, id="m1")


def test_open_active_resolved_and_related_branch_survive_reentry():
    state = ConversationRuntimeState(threads=[ThreadRecord("should_call_grete")],
                                     active_thread_id="should_call_grete")
    waiting = turn("Grete is still beyond the door.")
    advance_runtime(state, waiting, waiting.response, "gregor", "m1", strict_room=True)
    assert state.thread("should_call_grete").status == ThreadStatus.ACTIVE
    action = turn("Gregor calls Grete. Will she answer?", resolved_thread="should_call_grete",
                  opened_thread="does_grete_respond", transition="resolve")
    advance_runtime(state, action, action.response, "gregor", "m2", strict_room=True)
    assert state.thread("should_call_grete").status == ThreadStatus.RESOLVED
    assert state.thread("does_grete_respond").parent_id == "should_call_grete"
    assert state.select_active_thread().id == "does_grete_respond"
    restored = ConversationRuntimeState.from_storage(
        state.to_storage(), [message(action.response, MessageRole.CHARACTER, "gregor")],
    )
    # The snapshot only matches the persisted message ID, never arbitrary newer history.
    assert restored.version == RUNTIME_VERSION
    assert not restored.threads
    persisted = message(action.response, MessageRole.CHARACTER, "gregor")
    persisted.id = "m2"
    restored = ConversationRuntimeState.from_storage(state.to_storage(), [persisted])
    assert restored.active_thread_id == "does_grete_respond"
    assert restored.thread("should_call_grete").status == ThreadStatus.RESOLVED


def test_false_resolution_and_resolved_advice_are_hard_invalid():
    state = ConversationRuntimeState(threads=[ThreadRecord("should_call_grete")],
                                     active_thread_id="should_call_grete")
    vague = turn("Perhaps you should call Grete.", resolved_thread="should_call_grete",
                 transition="resolve")
    assert validated_delta(state, vague)["invalid_transition"]
    with pytest.raises(ValueError):
        advance_runtime(state, vague, vague.response, "doctor", "m1", strict_room=True)
    assert state.last_message_id is None
    state.thread("should_call_grete").status = ThreadStatus.RESOLVED
    state._sync_legacy()
    reused = validated_delta(state, turn("You should call Grete."))
    assert reused["resolved_reuse"] and reused["invalid_transition"]


def test_new_thread_label_cannot_rebrand_an_established_event_as_progress():
    state = ConversationRuntimeState(
        threads=[ThreadRecord("grete_music")], active_thread_id="grete_music",
        established_points=["gregor_calls_grete"],
    )
    repeated = turn("Gregor calls Grete again.", new_development="gregor_calls_grete",
                    new_thread_id="grete_call_again", transition="branch")
    delta = validated_delta(state, repeated)
    assert not delta["development"] and not delta["opened"]
    assert delta["invalid_transition"] and not delta["progression"]


def test_topic_budget_exhausts_and_new_event_reopens():
    state = ConversationRuntimeState(threads=[ThreadRecord("grete_music", budget=4)],
                                     active_thread_id="grete_music")
    stagnant = turn("I still think about the music.")
    for index in range(4):
        advance_runtime(state, stagnant, stagnant.response, "gregor", f"m{index}", strict_room=True)
    assert state.thread("grete_music").status == ThreadStatus.EXHAUSTED
    assert state.select_active_thread() is None
    assert validated_delta(state, stagnant)["exhausted_reuse"]
    event = turn("Grete arrives at the door with her violin.",
                 new_development="grete_arrives", reopened_thread_ids=["grete_music"],
                 transition="reopen")
    assert validated_delta(state, event)["reopened"] == "grete_music"
    advance_runtime(state, event, event.response, "gregor", "m5", strict_room=True)
    assert state.thread("grete_music").status == ThreadStatus.REOPENED
    assert state.thread("grete_music").turns_spent == 0


def test_unrelated_reopen_and_wrong_primary_target_rejected():
    state = ConversationRuntimeState(threads=[ThreadRecord("grete_music", ThreadStatus.EXHAUSTED),
                                              ThreadRecord("family_reaction")],
                                     active_thread_id="family_reaction")
    unrelated = turn("A stranger arrives.", new_development="stranger_arrives",
                     reopened_thread_ids=["grete_music"], transition="reopen")
    assert validated_delta(state, unrelated)["invalid_transition"]
    wrong = turn("The family reacts.", active_thread_id="grete_music")
    assert validated_delta(state, wrong)["wrong_target"]


def test_non_ascii_thread_identifier_cannot_poison_persisted_snapshot():
    state = ConversationRuntimeState()
    invalid = turn("Grete answers from the doorway.", opened_thread="그레테_대답",
                   transition="branch")
    assert validated_delta(state, invalid)["invalid_transition"]
    with pytest.raises(ValueError):
        advance_runtime(state, invalid, invalid.response, "gregor", "m1", strict_room=True)
    assert not state.threads


def test_v1_v2_missing_malformed_future_snapshots_rebuild_without_inventing_threads():
    messages = [message("이 맹추야.", MessageRole.CHARACTER, "jeomsuni")]
    for payload in (None, {"version": 1, "last_message_id": "m1", "open_threads": ["false_thread"]},
                    {"version": 2, "last_message_id": "m1", "open_threads": ["false_thread"]},
                    {"version": 4, "last_message_id": "m1"},
                    {"version": 3, "last_message_id": "m1", "threads": "broken"}):
        state = ConversationRuntimeState.from_storage(payload, messages)
        assert state.version == 3 and not state.threads
        assert state.active_style_categories("jeomsuni") == {"teasing_insult"}


def test_style_category_cooldown_covers_variants_but_not_persona():
    state = ConversationRuntimeState()
    advance_runtime(state, empty_turn("이 맹추야."), "이 맹추야.", "jeomsuni", "m1")
    assert style_categories("이 녀석아, 또 그러니?") == {"teasing_insult"}
    assert "teasing_insult" in state.active_style_categories("jeomsuni")
    outcome = select_quality_response(
        turn("이 녀석아, 또 그러니?"), same_speaker_recent=[], language="Korean",
        allow_repetition=False, retry=lambda _: turn("툴툴거리지만 난 네가 걱정돼."),
        runtime_state=state, speaker="jeomsuni", room=False,
    )
    assert outcome.initial.style_category_cooldown
    assert not outcome.final.style_category_cooldown
    assert outcome.text == "툴툴거리지만 난 네가 걱정돼."
    assert not style_categories(outcome.text)


def test_one_retry_only_and_invalid_room_turn_cannot_commit():
    state = ConversationRuntimeState(threads=[ThreadRecord("should_call_grete", ThreadStatus.RESOLVED)])
    state._sync_legacy()
    attempts = []

    def retry(_):
        attempts.append(1)
        return turn("You should call Grete again.")

    with pytest.raises(InvalidConversationTransitionError):
        select_quality_response(turn("You should call Grete."), same_speaker_recent=[],
                                language="English", allow_repetition=False, retry=retry,
                                runtime_state=state, speaker="doctor", room=True)
    assert len(attempts) == 1
    assert state.last_message_id is None
    assert state.thread("should_call_grete").status == ThreadStatus.RESOLVED


def test_user_callback_is_not_room_hard_blocked():
    state = ConversationRuntimeState(threads=[ThreadRecord("should_call_grete", ThreadStatus.RESOLVED)])
    state._sync_legacy()
    outcome = select_quality_response(
        turn("You could call Grete again."), same_speaker_recent=[], language="English",
        allow_repetition=True, retry=lambda _: pytest.fail("unnecessary retry"),
        runtime_state=state, speaker="gregor", room=False,
    )
    assert not outcome.initial.invalid_thread_transition


def test_prior_self_proof_and_peer_claim_do_not_grant_dsp_expertise():
    doctor = build_fidelity_contract("Country Doctor", CharacterProfileData(
        speech_style="terse and blunt", background_facts=["Treats patients"],
    ))
    question = "Show that convolution of two right-sided signals is right-sided."
    proof = (r"Let \(y[n]=(x*h)[n]\). Every term outside the support vanishes; "
             "therefore the convolution is right-sided for n < 0.")
    prior = message(proof, MessageRole.CHARACTER, "doctor")
    provenance = KnowledgeProvenance.for_user_chat([prior], question)
    result = assess_fidelity(proof, scope=KnowledgeScope.CANONICAL, persona_preserved=True,
                             assistant_mode=False, contract=doctor, request_text=question,
                             provenance=provenance)
    assert result.knowledge_scope_issue and result.prior_self_expertise_blocked
    assert Provenance.PRIOR_SELF_OUTPUT.value in provenance.categories()
    peer = KnowledgeProvenance.for_room([message(proof, MessageRole.CHARACTER, "professor")], "doctor")
    peer_result = assess_fidelity(proof, scope=KnowledgeScope.CANONICAL, persona_preserved=True,
                                  assistant_mode=False, contract=doctor, request_text=question,
                                  provenance=peer)
    assert peer_result.knowledge_scope_issue
    assert Provenance.PEER_CLAIM.value in peer.categories()


def test_explicit_user_teaching_and_canonical_professor_are_allowed():
    doctor = build_fidelity_contract("Country Doctor", CharacterProfileData())
    question = "Given my explanation, why is convolution of right-sided signals right-sided?"
    proof = "By definition, y[n] is a sum of terms. Therefore each term vanishes for n < 0."
    taught = KnowledgeProvenance.for_user_chat(
        [message("Convolution means a sum of shifted products.", MessageRole.USER)], question,
    )
    assert not assess_fidelity(proof, scope=KnowledgeScope.UNCERTAIN, persona_preserved=True,
                               assistant_mode=False, contract=doctor, request_text=question,
                               provenance=taught).knowledge_scope_issue
    professor = build_fidelity_contract("Professor Ada", CharacterProfileData(
        timeline=[TimelineEvent(event_key="current", absolute_year=2020,
                                state_changes=TimelineStateChanges(
                                    occupation="signal-processing professor",
                                    knowledge=["convolution and discrete-time signals"]))],
    ))
    prior = KnowledgeProvenance.for_user_chat(
        [message(proof, MessageRole.CHARACTER, "professor")], question,
    )
    result = assess_fidelity(proof, scope=KnowledgeScope.CANONICAL, persona_preserved=True,
                             assistant_mode=False, contract=professor, request_text=question,
                             provenance=prior)
    assert not result.knowledge_scope_issue and not result.prior_self_expertise_blocked


def test_provenance_and_runtime_prompt_are_compact_without_duplicating_history():
    history = [message("A technical proof " * 100, MessageRole.CHARACTER, "doctor")]
    provenance = KnowledgeProvenance.for_user_chat(history, "What now?")
    state = ConversationRuntimeState(threads=[ThreadRecord("family_reaction")])
    prompt = state.prompt_block("doctor", room=True) + provenance.prompt_block()
    assert len(prompt) < 1800
    assert "A technical proof" not in prompt
    assert "family_reaction" in prompt
    assert provenance.safe_counts()["prior_self_output"] == 1
