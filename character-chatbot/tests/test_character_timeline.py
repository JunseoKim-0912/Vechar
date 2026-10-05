import unittest
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.database import Base
from app.models import Character, TrainingSource, User, World, WorldSource, WorldSourceType
from app.schemas import (
    CharacterImportRequest, CharacterProfileData, ExtractCharacterFromWorldRequest,
    RelationshipState, TimelineEvent, TimelineStateChanges, WorldProfileData,
)
from app.routers import characters_router, worlds_router
from app.services.character_profile_service import (
    apply_user_correction, get_profile, merge_training_source, set_initial_profile,
)
from app.services.character_timeline import derive_chat_reference, reconcile_timeline
from app.services.chat_prompt_builder import build_chat_input, build_chat_instructions


def event(key, age=None, summary="", **kwargs):
    return TimelineEvent(event_key=key, age=age, summary=summary, **kwargs)


class TimelineReconciliationTests(unittest.TestCase):
    def test_flashback_and_out_of_order_ingestion_do_not_reset_age_or_state(self):
        timeline = reconcile_timeline([], [
            event("current", 21, "Now a guard", state_changes=TimelineStateChanges(
                occupation="guard", relationships=[RelationshipState(name="Mina", status="ally")],
                knowledge=["The gate code"], personality="calm")),
            event("flashback", 16, "At the academy", narrative_role="flashback",
                  state_changes=TimelineStateChanges(occupation="student", personality="impulsive")),
            event("middle", 18, "Joined the scouts", state_changes=TimelineStateChanges(occupation="scout")),
        ])
        self.assertEqual([item.event_key for item in timeline], ["flashback", "middle", "current"])
        reference = derive_chat_reference(timeline)
        self.assertEqual((reference.event_key, reference.age), ("current", 21))
        self.assertEqual(reference.state.occupation, "guard")
        self.assertEqual(reference.state.personality, "calm")
        self.assertEqual(reference.state.relationships[0].status, "ally")

    def test_death_is_boundary_and_postdeath_knowledge_is_excluded(self):
        timeline = reconcile_timeline([], [
            event("after", 26, "Ghost learns the secret", state_changes=TimelineStateChanges(
                knowledge=["Future secret"], occupation="ghost")),
            event("death", 24, "Dies at the bridge", is_death=True),
            event("last", 23, "Works as captain", state_changes=TimelineStateChanges(
                occupation="captain", physical_condition="healthy")),
        ])
        reference = derive_chat_reference(timeline)
        self.assertEqual(reference.phase, "immediately_before_death")
        self.assertEqual(reference.age, 24)
        self.assertEqual(reference.state.occupation, "captain")
        self.assertNotIn("Future secret", reference.state.knowledge)
        self.assertNotIn("Ghost learns", reference.state.knowledge)

    def test_same_age_postdeath_relation_stays_after_death(self):
        timeline = reconcile_timeline([], [
            event("afterlife", 24, "Afterlife revelation", relative_to="death",
                  relative_order="after", state_changes=TimelineStateChanges(knowledge=["afterlife truth"])),
            event("last-day", 24, "Final living day", state_changes=TimelineStateChanges(occupation="guard")),
            event("death", 24, "Died", is_death=True),
        ])
        self.assertEqual(timeline[-1].event_key, "afterlife")
        reference = derive_chat_reference(timeline)
        self.assertEqual(reference.state.occupation, "guard")
        self.assertNotIn("afterlife truth", reference.state.knowledge)

    def test_relative_offsets_preserve_order_without_fabricated_year(self):
        timeline = reconcile_timeline([], [
            event("c", summary="Six months before B", relative_to="b", relative_offset_months=-6,
                  precision="relative"),
            event("b", summary="Three years after A", relative_to="a", relative_offset_months=36,
                  precision="relative"),
            event("a", summary="Begins journey", precision="unknown"),
        ])
        self.assertEqual([item.event_key for item in timeline], ["a", "c", "b"])
        self.assertTrue(all(item.absolute_year is None and item.age is None for item in timeline))
        self.assertEqual(derive_chat_reference(timeline).event_key, "b")

    def test_explicit_dates_and_qualitative_relative_order(self):
        timeline = reconcile_timeline([], [
            event("after", summary="Following winter", relative_to="dated-later",
                  relative_order="after", relative_label="following winter", precision="relative"),
            event("dated-later", summary="Later day", absolute_date="2020-06-02", precision="exact"),
            event("dated-first", summary="Earlier day", absolute_date="2020-06-01", precision="exact"),
        ])
        self.assertEqual([item.event_key for item in timeline], ["dated-first", "dated-later", "after"])
        self.assertIsNone(timeline[-1].absolute_date)

    def test_relationship_and_occupation_evolve_in_event_time(self):
        timeline = reconcile_timeline([], [
            event("spouse", 23, "Married", state_changes=TimelineStateChanges(
                occupation="commander", relationships=[RelationshipState(name="Alice", status="spouse")])),
            event("stranger", 18, "Met Alice", state_changes=TimelineStateChanges(
                occupation="student", relationships=[RelationshipState(name="Alice", status="stranger")])),
            event("friend", 20, "Became friends", state_changes=TimelineStateChanges(
                occupation="soldier", relationships=[RelationshipState(name="Alice", status="friend")])),
        ])
        state = derive_chat_reference(timeline).state
        self.assertEqual(state.occupation, "commander")
        self.assertEqual(state.relationships[0].status, "spouse")

    def test_duplicate_events_track_sources_and_conflicting_age_remains_uncertain(self):
        first = event("met-mina", 20, "Met Mina", precision="exact")
        timeline = reconcile_timeline([], [first], source_id="source-1")
        timeline = reconcile_timeline(timeline, [event("met-mina", 21, "Met Mina", precision="exact")],
                                      source_id="source-2")
        self.assertEqual(len(timeline), 1)
        self.assertEqual(timeline[0].source_ids, ["source-1", "source-2"])
        self.assertIsNone(timeline[0].age)
        self.assertEqual(timeline[0].precision, "conflicting")
        self.assertIn("Conflicting age", timeline[0].temporal_uncertainty)

    def test_state_conflict_does_not_erase_certain_event_time(self):
        first = event("promotion", 21, "Promotion", precision="exact",
                      state_changes=TimelineStateChanges(occupation="captain"))
        second = event("promotion", 21, "Promotion", precision="exact",
                       state_changes=TimelineStateChanges(occupation="general"))
        timeline = reconcile_timeline([first], [second])
        self.assertEqual(timeline[0].precision, "exact")
        self.assertIn("Conflicting occupation", timeline[0].evidence_conflicts[0])
        self.assertEqual(derive_chat_reference(timeline).age, 21)

    def test_noncanonical_event_never_becomes_chat_state(self):
        timeline = reconcile_timeline([], [
            event("current", 21, "Present"),
            event("prophecy", 80, "Will rule the moon", canonicality="noncanonical",
                  state_changes=TimelineStateChanges(occupation="moon ruler")),
        ])
        self.assertEqual(derive_chat_reference(timeline).age, 21)

    def test_undated_events_do_not_claim_ingestion_order_is_current(self):
        timeline = reconcile_timeline([], [
            event("second-listed", summary="An undated event"),
            event("first-listed", summary="Another undated event"),
        ])
        reference = derive_chat_reference(timeline)
        self.assertIsNone(reference.event_key)
        self.assertEqual(reference.status, "unknown")
        prompt = build_chat_instructions("Ari", CharacterProfileData(
            background_facts=["Possibly future fact"], timeline=timeline,
            chat_reference_point=reference,
        ), WorldProfileData(world_summary="Possible future"), correction_prefix="/수정")
        self.assertIn("[TEMPORAL CANON", prompt)
        self.assertNotIn("Possibly future fact", prompt)
        self.assertNotIn("Possible future", prompt)

    def test_prompt_is_bounded_and_excludes_unlived_world_facts(self):
        timeline = reconcile_timeline([], [
            *[event(f"early-{age}", age, f"Earlier event {age}") for age in range(10, 20)],
            event("present", 21, "Knows the gate", state_changes=TimelineStateChanges(
                occupation="guard", knowledge=["gate code"],
                relationships=[RelationshipState(name="Mina", status="ally")])),
            event("death", 24, "Dies", is_death=True),
            event("future", 30, "Discovers secret future spell", state_changes=TimelineStateChanges(
                knowledge=["future spell"])),
        ])
        profile = CharacterProfileData(
            background_facts=["Future secret in legacy array"],
            relationships=["Future enemy"], timeline=timeline,
            chat_reference_point=derive_chat_reference(timeline),
        )
        prompt = build_chat_instructions(
            "Ari", profile, WorldProfileData(world_summary="Future empire falls", key_facts=["future spell"]),
            correction_prefix="/수정", current_message="gate",
        )
        self.assertIn("[TEMPORAL CANON", prompt)
        self.assertIn("Current occupation: guard", prompt)
        self.assertIn("Mina: ally", prompt)
        self.assertNotIn("future spell", prompt)
        self.assertNotIn("Future empire", prompt)
        self.assertNotIn("Future secret", prompt)
        self.assertNotIn("Future enemy", prompt)
        self.assertNotIn("Earlier event 10", prompt)
        self.assertLessEqual(prompt.count("Earlier event"), 3)

    def test_prior_prophecy_is_known_but_future_event_is_not_experienced(self):
        timeline = reconcile_timeline([], [
            event("heard", 18, "Heard a prophecy", state_changes=TimelineStateChanges(
                knowledge=["A prophecy says the kingdom may fall"])),
            event("death", 24, "Died", is_death=True),
            event("collapse", 26, "The kingdom actually collapsed"),
        ])
        profile = CharacterProfileData(timeline=timeline,
                                       chat_reference_point=derive_chat_reference(timeline))
        prompt = build_chat_instructions("Ari", profile, None, correction_prefix="/수정")
        self.assertIn("A prophecy says the kingdom may fall", prompt)
        self.assertNotIn("actually collapsed", prompt)

    def test_earlier_memory_is_reference_not_current_state_or_future_canon(self):
        timeline = reconcile_timeline([], [
            event("met", 18, "Met Alice", state_changes=TimelineStateChanges(
                occupation="student", relationships=[RelationshipState(name="Alice", status="friend")])),
            event("today", 21, "Works as a guard", state_changes=TimelineStateChanges(occupation="guard")),
            event("death", 24, "Dies", is_death=True),
            event("future", 25, "Future secret learned"),
        ])
        profile = CharacterProfileData(timeline=timeline, chat_reference_point=derive_chat_reference(timeline))
        instructions = build_chat_instructions("Ari", profile, None, correction_prefix="/수정")
        inputs = build_chat_input([], "Do you remember Alice?", memories=[
            "At 18 I first met Alice; I was a student.",
            "At 25 I learned the future secret.",
        ])
        self.assertIn("Current occupation: guard", instructions)
        self.assertIn("Treat past states as memories", instructions)
        self.assertNotIn("Future secret learned", instructions)
        self.assertIn("Canon and timeline win", inputs[0]["content"])
        self.assertEqual(inputs[0]["role"], "user")


class TimelineProfileServiceTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False},
                                    poolclass=StaticPool)
        Base.metadata.create_all(self.engine)
        self.db = Session(self.engine)
        user = User(email="timeline@example.invalid", password_hash="unused")
        self.db.add(user)
        self.db.flush()
        character = Character(user_id=user.id, name="Ari")
        self.db.add(character)
        self.db.commit()
        self.user_id, self.character_id = user.id, character.id

    def tearDown(self):
        self.db.close()
        self.engine.dispose()

    def test_prequel_does_not_override_current_persona_and_tracks_source(self):
        set_initial_profile(self.db, self.character_id, CharacterProfileData(
            personality_summary="calm", timeline=[event("current", 21, "Present",
                state_changes=TimelineStateChanges(occupation="guard"))],
        ))
        with patch("app.services.character_profile_service.generate_structured") as synthesis:
            row = merge_training_source(self.db, self.user_id, self.character_id,
                CharacterProfileData(personality_summary="impulsive", timeline=[
                    event("flashback", 16, "Academy years", narrative_role="flashback",
                          state_changes=TimelineStateChanges(occupation="student", personality="impulsive")),
                ]), source_id="source-2")
        synthesis.assert_not_called()
        saved = CharacterProfileData.model_validate(row.data)
        self.assertEqual(saved.chat_reference_point.age, 21)
        self.assertEqual(saved.chat_reference_point.state.occupation, "guard")
        self.assertEqual(saved.chat_reference_point.state.personality, "calm")
        self.assertEqual(saved.personality_summary, "calm")
        self.assertEqual(saved.timeline[0].source_ids, ["source-2"])

    def test_second_source_three_years_later_advances_chat_age(self):
        set_initial_profile(self.db, self.character_id, CharacterProfileData(
            personality_summary="steady", timeline=[event("age-18", 18, "Begins service",
                state_changes=TimelineStateChanges(occupation="student"))],
        ))
        from app.schemas import CharacterSynthesisResult
        with patch("app.services.character_profile_service.generate_structured",
                   return_value=CharacterSynthesisResult(personality_summary="steady", speech_style="direct")):
            row = merge_training_source(self.db, self.user_id, self.character_id,
                CharacterProfileData(personality_summary="steady", timeline=[
                    event("age-21", 21, "Three years later, joins guard", relative_to="age-18",
                          relative_offset_months=36, precision="relative",
                          state_changes=TimelineStateChanges(occupation="guard")),
                ]), source_id="source-2")
        saved = CharacterProfileData.model_validate(row.data)
        self.assertEqual(saved.chat_reference_point.age, 21)
        self.assertEqual(saved.chat_reference_point.state.occupation, "guard")
        self.assertEqual([item.event_key for item in saved.timeline], ["age-18", "age-21"])

    def test_correction_recomputes_current_age_and_relationship(self):
        initial = CharacterProfileData(timeline=[event("current", 21, "Present",
            state_changes=TimelineStateChanges(occupation="guard", relationships=[
                RelationshipState(name="Mina", status="friend")]))])
        set_initial_profile(self.db, self.character_id, initial)
        corrected = CharacterProfileData(timeline=[event("current", 22, "Present",
            state_changes=TimelineStateChanges(occupation="captain", relationships=[
                RelationshipState(name="Mina", status="sister")]))])
        with patch("app.services.character_profile_service.generate_structured", return_value=corrected):
            row = apply_user_correction(self.db, self.user_id, self.character_id,
                                        "현재 나이 22살, 직업 captain, Mina는 sister")
        saved = CharacterProfileData.model_validate(row.data)
        self.assertEqual(saved.chat_reference_point.age, 22)
        self.assertEqual(saved.chat_reference_point.state.occupation, "captain")
        self.assertEqual(saved.chat_reference_point.state.relationships[0].status, "sister")

    def test_import_rederives_reference_from_json_timeline(self):
        imported = CharacterProfileData(timeline=[
            event("later", 21, "Present"), event("earlier", 18, "School"),
        ])
        payload = CharacterImportRequest(export_type="character", schema_version=1,
                                         name="Imported Ari", profile_data=imported)
        with patch.object(characters_router, "check_import_quota"), \
             patch.object(characters_router, "log_import"):
            character = characters_router.import_character(payload, user_id=self.user_id, db=self.db)
        saved = CharacterProfileData.model_validate(get_profile(self.db, character.id).data)
        self.assertEqual(saved.chat_reference_point.age, 21)
        self.assertEqual([item.event_key for item in saved.timeline], ["earlier", "later"])

    def test_world_derived_character_keeps_source_provenance(self):
        world = World(user_id=self.user_id, name="Ari's world")
        self.db.add(world)
        self.db.flush()
        self.db.add(WorldSource(world_id=world.id, source_type=WorldSourceType.DESCRIPTION,
                                raw_text="Ari joins the guard", char_count=19))
        self.db.commit()
        extracted = CharacterProfileData(timeline=[event("guard", 21, "Ari joins the guard")])
        with patch.object(worlds_router, "extract_character_from_world_text", return_value=extracted), \
             patch.object(worlds_router, "get_limits", return_value={"max_characters": 100}):
            character = worlds_router.extract_character_from_world(
                world.id, ExtractCharacterFromWorldRequest(name="Ari"),
                user_id=self.user_id, db=self.db,
            )
        source = self.db.query(TrainingSource).filter(TrainingSource.character_id == character.id).one()
        saved = CharacterProfileData.model_validate(get_profile(self.db, character.id).data)
        self.assertEqual(saved.chat_reference_point.age, 21)
        self.assertEqual(saved.timeline[0].source_ids, [source.id])


if __name__ == "__main__":
    unittest.main()
