"""Regression tests for the deterministic emergency detector (audit: 14/15 realistic phrasings were missed)."""
import pytest

from app.agents.emergency import collect_texts
from app.services.clinical_rules import detect_emergency, detect_emergency_detailed

MUST_FLAG = [
    "I can't breathe",
    "I am short of breath and my muscles are too weak to cough",
    "I passed out and woke up on the floor",
    "my wife collapsed and is not responding",
    "my speech is slurring and my mouth is crooked",
    "the left side of my face has dropped",
    "I can't move my right arm or leg",
    "sudden severe thunderclap headache, worst pain ever",
    "I lost vision in one eye suddenly",
    "the seizure has gone on for more than 10 minutes",
    "he had three seizures in a row without waking up",
    "my words are coming out garbled",
    "I can’t speak",          # curly apostrophe
    "I CAN'T SPEAK",
    "can't   speak",              # irregular whitespace
    "weakness spreading upward from my feet",
    "status epilepticus, seizure not stopping",
    "face drooping",              # original literal phrase must not regress
    "worst headache of my life",
]


@pytest.mark.parametrize("text", MUST_FLAG)
def test_realistic_emergency_phrasings_are_flagged(text):
    flagged, matched = detect_emergency([text])
    assert flagged, f"missed emergency: {text!r}"
    assert matched


MUST_NOT_FLAG = [
    "mild tingling in left hand, otherwise feeling fine",
    "I exercise for 30 minutes most days and feel fine",
    "I took a walk, breathing is normal",
    "slight headache after a long day",
]


@pytest.mark.parametrize("text", MUST_NOT_FLAG)
def test_benign_text_not_flagged(text):
    assert detect_emergency([text])[0] is False


@pytest.mark.parametrize("text", [
    "no slurred speech, no face drooping",
    "denies arm weakness",
    "no seizure over 5 minutes",
    "without any trouble breathing",
])
def test_explicitly_negated_symptoms_are_not_flagged_but_are_reported(text):
    d = detect_emergency_detailed([text])
    assert d["is_emergency"] is False
    assert d["negated"], "a suppressed match must be reported, never silently dropped"


def test_negation_does_not_leak_across_clauses():
    # 'no headache' must not suppress 'cannot speak' in the next clause.
    assert detect_emergency(["no headache but I cannot speak"])[0] is True
    assert detect_emergency(["no pain. My face is drooping"])[0] is True


def test_historical_mention_is_still_flagged_conservatively():
    # We deliberately do NOT try to detect 'history of'; over-triage is safer than a missed stroke.
    assert detect_emergency(["my father had a stroke with slurred speech years ago"])[0] is True


def test_none_and_empty_inputs_do_not_raise():
    assert detect_emergency([]) == (False, [])
    assert detect_emergency([None, "", "   "])[0] is False  # type: ignore[list-item]


def test_collect_texts_scans_name_and_free_text_and_caregiver_notes():
    # Previously `free_text or name` meant a benign free_text hid an alarming name.
    payload = {"symptoms": [{"name": "face drooping", "free_text": "feeling fine really"}], "caregiver_notes": "he cannot speak"}
    texts = collect_texts(payload)
    assert "face drooping" in texts and "he cannot speak" in texts
    assert detect_emergency(texts)[0] is True
