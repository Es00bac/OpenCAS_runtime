"""Tests for the utterance parser that handles ambiguous/coded elements."""

from __future__ import annotations

import pytest

from opencas.nlp.utterance_parser import (
    AmbiguityTier,
    ElementType,
    UtteranceParser,
    catalog_elements,
    parse_utterance,
)


class TestScoredLexicalItem:
    def test_musubi_with_value(self):
        text = "'musubi' (value 0.98)"
        result = parse_utterance(text)
        scored = [e for e in result.elements if e.element_type == ElementType.SCORED_LEXICAL]
        assert len(scored) == 1
        elem = scored[0]
        assert elem.raw_text == "musubi' (value 0.98)"
        assert elem.annotations["term"] == "musubi"
        assert elem.annotations["score"] == 0.98
        assert elem.ambiguity_tier == AmbiguityTier.LEXICAL
        assert any("Hawaiian" in r for r in elem.readings)
        assert any("OpenCAS" in r for r in elem.readings)

    def test_musubi_polysemy_readings(self):
        text = "musubi: 0.98"
        result = parse_utterance(text)
        scored = [e for e in result.elements if e.element_type == ElementType.SCORED_LEXICAL]
        assert scored
        assert len(scored[0].readings) >= 3


class TestNonStandardString:
    def test_havjarrod_m_flagged(self):
        text = "havjarrod m"
        result = parse_utterance(text)
        handles = [e for e in result.elements if e.element_type == ElementType.NON_STANDARD_STRING]
        assert len(handles) == 1
        elem = handles[0]
        assert elem.raw_text == "havjarrod m"
        assert elem.ambiguity_tier == AmbiguityTier.STRUCTURAL
        assert elem.annotations["needs_external_key"] is True
        assert "potential_trust_token" in elem.readings

    def test_long_alphanumeric_blob(self):
        text = "identifier abc123def456ghi"
        result = parse_utterance(text)
        handles = [e for e in result.elements if e.element_type == ElementType.NON_STANDARD_STRING]
        # The regex only fires on 12+ char blobs or mixed patterns
        assert any("abc123def456ghi" in h.raw_text for h in handles)


class TestInterpretiveFrame:
    def test_signature_trust_token_hypothesis(self):
        text = "it may be a signature/trust-token rather than a bug"
        result = parse_utterance(text)
        frames = [e for e in result.elements if e.element_type == ElementType.INTERPRETIVE_FRAME]
        assert len(frames) == 1
        elem = frames[0]
        assert "signature" in elem.annotations["hypothesis"].lower()
        assert "bug" in elem.annotations["default"].lower()
        assert elem.ambiguity_tier == AmbiguityTier.PRAGMATIC


class TestMetaInstruction:
    def test_hold_ambiguity_detected(self):
        text = "hold ambiguity; maintain all readings as co-equal"
        result = parse_utterance(text)
        metas = [e for e in result.elements if e.element_type == ElementType.META_INSTRUCTION]
        assert len(metas) >= 1
        assert any("hold" in m.raw_text.lower() for m in metas)
        assert result.meta_constraints

    def test_meta_suspends_resolution(self):
        text = (
            "musubi (0.98), havjarrod m, "
            "it may be a signature rather than a bug, "
            "hold ambiguity"
        )
        result = parse_utterance(text)
        suspended = [e for e in result.elements if e.status == "suspended"]
        assert suspended
        for elem in suspended:
            assert elem.ambiguity_tier in (
                AmbiguityTier.LEXICAL,
                AmbiguityTier.STRUCTURAL,
                AmbiguityTier.PRAGMATIC,
            )


class TestCatalogAPI:
    def test_catalog_structure(self):
        text = "musubi (0.98), havjarrod m, it may be a signature rather than a bug, hold ambiguity"
        catalog = catalog_elements(text)
        assert "source" in catalog
        assert "timestamp" in catalog
        assert "meta_constraints" in catalog
        assert "elements" in catalog
        assert isinstance(catalog["elements"], list)
        tiers = {e["tier"] for e in catalog["elements"]}
        assert "LEXICAL" in tiers
        assert "STRUCTURAL" in tiers
        assert "PRAGMATIC" in tiers
        assert "META" in tiers


class TestParserInternals:
    def test_custom_polysemy_lexicon(self):
        parser = UtteranceParser(polysemy_lexicon={"knot": ["binding", "bond", "problem"]})
        result = parser.parse("knot (0.5)")
        scored = [e for e in result.elements if e.element_type == ElementType.SCORED_LEXICAL]
        assert scored[0].readings == ["binding", "bond", "problem"]

    def test_external_key_resolver(self):
        resolver = lambda handle: handle == "known_key"
        parser = UtteranceParser(external_key_resolver=resolver)
        result = parser.parse("known_key")
        handles = [e for e in result.elements if e.element_type == ElementType.NON_STANDARD_STRING]
        assert handles[0].annotations["resolvable"] is True

    def test_by_tier_filter(self):
        text = "musubi (0.98), havjarrod m, hold ambiguity"
        result = parse_utterance(text)
        lexical = result.by_tier(AmbiguityTier.LEXICAL)
        assert any(e.element_type == ElementType.SCORED_LEXICAL for e in lexical)

    def test_unresolved_filter(self):
        text = "musubi (0.98), hold ambiguity"
        result = parse_utterance(text)
        unresolved = result.unresolved()
        assert all(e.status == "suspended" for e in unresolved)
