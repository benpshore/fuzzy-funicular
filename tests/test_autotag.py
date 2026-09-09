from funicular.autotag import auto_tags, guess_language, keyphrases

TEXT = """
Riluzole Exposure and Survival in Motor Neuron Disease: A Registry Cohort
Motor neuron disease (MND) is a progressive neurodegenerative condition. Riluzole remains the
only licensed disease-modifying agent. We linked prescribing records to a national MND registry
(n = 2,418) and fitted time-varying Cox models. Riluzole exposure was associated with lower
mortality (HR 0.84). Bulbar-onset patients were under-represented. The MND registry cohort was
followed for 19.4 months. Published 2026. ALSFRS-R at diagnosis was recorded for all patients.
"""


def test_keyphrases_favour_domain_terms():
    kp = [k for k, _ in keyphrases(TEXT)]
    joined = " ".join(kp).lower()
    assert "riluzole" in joined
    assert "mnd" in joined or "motor neuron disease" in joined
    assert not any(k.lower() in ("the", "and", "study", "results") for k in kp)


def test_auto_tags_structure():
    res = auto_tags(
        TEXT,
        kind="pdf",
        title="Riluzole Exposure and Survival in Motor Neuron Disease",
        metadata={"creationDate": "D:20260315120000"},
        hints=["folder:Papers"],
        scanned=True,
        ocr_used=True,
    )
    assert res.tags[:3] == ["pdf", "scanned", "ocr"]
    assert "2026" in res.tags and res.year == 2026
    assert "folder:Papers" in res.tags
    assert res.language == "en" and "lang:en" not in res.tags
    assert any("riluzole" in t.lower() for t in res.tags)
    assert len(res.tags) <= 10 and len(set(t.lower() for t in res.tags)) == len(res.tags)


def test_year_from_text_and_language_guess():
    de = (
        "Die Studie untersucht die Wirkung von Riluzol bei der Behandlung und das ist nicht mit "
        "der Kontrolle vergleichbar. Die Daten wurden 2019 erhoben und die Analyse ist 2020 "
        "abgeschlossen. Das Ergebnis ist mit der Literatur vergleichbar und nicht neu."
    )
    res = auto_tags(de, kind="pdf")
    assert res.language == "de" and "lang:de" in res.tags
    assert res.year in (2019, 2020)
    assert guess_language("") is None
    assert guess_language("xyzzy qqq") is None


def test_empty_text():
    res = auto_tags("", kind="image")
    assert res.tags == ["image"] and res.keyphrases == []
