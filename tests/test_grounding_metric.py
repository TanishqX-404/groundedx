from groundedx.faithfulness.grounding_metric import grounding_check


def test_grounding_requires_two_shared_content_terms():
    evidence = {"c1": "latency increased after DNS resolution failure"}
    assert grounding_check("latency increased after DNS failure", ["c1"], evidence)
    assert not grounding_check("the system is unhealthy", ["c1"], evidence)


def test_grounding_requires_citation_and_retrieved_membership():
    evidence = {"c1": "CPU throttling affected request latency"}
    assert not grounding_check("CPU throttling affected latency", [], evidence)
    assert not grounding_check("CPU throttling affected latency", ["missing"], evidence)
    assert grounding_check("CPU throttling affected latency", ["c1"], evidence)
