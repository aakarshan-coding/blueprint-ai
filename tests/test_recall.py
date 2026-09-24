from graphrag.eval.recall import recall_at_k


def test_hit_when_correct_id_is_anywhere_in_retrieved_list():
    assert recall_at_k(retrieved=["x", "y", "z"], correct={"y"}) is True


def test_miss_when_correct_id_is_absent():
    assert recall_at_k(retrieved=["x", "y", "z"], correct={"w"}) is False


def test_hit_when_any_correct_id_matches_a_split_section():
    # A question whose source section was split into two chunks (design §5)
    # counts as a hit if either half was retrieved.
    assert recall_at_k(retrieved=["x", "half_b"], correct={"half_a", "half_b"}) is True


def test_empty_retrieved_list_is_a_miss():
    assert recall_at_k(retrieved=[], correct={"y"}) is False
