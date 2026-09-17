import pytest

from backend.family_graph import validate_relationship_pair


def test_accepts_directed_parent_child_relationship():
    validate_relationship_pair("父亲", "女儿")


def test_accepts_directed_extended_family_relationship():
    validate_relationship_pair("叔叔", "侄女")


def test_rejects_invalid_inverse_relationship():
    with pytest.raises(ValueError, match="不匹配"):
        validate_relationship_pair("父亲", "父亲")
