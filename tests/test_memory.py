from agent.memory import LongTermMemory


def test_remember_dedup(tmp_path):
    m = LongTermMemory(str(tmp_path / "mem.json"))
    m.remember("用户常住北京")
    m.remember("用户常住北京")
    assert len(m.entries) == 1


def test_recall(tmp_path):
    m = LongTermMemory(str(tmp_path / "mem.json"))
    m.remember("用户常住北京")
    m.remember("用户喜欢靠窗座位")
    recalled = m.recall("我住在北京", top_k=5)
    assert any("北京" in e["content"] for e in recalled)


def test_persistence(tmp_path):
    p = str(tmp_path / "mem.json")
    m = LongTermMemory(p)
    m.remember("偏好靠窗座位")
    m2 = LongTermMemory(p)
    assert any("靠窗" in e["content"] for e in m2.entries)


def test_score():
    assert LongTermMemory._score("北京", "北京") > 0
    assert LongTermMemory._score("北京", "上海") == 0.0


def test_clear(tmp_path):
    p = str(tmp_path / "mem.json")
    m = LongTermMemory(p)
    m.remember("用户常住北京")
    m.clear()
    assert m.entries == []
