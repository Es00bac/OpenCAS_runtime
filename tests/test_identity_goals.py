import pytest
from pathlib import Path
from opencas.identity import IdentityManager, IdentityStore

def test_add_inferred_goal_with_provenance(tmp_path: Path):
    store = IdentityStore(tmp_path)
    mgr = IdentityManager(store)
    mgr.load()
    
    # Test adding first goal
    mgr.add_inferred_goal("test goal 1", provenance="daydream")
    assert len(mgr.user_model.inferred_goals) == 1
    assert mgr.user_model.inferred_goals[0]["text"] == "test goal 1"
    assert mgr.user_model.inferred_goals[0]["provenance"] == "daydream"

def test_inferred_goal_cap_and_eviction(tmp_path: Path):
    store = IdentityStore(tmp_path)
    mgr = IdentityManager(store)
    mgr.load()
    
    # Fill up to cap of 8
    for i in range(8):
        mgr.add_inferred_goal(f"goal {i}", provenance="daydream")
    
    assert len(mgr.user_model.inferred_goals) == 8
    
    # Add 9th goal, should evict "goal 0"
    mgr.add_inferred_goal("goal 8", provenance="daydream")
    assert len(mgr.user_model.inferred_goals) == 8
    # Assuming eviction of oldest daydream goal (index 0 if it's daydream)
    assert mgr.user_model.inferred_goals[0]["text"] == "goal 1"
    assert mgr.user_model.inferred_goals[-1]["text"] == "goal 8"

def test_inferred_goal_preserves_operator_goals(tmp_path: Path):
    store = IdentityStore(tmp_path)
    mgr = IdentityManager(store)
    mgr.load()
    
    # Add an operator goal
    mgr.add_inferred_goal("operator goal", provenance="operator")
    
    # Fill the rest with daydream goals
    for i in range(7):
        mgr.add_inferred_goal(f"daydream {i}", provenance="daydream")
        
    assert len(mgr.user_model.inferred_goals) == 8
    
    # Add one more daydream goal. 
    # It should NOT evict "operator goal", but instead evict the oldest daydream goal ("daydream 0")
    mgr.add_inferred_goal("new daydream", provenance="daydream")
    
    assert len(mgr.user_model.inferred_goals) == 8
    texts = [g["text"] for g in mgr.user_model.inferred_goals]
    assert "operator goal" in texts
    assert "daydream 0" not in texts
    assert "new daydream" in texts

def test_inferred_goal_migration(tmp_path: Path):
    # This one tests that if we load a user.json with List[str], it migrates
    import json
    user_file = tmp_path / "user.json"
    user_data = {
        "inferred_goals": ["legacy goal 1", "legacy goal 2"]
    }
    with open(user_file, "w") as f:
        json.dump(user_data, f)
        
    store = IdentityStore(tmp_path)
    mgr = IdentityManager(store)
    mgr.load()
    
    assert len(mgr.user_model.inferred_goals) == 2
    assert isinstance(mgr.user_model.inferred_goals[0], dict)
    assert mgr.user_model.inferred_goals[0]["text"] == "legacy goal 1"
    assert mgr.user_model.inferred_goals[0]["provenance"] == "legacy"

def test_mixed_inferred_goal_migration(tmp_path: Path):
    import json
    user_file = tmp_path / "user.json"
    user_data = {
        "inferred_goals": [
            {"text": "structured goal", "provenance": "daydream"},
            "legacy goal",
        ]
    }
    with open(user_file, "w") as f:
        json.dump(user_data, f)

    store = IdentityStore(tmp_path)
    mgr = IdentityManager(store)
    mgr.load()

    assert mgr.user_model.inferred_goals == [
        {"text": "structured goal", "provenance": "daydream"},
        {"text": "legacy goal", "provenance": "legacy"},
    ]
