import time

from fastapi.testclient import TestClient

from genesis.environments.synthetic_world import SyntheticWorld, SyntheticWorldConfig
from services.api.main import create_app
from services.api.settings import Settings


def _client(tmp_path) -> TestClient:
    settings = Settings(
        db_path=tmp_path / "api.sqlite3",
        event_log_path=tmp_path / "events.jsonl",
        checkpoint_dir=tmp_path / "checkpoints",
        research_log_path=tmp_path / "research_log.md",
    )
    return TestClient(create_app(settings))


def _seed_experiences(client: TestClient, n: int = 120) -> None:
    world = SyntheticWorld(SyntheticWorldConfig(rule="linear", seed=2))
    dataset = world.generate(n, seed=2)
    for record in world.to_observations(dataset, seed=2):
        response = client.post("/experiences", json=record.model_dump(mode="json"))
        assert response.status_code == 201, response.text


def test_health_and_version(tmp_path):
    client = _client(tmp_path)
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/version").json()["version"]


def test_experience_validation_and_listing(tmp_path):
    client = _client(tmp_path)
    bad = client.post("/experiences", json={"features": {"f1": "not-a-number"}})
    assert bad.status_code == 422
    good = client.post(
        "/experiences",
        json={"features": {"f1": 0.2}, "outcome": 1.0},
    )
    assert good.status_code == 201
    listed = client.get("/experiences").json()["experiences"]
    assert len(listed) == 1
    assert listed[0]["features"] == {"f1": 0.2}


def test_learning_predict_checkpoint_restore(tmp_path):
    client = _client(tmp_path)
    _seed_experiences(client)

    step = client.post(
        "/learning/step",
        json={
            "experiment_name": "api-test",
            "feature_names": ["f1", "f2", "f3"],
            "neural": {"epochs": 20, "hidden_sizes": [8], "seed": 1},
        },
    )
    assert step.status_code == 200, step.text
    body = step.json()
    assert body["status"] == "completed"
    assert body["checkpoint_id"]
    assert body["metrics"]["mse"] >= 0.0

    prediction = client.post(
        "/predict", json={"features": {"f1": 0.1, "f2": 0.2, "f3": -0.1}, "actual": 0.5}
    )
    assert prediction.status_code == 200, prediction.text
    assert prediction.json()["error"] is not None

    checkpoints = client.get("/checkpoints").json()["checkpoints"]
    assert len(checkpoints) == 1
    restore = client.post(f"/checkpoints/{body['checkpoint_id']}/restore")
    assert restore.status_code == 200
    assert restore.json()["restored"] is True


def test_predict_without_learner_is_conflict(tmp_path):
    client = _client(tmp_path)
    response = client.post("/predict", json={"features": {"f1": 0.0}})
    assert response.status_code == 409


def test_experiment_validation_rejects_unknown_kind(tmp_path):
    client = _client(tmp_path)
    response = client.post("/experiments", json={"experiment": "E999", "config": {}})
    assert response.status_code == 422


def test_tiny_experiment_runs_in_background(tmp_path):
    client = _client(tmp_path)
    start = client.post(
        "/experiments",
        json={
            "experiment": "E001",
            "config": {
                "seeds": [0],
                "n_train": 80,
                "n_test": 50,
                "sample_sizes": [40],
                "noise_levels": [0.0],
                "run_noise_study": False,
                "neural": {"epochs": 20, "hidden_sizes": [8], "seed": 0},
            },
        },
    )
    assert start.status_code == 202, start.text
    experiment_id = start.json()["id"]

    status = "running"
    for _ in range(100):
        record = client.get(f"/experiments/{experiment_id}").json()
        status = record["status"]
        if status != "running":
            break
        time.sleep(0.2)
    assert status == "completed", record
    assert record["result"]["experiment"] == "E001"
    metrics = client.get(f"/experiments/{experiment_id}/metrics").json()
    assert any(metric["name"].startswith("E001.") for metric in metrics)


def test_e003_persists_revision_history(tmp_path):
    client = _client(tmp_path)
    start = client.post(
        "/experiments",
        json={
            "experiment": "E003",
            "config": {
                "seeds": [0],
                "n_train": 60,
                "n_new": 30,
                "n_new_val": 20,
                "n_test": 40,
                "adaptation_epochs": [5],
                "neural": {"epochs": 10, "hidden_sizes": [8], "seed": 0},
            },
        },
    )
    assert start.status_code == 202, start.text
    experiment_id = start.json()["id"]
    status = "running"
    for _ in range(100):
        record = client.get(f"/experiments/{experiment_id}").json()
        status = record["status"]
        if status != "running":
            break
        time.sleep(0.2)
    assert status == "completed", record
    revisions = client.app.state.genesis.repository.list_revisions(
        experiment_id=experiment_id
    )
    assert len(revisions) == 1
    assert revisions[0]["payload"]["experiment"] == "E003"
    response = client.get(f"/experiments/{experiment_id}/revisions")
    assert response.status_code == 200
    assert response.json()["revisions"][0]["payload"]["experiment"] == "E003"


def test_e003_retention_persists_structured_revisions(tmp_path):
    client = _client(tmp_path)
    start = client.post(
        "/experiments",
        json={
            "experiment": "E003-RETENTION",
            "config": {
                "seeds": [0],
                "n_train": 100,
                "n_new": 40,
                "n_new_val": 30,
                "n_test": 40,
                "adaptation_epochs": [5],
                "neural": {"epochs": 10, "hidden_sizes": [8], "seed": 0},
            },
        },
    )
    assert start.status_code == 202, start.text
    experiment_id = start.json()["id"]
    status = "running"
    record = {}
    for _ in range(100):
        record = client.get(f"/experiments/{experiment_id}").json()
        status = record["status"]
        if status != "running":
            break
        time.sleep(0.2)
    assert status == "completed", record
    response = client.get(f"/experiments/{experiment_id}/revisions")
    assert response.status_code == 200
    payload = response.json()["revisions"][0]["payload"]
    assert payload["experiment"] == "E003-v2"
    assert "curve" in payload
    assert "criterion" in payload
    assert client.get("/revisions").status_code == 200
