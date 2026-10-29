"""Genesis FastAPI application.

The API is a thin, validating facade over the research engine. It does not
contain learning logic: it loads experiences, invokes :mod:`genesis.learning`
and :mod:`genesis.evaluation`, persists results, and reports them faithfully.
"""

from __future__ import annotations

import threading
import uuid
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from genesis.evaluation.datasets import split_indices
from genesis.evaluation.metrics import regression_report
from genesis.experience.event_log import EventLog
from genesis.experience.schema import Observation
from genesis.learning.learner import Learner, LearnerConfig
from genesis.storage.repository import ExperimentRecord, Repository
from genesis.storage.sqlite_repository import SQLiteRepository
from genesis.version import __version__
from services.api.experiments import build_config, launch_experiment
from services.api.models import (
    CheckpointListResponse,
    CheckpointView,
    ExperimentListResponse,
    ExperimentStartRequest,
    ExperimentView,
    ExperienceListResponse,
    HealthResponse,
    LearningStepRequest,
    LearningStepResponse,
    LogView,
    MetricView,
    PredictRequest,
    PredictResponse,
    ResearchLogResponse,
    RestoreResponse,
    VersionResponse,
)
from services.api.settings import Settings


class AppState:
    """Mutable process state shared by request handlers."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.repository: Repository = SQLiteRepository(settings.db_path)
        self.event_log = EventLog(settings.event_log_path)
        self.active_learner: Learner | None = None
        self.lock = threading.RLock()


def _experiment_view(record: ExperimentRecord) -> ExperimentView:
    return ExperimentView(
        id=record.id,
        name=record.name,
        status=record.status,
        config=record.config,
        seed=record.seed,
        result=record.result,
        error=record.error,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    settings.prepare_directories()
    state = AppState(settings)

    app = FastAPI(
        title="Genesis Research API",
        version=__version__,
        description="Experimental experience-driven representation learning.",
    )
    app.state.genesis = state
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            version=__version__,
            environment=settings.environment,
            database=str(settings.db_path),
        )

    @app.get("/version", response_model=VersionResponse)
    def version() -> VersionResponse:
        return VersionResponse(version=__version__)

    # -- experiences -----------------------------------------------------
    @app.post("/experiences", response_model=Observation, status_code=201)
    def create_experience(observation: Observation) -> Observation:
        try:
            state.repository.add_experience(observation)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        state.event_log.append(
            "experience_created",
            {"observation_id": observation.id, "has_target": observation.has_target},
        )
        return observation

    @app.get("/experiences", response_model=ExperienceListResponse)
    def list_experiences(
        limit: int = Query(default=100, ge=1, le=1000),
        offset: int = Query(default=0, ge=0),
    ) -> ExperienceListResponse:
        return ExperienceListResponse(
            experiences=state.repository.list_experiences(limit=limit, offset=offset)
        )

    # -- learning --------------------------------------------------------
    @app.post("/learning/step", response_model=LearningStepResponse)
    def learning_step(request: LearningStepRequest) -> LearningStepResponse:
        records = state.repository.list_experiences(limit=100_000)
        labeled = [record for record in records if record.has_target]
        if len(labeled) < 2:
            raise HTTPException(
                status_code=422,
                detail="at least two labeled experiences are required for a learning step",
            )

        if request.feature_names:
            feature_names = list(request.feature_names)
        else:
            names: set[str] = set()
            for record in labeled:
                names.update(record.features)
            feature_names = sorted(names)
        if not feature_names:
            raise HTTPException(status_code=422, detail="no feature names available")
        for record in labeled:
            if any(name not in record.features for name in feature_names):
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"experience {record.id!r} is missing one of the requested "
                        "feature names"
                    ),
                )

        experiment_id = str(uuid.uuid4())
        neural = request.neural.model_copy(
            update={"seed": request.seed if request.seed is not None else request.neural.seed}
        )
        config = LearnerConfig(feature_names=feature_names, neural=neural)
        state.repository.create_experiment(
            experiment_id=experiment_id,
            name=request.experiment_name,
            status="running",
            config=config.model_dump(mode="json"),
            seed=neural.seed,
        )
        state.event_log.append(
            "learning_step_started",
            {"experiment_id": experiment_id, "n_labeled": len(labeled)},
        )

        try:
            indices = split_indices(
                len(labeled), [1.0 - request.validation_fraction, request.validation_fraction],
                seed=neural.seed,
            )
            train = [labeled[int(i)] for i in indices[0]]
            validation = [labeled[int(i)] for i in indices[1]]
            learner = Learner(config)
            learner.fit(train, validation_records=validation or None)
            evaluation_records = validation or train
            predictions = learner.predict(learner.encoder.transform(evaluation_records))
            metrics = regression_report(
                np.asarray([record.outcome for record in evaluation_records]),
                predictions,
            )

            checkpoint_id = None
            if request.save_checkpoint:
                checkpoint_dir = settings.checkpoint_dir.resolve()
                metadata = learner.save(
                    checkpoint_dir,
                    metrics={"mse": metrics["mse"], "r2": metrics["r2"]},
                    tags=["learning_step", request.experiment_name],
                )
                state.repository.add_checkpoint(
                    path=str(checkpoint_dir / f"{metadata.checkpoint_id}.pt"),
                    metadata=metadata,
                )
                checkpoint_id = metadata.checkpoint_id

            with state.lock:
                state.active_learner = learner

            payload: dict[str, Any] = {
                "metrics": metrics,
                "checkpoint_id": checkpoint_id,
                "n_train": len(train),
                "n_validation": len(validation),
                "learner": learner.describe(),
            }
            state.repository.update_experiment(
                experiment_id, status="completed", result=payload
            )
            state.event_log.append(
                "learning_step_completed",
                {
                    "experiment_id": experiment_id,
                    "checkpoint_id": checkpoint_id,
                    "mse": metrics["mse"],
                },
            )
            return LearningStepResponse(experiment_id=experiment_id, status="completed", **payload)
        except Exception as exc:  # noqa: BLE001 - failures must be persisted
            state.repository.update_experiment(
                experiment_id, status="failed", error=f"{type(exc).__name__}: {exc}"
            )
            state.event_log.append(
                "learning_step_failed",
                {"experiment_id": experiment_id, "error": str(exc)},
            )
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        learner: Learner | None
        checkpoint_id = request.checkpoint_id
        if checkpoint_id is not None:
            record = state.repository.get_checkpoint(checkpoint_id)
            if record is None:
                raise HTTPException(status_code=404, detail="checkpoint not found")
            learner, _ = Learner.from_checkpoint(record.path)
        else:
            with state.lock:
                learner = state.active_learner
            checkpoint_id = learner.checkpoint_id if learner is not None else None
        if learner is None:
            raise HTTPException(
                status_code=409,
                detail="no active learner; run a learning step or restore a checkpoint",
            )

        observation = Observation(
            features=request.features,
            action=request.action,
            outcome=request.actual,
        )
        try:
            prediction = learner.predict_value(observation)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        error = None if request.actual is None else prediction - request.actual
        state.event_log.append(
            "prediction_made",
            {
                "checkpoint_id": checkpoint_id,
                "prediction": prediction,
                "actual": request.actual,
            },
        )
        return PredictResponse(
            prediction=prediction,
            actual=request.actual,
            error=error,
            checkpoint_id=checkpoint_id,
            feature_names=list(learner.spec.feature_names),
        )

    # -- experiments -----------------------------------------------------
    @app.post("/experiments", response_model=ExperimentView, status_code=202)
    def start_experiment(request: ExperimentStartRequest) -> ExperimentView:
        try:
            config = build_config(request.experiment, request.config)
        except Exception as exc:  # noqa: BLE001 - validation feedback
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        experiment_id = str(uuid.uuid4())
        seeds = getattr(config, "seeds", None)
        seed = int(seeds[0]) if seeds else settings.default_seed
        record = state.repository.create_experiment(
            experiment_id=experiment_id,
            name=request.experiment,
            status="running",
            config=config.model_dump(mode="json"),
            seed=seed,
        )
        state.event_log.append(
            "experiment_started", {"experiment_id": experiment_id, "kind": request.experiment}
        )
        launch_experiment(
            state.repository,
            settings,
            record,
            request.experiment,
            request.config,
        )
        return _experiment_view(record)

    @app.get("/experiments", response_model=ExperimentListResponse)
    def list_experiments(
        limit: int = Query(default=50, ge=1, le=500)
    ) -> ExperimentListResponse:
        return ExperimentListResponse(
            experiments=[
                _experiment_view(record)
                for record in state.repository.list_experiments(limit=limit)
            ]
        )

    @app.get("/experiments/{experiment_id}", response_model=ExperimentView)
    def get_experiment(experiment_id: str) -> ExperimentView:
        record = state.repository.get_experiment(experiment_id)
        if record is None:
            raise HTTPException(status_code=404, detail="experiment not found")
        return _experiment_view(record)

    @app.get("/experiments/{experiment_id}/metrics", response_model=list[MetricView])
    def get_metrics(experiment_id: str) -> list[MetricView]:
        if state.repository.get_experiment(experiment_id) is None:
            raise HTTPException(status_code=404, detail="experiment not found")
        return [
            MetricView(
                name=metric.name,
                value=metric.value,
                step=metric.step,
                payload=metric.payload,
            )
            for metric in state.repository.list_metrics(experiment_id)
        ]

    # -- checkpoints -----------------------------------------------------
    @app.get("/checkpoints", response_model=CheckpointListResponse)
    def list_checkpoints(
        limit: int = Query(default=100, ge=1, le=500)
    ) -> CheckpointListResponse:
        return CheckpointListResponse(
            checkpoints=[
                CheckpointView(
                    checkpoint_id=record.checkpoint_id,
                    path=record.path,
                    metadata=record.metadata,
                    created_at=record.created_at,
                )
                for record in state.repository.list_checkpoints(limit=limit)
            ]
        )

    @app.post("/checkpoints/{checkpoint_id}/restore", response_model=RestoreResponse)
    def restore_checkpoint(checkpoint_id: str) -> RestoreResponse:
        record = state.repository.get_checkpoint(checkpoint_id)
        if record is None:
            raise HTTPException(status_code=404, detail="checkpoint not found")
        try:
            learner, metadata = Learner.from_checkpoint(record.path)
        except Exception as exc:  # noqa: BLE001 - report corrupt checkpoints clearly
            raise HTTPException(status_code=500, detail=f"checkpoint load failed: {exc}") from exc
        with state.lock:
            state.active_learner = learner
        state.event_log.append("checkpoint_restored", {"checkpoint_id": checkpoint_id})
        return RestoreResponse(
            checkpoint_id=checkpoint_id,
            restored=True,
            learner=learner.describe(),
            metadata=metadata.model_dump(mode="json"),
        )

    # -- research log ----------------------------------------------------
    @app.get("/research-log", response_model=ResearchLogResponse)
    def research_log() -> ResearchLogResponse:
        markdown = ""
        if settings.research_log_path.exists():
            markdown = settings.research_log_path.read_text(encoding="utf-8")
        return ResearchLogResponse(
            markdown=markdown,
            logs=[
                LogView(
                    id=log.id,
                    level=log.level,
                    message=log.message,
                    payload=log.payload,
                    created_at=log.created_at,
                )
                for log in state.repository.list_logs(limit=100)
            ],
        )

    return app

