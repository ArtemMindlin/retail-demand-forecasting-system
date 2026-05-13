from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Any

from retail_forecasting.config import load_config
from retail_forecasting.forecasting.pipeline import run_experiment


app = FastAPI(
    title="Retail Demand Forecasting API",
    description="Operational API for generating replenishment recommendations using probabilistic machine learning.",
    version="1.0.0",
)


class ScoreRequest(BaseModel):
    config_path: str = Field(
        default="configs/default.yaml",
        description="Path to the base configuration YAML.",
    )
    run_name: str | None = Field(
        default=None, description="Custom run name for the operational output."
    )


class ScoreResponse(BaseModel):
    status: str
    run_directory: str
    recommendations_generated: int
    recommendations: list[dict[str, Any]]


@app.get("/health")
def health_check() -> dict[str, str]:
    """Verify that the API is up and running."""
    return {"status": "ok", "service": "Retail Forecasting API"}


@app.post("/predict_orders", response_model=ScoreResponse)
def predict_orders(request: ScoreRequest) -> ScoreResponse:
    """
    Trigger the operational scoring pipeline.

    This endpoint loads the configuration, enforces the 'score_daily' operational mode,
    runs the pipeline (which includes data validation, model prediction, Conformal Prediction,
    and LP Order Optimization), and returns the final actionable order recommendations.
    """
    config_file = Path(request.config_path)
    if not config_file.exists():
        raise HTTPException(
            status_code=404, detail=f"Configuration file not found: {config_file}"
        )

    try:
        settings = load_config(config_file)

        # Enforce operational mode
        new_project = settings.project.model_copy(update={"run_mode": "score_daily"})
        settings = settings.model_copy(update={"project": new_project})

        # Handle custom run name
        if request.run_name:
            new_reporting = settings.reporting.model_copy(
                update={"run_name": request.run_name}
            )
            settings = settings.model_copy(update={"reporting": new_reporting})

        # Run the end-to-end pipeline
        artifacts = run_experiment(settings)

        if artifacts.run_directory is None or artifacts.reorder_recommendations is None:
            raise HTTPException(
                status_code=500,
                detail="Pipeline failed to generate operational artifacts.",
            )

        recommendations_df = artifacts.reorder_recommendations

        # We might not want to return thousands of rows in the payload, but for this MVP we return all.
        # Ensure NaNs are handled for JSON serialization
        recommendations_df = recommendations_df.fillna(value="")

        recs_list = recommendations_df.to_dict(orient="records")

        return ScoreResponse(
            status="success",
            run_directory=str(artifacts.run_directory),
            recommendations_generated=len(recs_list),
            recommendations=recs_list,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Pipeline execution failed: {str(e)}"
        )
