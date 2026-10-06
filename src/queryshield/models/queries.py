from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator, Field


class RequestTimeWindow(BaseModel):
    """Request-level window (a date picker); the same rules as the model's window declarations."""

    model_config = ConfigDict(extra="forbid")
    start: str = Field(max_length=40)
    end: str = Field(max_length=40)
    timezone: Literal["UTC"] = "UTC"

    @model_validator(mode="after")
    def check_window(self) -> "RequestTimeWindow":
        from queryshield.agent.metric_intent import MetricDeclarationError, normalize_time_window

        try:
            normalize_time_window(self.as_window(), field="time_window")
        except MetricDeclarationError as exc:
            raise ValueError(exc.message) from exc
        return self

    def as_window(self) -> dict[str, str]:
        return {"start": self.start, "end": self.end, "timezone": self.timezone}


def _strip_nonblank(value: object, name: str) -> object:
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            raise ValueError(f"{name} must not be blank")
        return cleaned
    return value


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(max_length=500)
    time_window: RequestTimeWindow | None = None

    @field_validator("question", mode="before")
    @classmethod
    def strip_question(cls, value: object) -> object:
        return _strip_nonblank(value, "question")


class QueryProposalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    proposal: str = Field(max_length=4000)

    @field_validator("proposal", mode="before")
    @classmethod
    def strip_proposal(cls, value: object) -> object:
        return _strip_nonblank(value, "proposal")
