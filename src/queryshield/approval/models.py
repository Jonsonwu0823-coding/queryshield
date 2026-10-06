"""Strict request models for approval/cancellation endpoints."""

from pydantic import BaseModel, ConfigDict, StrictBool, field_validator


class ApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    approval_id: str
    decision: str

    @field_validator("approval_id")
    @classmethod
    def valid_approval_id(cls, value: str) -> str:
        if type(value) is not str or not value.strip():
            raise ValueError("approval_id must not be blank")
        return value

    @field_validator("decision")
    @classmethod
    def valid_decision(cls, value: str) -> str:
        if value not in {"approve", "reject"}:
            raise ValueError("decision must be approve or reject")
        return value


class EmptyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ResumeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str

    @field_validator("answer")
    @classmethod
    def valid_answer(cls, value: str) -> str:
        if type(value) is not str or not value.strip() or len(value) > 1000:
            raise ValueError("answer must be a non-empty string")
        return value.strip()


class PreferenceUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: str
    confirmed: StrictBool
