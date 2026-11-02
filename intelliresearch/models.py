from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


@dataclass(frozen=True)
class Upload:
    name: str
    data: bytes


@dataclass(frozen=True)
class Page:
    document_id: str
    filename: str
    number: int
    text: str
    visual_notes: str = ""


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    filename: str
    page: int
    text: str
    kind: str = "text"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Evidence(StrictModel):
    source_id: str
    quote: str = Field(min_length=8, max_length=700)


class Claim(StrictModel):
    text: str = Field(min_length=1, max_length=2000)
    evidence: list[Evidence] = Field(min_length=1, max_length=6)


class GroundedAnswer(StrictModel):
    supported: bool
    claims: list[Claim] = Field(default_factory=list, max_length=12)
    questions: list[str] = Field(default_factory=list, max_length=5)


class ChartPoint(StrictModel):
    label: str = Field(min_length=1, max_length=150)
    value: float = Field(allow_inf_nan=False)
    evidence: Evidence


class ChartSpec(StrictModel):
    title: str = Field(max_length=200)
    kind: Literal["bar", "line"]
    unit: str = Field(max_length=80)
    points: list[ChartPoint] = Field(min_length=1, max_length=50)


@dataclass(frozen=True)
class Answer:
    markdown: str
    sources: tuple[Chunk, ...] = ()


class ResearchError(Exception):
    """A message safe to show in the UI (never raw provider errors)."""
