from pydantic import BaseModel, Field


class IncomingReview(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    # Optional provenance, carried through to the joined output so the eval
    # script can match results back to dataset labels — the label itself is
    # never sent into the pipeline.
    source: str | None = Field(default=None, max_length=64)
    source_id: str | None = Field(default=None, max_length=128)
