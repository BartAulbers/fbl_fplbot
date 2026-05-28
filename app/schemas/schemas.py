"""
Pydantic schemas for API request/response models.
"""
from pydantic import BaseModel, Field
from typing import Optional


class PlayerBase(BaseModel):
    player_id: int
    web_name: str
    position: str
    now_cost: float
    team_id: int


class PlayerXPts(PlayerBase):
    xpts: float
    xpts_3gw: float
    xpts_5gw: float
    ownership: float
    consistency: float


class SquadPlayerOut(BaseModel):
    player_id: int
    web_name: str
    position: str
    cost: float
    xpts: float
    xpts_3gw: float
    ownership: float
    consistency: float
    is_starting: bool
    is_captain: bool
    is_vice: bool
    bench_order: Optional[int]


class SquadResultOut(BaseModel):
    squad: list[SquadPlayerOut]
    total_cost: float
    projected_pts_gw: float
    projected_pts_3gw: float
    budget_remaining: float
    solver_status: str


class OptimizeRequest(BaseModel):
    budget: float = Field(default=100.0, ge=50, le=120)
    risk_appetite: float = Field(default=0.5, ge=0.0, le=1.0)
    horizon: str = Field(default="1gw", pattern="^(1gw|3gw|5gw)$")
    locked_player_ids: list[int] = []
    excluded_player_ids: list[int] = []


class TransferPlayerDict(BaseModel):
    player_id: int
    web_name: str
    position: str
    cost: float
    xpts: float
    xpts_3gw: float
    ownership: float
    fdr_3gw: float


class TransferSuggestionOut(BaseModel):
    player_out: TransferPlayerDict
    player_in: TransferPlayerDict
    expected_gain_1gw: float
    expected_gain_3gw: float
    hit_required: bool
    net_gain: float
    reasoning: str
    confidence: str


class TransferPlanOut(BaseModel):
    suggestions: list[TransferSuggestionOut]
    free_transfers_available: int
    current_gw: int
    recommendation: str


class TransferRequest(BaseModel):
    free_transfers: int = Field(default=1, ge=0, le=5)
    risk_appetite: float = Field(default=0.5, ge=0.0, le=1.0)
    max_suggestions: int = Field(default=5, ge=1, le=20)


class FixtureRunRow(BaseModel):
    team_id: int
    team_name: str
    avg_fdr: float
    min_fdr: float
    max_fdr: float
    n_fixtures: int
    fixtures: str
    has_blank: int
    has_double: int


class DifferentialRow(BaseModel):
    player_id: int
    web_name: str
    position: str
    now_cost: float
    selected_by_percent: float
    xpts: float
    xpts_3gw: float
    diff_score: float


class CaptainRow(BaseModel):
    player_id: int
    web_name: str
    position: str
    xpts: float
    consistency: float
    fixture_score_3gw: float
    selected_by_percent: float
    captain_score: float


class PipelineStatus(BaseModel):
    status: str
    message: str
    current_gw: Optional[int]
