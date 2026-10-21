from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class PropLine:
    source: str
    player: str
    team: str
    opponent: str
    stat: str  # "kills" | "headshots"
    maps: tuple  # e.g. (1, 2), (1, 2, 3), (3,)
    line: float
    start_time: Optional[datetime] = None
    over_decimal: Optional[float] = None
    under_decimal: Optional[float] = None
    odds_type: str = "standard"  # PrizePicks: standard / goblin / demon
    over_only: bool = False


@dataclass
class MatchOdds:
    source: str
    team1: str
    team2: str
    start_time: Optional[datetime]
    best_of: int
    p_series_team1: Optional[float] = None
    p_map_team1: dict = field(default_factory=dict)  # map number -> devigged prob
    rounds_ladder: dict = field(default_factory=dict)  # map number -> [(line, p_over)]
    p_overtime: dict = field(default_factory=dict)  # map number -> prob
    p_map3: Optional[float] = None
    series_dec: Optional[tuple] = None  # quoted decimal series prices (team1, team2), with vig; sportsbooks only
    p_map_quoted: Optional[dict] = None  # map number -> devigged prob the book actually quoted (before series inference)


@dataclass
class MapRecord:
    date: datetime
    opponent: str
    map_name: str
    rounds: int
    rounds_won: int
    kills: int
    headshots: Optional[int] = None


@dataclass
class PlayerStats:
    player: str
    team: str
    source: str
    maps: list  # list[MapRecord]
    hs_pct: Optional[float] = None
    hs_sample_kills: int = 0
    kpr_summary: Optional[float] = None
    rounds_summary: int = 0
    url: str = ""
