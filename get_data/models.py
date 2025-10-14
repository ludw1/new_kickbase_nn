from pydantic import BaseModel, Field
from typing import Optional


class UserDict(BaseModel):
    id: int
    name: str
    email: str


class LeaguesDict(BaseModel):
    id: str
    name: str
    creator: str
    creation: str  # Creation date
    au: int  # Probably active users


class LoginResponse(BaseModel):
    u: UserDict
    srvl: list[LeaguesDict]
    tkn: str  # Token
    tknex: str  # Token expiration date


class TeamResponse(BaseModel):
    class Player(BaseModel):
        i: str
        n: str
        pos: int
        mv: int

    tid: int
    tn: str
    it: list[Player]


class PlayerMarketValueResponse(BaseModel):
    class MarketValueEntry(BaseModel):
        dt: int  # Date
        mv: int  # Market value

    it: list[MarketValueEntry]

class PlayerInfo(BaseModel):
    i: int
    fn: Optional[str] = Field(default=None) # First name
    ln: str # Last name
    shn: int # Shirt number
    tid: int # Team ID
    tn: str # Team name
    st: int # Status, probably 0 for fit, 1 for injury
    stl: list[int] # Status list
    stxt: Optional[str] = Field(default=None) # Status text
    pos: int # Position
    iposl: bool # No idea
    tp: Optional[int] = Field(default=None) # Total points
    ap: Optional[int] = Field(default=None) # Average points
    g: Optional[int] = Field(default=None) # Goals
    a: Optional[int] = Field(default=None) # Assists


class PlayerPerformanceEntry(BaseModel):
    class PerformanceDetail(BaseModel):
        class MatchInfo(BaseModel):
            day: int # Matchday
            p: Optional[int] = Field(default=None) # Points in that match
            mp: Optional[str] = Field(default=None) # Minutes played in that match
            md: str # Match date in format 2023-08-18T15:30:00Z
            t1: str # Team 1 id
            t2: str # Team 2 id
            t1g: Optional[int] = Field(default=None) # Team 1 goals
            t2g: Optional[int] = Field(default=None) # Team 2 goals
            pt: Optional[str] = Field(default=None) # Team id of the player
            st: Optional[int] = Field(default=None) # Status, 5 is fit and playing, anything else I don't know
            cur: bool # Is this the current match day
            ap: Optional[int] = Field(default=None) # Average points
            tp: Optional[int] = Field(default=None) # Total points
            asp: Optional[int] = Field(default=None) # Average points across all seasons


        ti: str # Season title 2013/2014 for example
        n: str # Name of the competition
        ph: list[MatchInfo] # list of matches

    it: list[PerformanceDetail]
