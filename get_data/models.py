from pydantic import BaseModel


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