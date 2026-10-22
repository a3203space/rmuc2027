"""Small versioned wire contract; never accept arbitrary engine commands."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from . import PROTOCOL, SLOTS


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Join(Message):
    protocol: Literal[PROTOCOL] = PROTOCOL
    nickname: str = Field(min_length=1, max_length=24)
    token: str | None = Field(default=None, max_length=128)
    host_key: str | None = Field(default=None, max_length=128)

    @field_validator("nickname")
    @classmethod
    def clean_name(cls, value):
        value = value.strip()
        if not value or any(ord(c) < 32 for c in value):
            raise ValueError("昵称不能为空或包含控制字符")
        return value


class Seat(Message):
    team: Literal["red", "blue"]
    slot: int

    @field_validator("slot")
    @classmethod
    def valid_slot(cls, value):
        if value not in SLOTS:
            raise ValueError("该席位不属于 2027 阵容")
        return value


class Ready(Message):
    ready: bool


class Profile(Message):
    chassis: Literal["hp", "power"]
    weapon: Literal["cooling", "burst"]


class SeatConfig(Seat):
    model: int = Field(strict=True)
    strategy: str = Field(default="baseline", max_length=160)


class TeamConfig(Message):
    team: Literal["red", "blue"]
    mode: Literal["independent", "team"]
    strategy: str = Field(default="baseline", max_length=160)


class StageReady(Message):
    match_id: str
    error: str = Field(default="", max_length=300)


class StrategyImport(Message):
    files: dict[str, str]


class Input(Message):
    type: Literal["input"] = "input"
    match_id: str
    seq: int = Field(ge=0, strict=True)
    forward: float = Field(default=0, ge=-1, le=1)
    strafe: float = Field(default=0, ge=-1, le=1)
    yaw: float = Field(ge=-1e6, le=1e6)
    pitch: float = Field(ge=-1.57, le=1.57)
    fire: bool = False
    boost: bool = False
    jump: bool = False
    crouch: bool = False
    active: bool = True
    transform: bool | None = None


class Operation(Message):
    type: Literal["operation"] = "operation"
    match_id: str
    seq: int = Field(ge=0, strict=True)
    name: Literal["resume", "pick", "assemble", "cancel", "install", "revive",
                  "heal", "ammo", "posture", "air_target", "air_launch", "air_return"]
    value: str = Field(default="", max_length=32)
