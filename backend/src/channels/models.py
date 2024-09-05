from sqlmodel import SQLModel, Field, Column
import sqlalchemy.dialects.postgresql as pg
from datetime import datetime
import uuid

class Channel(SQLModel, table=True):
    __tablename__ = "channel"

    id: int = Field(
        sa_column=Column(
            pg.INTEGER,
            primary_key=True,
            unique=True,
            nullable=False,
            info={"description": "Unique identifier for the channel"},
            autoincrement=True
        )
    )

    user_id: uuid.UUID  = Field(nullable=False,foreign_key="user_accounts.uid")
    name: str = Field(nullable=True)
    description: str = Field(nullable=True)
    created_at: datetime = Field(sa_column=Column(pg.TIMESTAMP, default=datetime.now))


    def __repr__(self) -> str:
        return f"<Channel No. {self.id}>"