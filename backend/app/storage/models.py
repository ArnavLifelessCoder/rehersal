"""SQLAlchemy models.

Mirrors the spec's schema with two deliberate changes:

  * `jd_embedding` is a JSON float array rather than a pgvector column. The
    session flow only ever compares a JD against turns *within* one session, so
    there is no ANN search to accelerate -- pgvector would be a hard dependency
    bought for nothing. Cross-session JD search, if it ever ships, is the point
    to add it.
  * Feature vectors are stored whole (`features` JSON) next to the derived
    scores. Scores are a lossy function of features; keeping the inputs means a
    retrained scorer can be re-run over historical sessions instead of needing a
    fresh data collection. This is the cheapest possible insurance for §7's
    ablation, and it costs about 2KB per turn.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    JSON, DateTime, Float, ForeignKey, Integer, String, Text, create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    email: Mapped[str | None] = mapped_column(String(255), unique=True, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class Session(Base):
    __tablename__ = "sessions"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    user_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("users.id"), nullable=True)
    track: Mapped[str] = mapped_column(String(32))
    jd_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    jd_embedding: Mapped[list | None] = mapped_column(JSON, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="active")
    starred: Mapped[int] = mapped_column(Integer, default=0)  # starred => blobs kept

    turns: Mapped[list["Turn"]] = relationship(back_populates="session", cascade="all, delete")
    report: Mapped["SessionReport | None"] = relationship(
        back_populates="session", cascade="all, delete", uselist=False
    )


class Turn(Base):
    __tablename__ = "turns"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(32), ForeignKey("sessions.id"))
    idx: Mapped[int] = mapped_column(Integer)
    role: Mapped[str] = mapped_column(String(16))
    question_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    transcript: Mapped[str | None] = mapped_column(Text, nullable=True)
    audio_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    duration_s: Mapped[float] = mapped_column(Float, default=0.0)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    features: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    delivery: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    content: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    fillers: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    timings: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    session: Mapped[Session] = relationship(back_populates="turns")


class SessionReport(Base):
    __tablename__ = "session_reports"
    session_id: Mapped[str] = mapped_column(String(32), ForeignKey("sessions.id"), primary_key=True)
    overall_delivery: Mapped[float] = mapped_column(Float, default=0.0)
    overall_content: Mapped[float] = mapped_column(Float, default=0.0)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    payload: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    session: Mapped[Session] = relationship(back_populates="report")


def make_engine(database_url: str):
    kwargs: dict = {"future": True}
    if database_url.startswith("sqlite"):
        # check_same_thread=False: the actor runs scoring in threads, and
        # SQLite's default thread affinity would reject those writes.
        kwargs["connect_args"] = {"check_same_thread": False}
    engine = create_engine(database_url, **kwargs)
    Base.metadata.create_all(engine)
    return engine


def make_session_factory(engine):
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)
