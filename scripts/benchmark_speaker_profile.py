"""Latency benchmark for the Speaker -> Person -> Profile chain (V0.9.9.31.1).

This script measures *plumbing* latency only. It never reads audio, never
loads the ERes2NetV2 model and never touches a real user database: everything
runs in a temporary directory over deterministic synthetic embeddings. The
numbers are therefore **not** recognition accuracy and **not** real-device
latency. ASR Final -> Identity Confirmed P50/P95 must be read from a real
session's ``/speaker status`` after device testing; nothing here may be quoted
as an accuracy or latency claim for the product.

Measured stages (each over ``--iterations`` runs, P50/P95 in milliseconds):

  match            matcher decision over ``--voices`` synthetic profiles
  resolve_identity SpeakerResult -> ResolvedIdentity (SQLite binding lookup)
  resolve_profile  speaker id -> person id -> profile
  preview_link     explicit-link preview (read-only)

No raw audio, embeddings or private text are written to disk or logs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np

from companion.people.identity_resolver import IdentityResolver, IdentityState
from companion.people.speaker_profile_link import (
    PROFILE_READY,
    SpeakerProfileLinkService,
)
from companion.runtime import build_runtime
from companion.speaker_id.config import EMBEDDING_DIM, load_speaker_config
from companion.speaker_id.matcher import SpeakerMatcher
from companion.speaker_id.models import QUALITY_HIGH, SpeakerResult
from companion.speaker_id.quality import VoiceQuality
from companion.speaker_id.store import SpeakerStore
from companion.storage.database import Database


def percentile(samples: list[float], fraction: float) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    index = int(round(fraction * (len(ordered) - 1)))
    return ordered[max(0, min(len(ordered) - 1, index))]


def summarise(samples: list[float]) -> dict:
    return {
        "n": len(samples),
        "mean_ms": round(statistics.fmean(samples), 3) if samples else 0.0,
        "p50_ms": round(percentile(samples, 0.50), 3),
        "p95_ms": round(percentile(samples, 0.95), 3),
        "max_ms": round(max(samples), 3) if samples else 0.0,
    }


def synthetic_vector(index: int) -> np.ndarray:
    """Deterministic unit vector; identity-shaped, deliberately not speech."""
    vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    vector[index % EMBEDDING_DIM] = 1.0
    vector[(index * 7 + 3) % EMBEDDING_DIM] = 0.35
    return vector / np.linalg.norm(vector)


def model_key(config) -> str:
    return f"{config.model_id}@{config.model_revision}"


def seed_speakers(store: SpeakerStore, config, voices: int) -> list[str]:
    speaker_ids: list[str] = []
    for index in range(voices):
        speaker_id = store.create_candidate()
        outcome = store.add_embedding(
            speaker_id,
            synthetic_vector(index),
            quality=QUALITY_HIGH,
            voiced_seconds=2.0,
            model_id=model_key(config),
            embedding_version=config.embedding_version,
            is_verified_reference=True,
        )
        if outcome != "ADDED":
            raise SystemExit(f"cannot seed synthetic voice {index}: {outcome}")
        stable = store.promote_speaker(speaker_id)
        speaker_ids.append(stable or speaker_id)
    return speaker_ids


def known_result(speaker_id: str, uid: int) -> SpeakerResult:
    return SpeakerResult(
        utterance_id=uid,
        speaker_id=speaker_id,
        person_id=None,
        status="KNOWN",
        reid_quality=QUALITY_HIGH,
        similarity=0.9,
        margin=0.3,
        voiced_seconds=2.2,
        provisional=False,
        model_id="benchmark",
        inference_ms=0.0,
        completed_at=0.0,
    )


async def run(args) -> int:
    config = load_speaker_config(env={})
    quality = VoiceQuality(usable=True, voiced_seconds=2.2, quality=QUALITY_HIGH)

    with tempfile.TemporaryDirectory(prefix="ava-profile-bench-") as tmp:
        tmp_path = Path(tmp)
        speaker_db = tmp_path / "speakers.sqlite3"
        main_db = tmp_path / "ava.sqlite3"

        store = SpeakerStore(speaker_db)
        store.initialize()
        database = await Database(main_db).connect()
        await database.initialize()
        try:
            rt = build_runtime(database)
            speaker_ids = seed_speakers(store, config, args.voices)
            if not speaker_ids:
                raise SystemExit("--voices must be at least 1")

            persons = []
            for index, speaker_id in enumerate(speaker_ids):
                person = await rt.registry.register(
                    f"bench-person-{index}",
                    "friend",
                    allow_processing=True,
                    allow_persistence=True,
                    allow_memory_learning=False,
                )
                await rt.speaker_bindings.bind_speaker(
                    person.person_id, speaker_id, "benchmark_seed"
                )
                persons.append(person)

            matcher = SpeakerMatcher(config)
            candidates, _ = store.load_candidates(
                model_id=model_key(config),
                embedding_version=config.embedding_version,
            )
            if not candidates:
                raise SystemExit("no synthetic candidates loaded")

            resolver = IdentityResolver(
                rt.repos.person, rt.speaker_bindings, store
            )
            link = SpeakerProfileLinkService(
                rt.speaker_bindings,
                rt.repos.person,
                store,
                rt.profile_service,
                audit=rt.repos.audit,
            )

            vectors = [
                synthetic_vector(index)
                for index in range(args.voices)
            ]
            query = vectors[0]
            target_speaker = speaker_ids[0]
            target_person = persons[0].person_id

            match_ms: list[float] = []
            resolve_ms: list[float] = []
            profile_ms: list[float] = []
            preview_ms: list[float] = []
            confirmed = 0
            ready = 0

            for iteration in range(args.iterations):
                started = time.monotonic()
                decision = matcher.match(
                    query, candidates, quality=quality
                )
                match_ms.append((time.monotonic() - started) * 1000.0)
                if decision.status not in ("KNOWN", "CANDIDATE_MATCH"):
                    # Sanity only: a synthetic query must reach the matcher,
                    # it is not an accuracy measurement.
                    raise SystemExit(
                        "synthetic query did not reach the KNOWN gate"
                    )

                started = time.monotonic()
                identity = await resolver.resolve(
                    known_result(target_speaker, iteration),
                    run_id="benchmark",
                    utterance_id=iteration,
                )
                resolve_ms.append((time.monotonic() - started) * 1000.0)
                if identity.state is IdentityState.CONFIRMED:
                    confirmed += 1
                if identity.person_id != target_person:
                    raise SystemExit(
                        "binding did not resolve to the seeded person"
                    )

                started = time.monotonic()
                resolution = await link.resolve_profile(target_speaker)
                profile_ms.append((time.monotonic() - started) * 1000.0)
                if resolution.state == PROFILE_READY:
                    ready += 1

                started = time.monotonic()
                await link.preview_link(target_speaker, target_person)
                preview_ms.append((time.monotonic() - started) * 1000.0)

            report = {
                "mode": "synthetic-offline",
                "caveat": (
                    "synthetic vectors in a temporary database; plumbing"
                    " latency only, never accuracy or real-device latency"
                ),
                "voices": args.voices,
                "iterations": args.iterations,
                "stages_ms": {
                    "match": summarise(match_ms),
                    "resolve_identity": summarise(resolve_ms),
                    "resolve_profile": summarise(profile_ms),
                    "preview_link": summarise(preview_ms),
                },
                "sanity": {
                    "confirmed_identity": f"{confirmed}/{args.iterations}",
                    "profile_ready": f"{ready}/{args.iterations}",
                },
                "real_device_latency": (
                    "untested here; read ASR Final -> Identity Confirmed"
                    " P50/P95 from /speaker status on a real device"
                ),
                "raw_audio_used": False,
                "real_user_database_used": False,
            }

            if args.json:
                Path(args.json).write_text(
                    json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            print("Speaker -> Person -> Profile benchmark")
            print(
                f"voices={args.voices} iterations={args.iterations}"
                " (synthetic, temporary database; no audio)"
            )
            print(f"{'stage':<18}{'n':>6}{'p50 ms':>10}{'p95 ms':>10}{'max ms':>10}")
            for name, stats in report["stages_ms"].items():
                print(
                    f"{name:<18}{stats['n']:>6}{stats['p50_ms']:>10.3f}"
                    f"{stats['p95_ms']:>10.3f}{stats['max_ms']:>10.3f}"
                )
            print(
                "note: plumbing latency only. Recognition accuracy and"
                " real-device ASR-to-identity latency are untested here."
            )
            if args.json:
                print(f"json: {args.json}")
            return 0
        finally:
            store.close()
            await database.close()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Offline P50/P95 benchmark for the Speaker -> Person -> Profile"
            " chain. Never reads audio or a real user database."
        )
    )
    parser.add_argument(
        "--voices",
        type=int,
        default=8,
        help="synthetic voice profiles to seed (default: 8)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="measured runs per stage (default: 100)",
    )
    parser.add_argument(
        "--json", type=str, default=None, help="write the report as JSON"
    )
    args = parser.parse_args(argv)
    if args.voices < 1:
        parser.error("--voices must be at least 1")
    if args.iterations < 1:
        parser.error("--iterations must be at least 1")
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
