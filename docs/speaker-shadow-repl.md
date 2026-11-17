# Formal REPL speaker shadow — phases 0 and 1

This is an observation-only integration of the existing speaker-ID engine into the real voice REPL. There is no automatic Person, Profile, Router, or Memory attribution.

## Feature flag and rollback

- Default is AVA_REPL_SPEAKER_MODE=off; the original ASRManager is constructed and voice routing is unchanged.
- Enable explicitly with AVA_REPL_SPEAKER_MODE=shadow when running python -m companion repl --provider deepseek --voice.
- Any unsupported flag value fails closed to off.
- Shadow mode overrides AVA_VOICE_PERSON_BINDING=1 inside this REPL to preserve the existing startup person for voice, without resolving any recognized speaker.
- Roll back by setting AVA_REPL_SPEAKER_MODE=off or unsetting it and restarting the REPL. No schema migration is performed.

## Lifecycle, privacy, failure handling

Only in shadow mode does each microphone session own a DeferredSpeakerObserver. Its model loads and warms on a background thread after ASR startup. ASR finals are never delayed waiting for identity; a final can show PENDING, followed by a later speaker update. Failed model/configuration never blocks ASR. The observer closes with /mic off, device-start failure, REPL shutdown or a subsequent /mic restart.

The existing voice-print engine can persist pseudonymous embeddings and observations in its own local SQLite database while shadow is on. Explicit consent is necessary before collecting these. Use AVA_SPEAKER_DB with a separate SQLite path to avoid modifying existing test identities. This phase writes no additional audio or transcript logs, no Person or Profile records, and no long-term memory.

## Diagnostics

Use /mic status to see speaker_shadow=LOADING / READY / UNAVAILABLE and the number of completed worker updates. Metadata-only events are written to logs/companion.log, or COMPANION_LOG when configured. They include speaker_shadow_created, speaker_shadow_final (possibly PENDING), speaker_shadow_update (possibly late) and safe fallback warnings. Each microphone startup gets a run_id to disambiguate repeating ASR utterance IDs. Raw similarity is not a probability.

## Verification

Run: python -m pytest -q tests/test_repl_speaker_shadow.py tests/test_asr_speaker_integration.py tests/test_voice_integration.py

On the Mac, compare baseline versus shadow ASR-final time, TTS-start time, CPU and memory. Even with nonblocking thread callbacks, local model inference may compete for CPU and affect latency. Disable shadow if voice reliability regresses.

Not implemented in phases 0/1: stable person binding, speaker aliases/merges, profile learning, historical attribution, multi-person routing, overlapping-speech diarization.
