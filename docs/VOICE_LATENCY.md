# Ava voice latency diagnosis (VAD end → first audio output)

The existing `/tts status` summary used to begin when a voice turn had
**already** passed ASR and speaker identity. That made the visible total lower
than the human-perceived response latency. This branch carries the local
`vad_end_at` timestamp from ASR into the same monotonic timebase as the
Router, Speaker, and TTS.

## Start and inspect

From the repository root:

```bash
COMPANION_LOG=logs/voice-latency.log \
  .venv/bin/python -m companion repl --provider deepseek --voice --debug
```

Run `/mic on` if the microphone is not already listening. Talk normally,
wait until Ava has spoken, then type:

```text
/latency
/tts status
/speaker status
/mic status
```

To read machine-readable per-turn traces in another terminal:

```bash
grep -E 'voice_asr_latency|voice_gate_latency|router_latency|router_tool_latency|voice_e2e_latency|action_saved' \
  logs/voice-latency.log | tail -100
```

**Privacy:** these latency lines use numeric timestamps, run/utterance/event
IDs and durations, not voice samples, transcripts or API keys. The general
DEBUG log can include other diagnostic details; share selected latency lines,
not the entire file or your `.env`.

## Stage meanings

- `vad_to_asr_ms`: VAD marked the end of the user's speech → ASR final
- `asr_to_identity_ms`: ASR final → resolved person, including bounded waits
- `identity_to_router_ms`: identity resolution → end of semantic routing,
  including local preparation and any tool calls (not pure network LLM time)
- `router_to_response_ms`: routing completed → response stream started
- `response_to_text_ms`: response stream started → first visible text
- `text_to_segment_ms`: first text → first TTS-ready segment
- `tts_connect_ms`: TTS WebSocket handshake, which can overlap segment wait
- `segment_to_pcm_ms`: first segment sent → first received PCM
- `pcm_to_output_ms`: first PCM received → first playback write completed
- `vad_to_output_ms`: VAD end → first PCM frame accepted by PortAudio

The last item is **not** a physical acoustic measurement. Browser/device
buffers and wireless speakers may add extra delay.

TTS handshake and first segment preparation can overlap. Do not add every
row together as if they were serial.

For missing/TTS-off voice replies, check `voice_asr_latency` and
`identity_decision`, plus `action_saved` for Router/Speaker timing. A
turn rejected by speaker identity has no valid TTS trace.

## Controlled first-clause A/B

The default TTS splitter waits for a full sentence. On a long first sentence,
waiting for its punctuation can delay speech even after first text arrives.

To optionally emit the **first safe clause** sooner, launch with:

```bash
AVA_TTS_FIRST_CLAUSE_CHARS=12 \
  .venv/bin/python -m companion repl --provider deepseek --voice
```

A comma/semicolon/colon can end only that first segment, after the minimum
visible-character count. After that one early segment, the normal
sentence-first splitter resumes. Embedded tags, code, URLs, and numeric
punctuation still obey the existing safety parser.

`AVA_TTS_FIRST_CLAUSE_CHARS=0` (default) disables the experiment.

Record 5–10 similarly long sentences under each setting. Compare
`text_to_segment_ms`, `vad_to_output_ms`, and whether the spoken phrasing
sounds natural. Keep 0 if quality deteriorates or there is no measurable gain.

## Boundaries

This instrumentation does not change Router choice, identity security
thresholds, ASR provider, or DeepSeek. There is no claim that the model/network
latency improved. The first-clause switch is an optional measured tradeoff.

## Targeted diagnosis after a slow turn

`router_latency` reports **initial LLM**, **follow-up LLM**,
**tool execution** and **local/DB overhead** separately. For tool work,
`router_tool_latency` logs each tool name, status, and duration, without
storing search queries or response text.

An `Identity → Router` measurement includes local preparation **before**
RouterService, so it is not equivalent to `initial_llm_ms`. Pair the
`voice_gate_latency event_id=...` and `router_latency event_id=...`
records to find the same turn.

`voice_asr_latency` now records
`vad_to_input_closed_ms` (send-queue drain and local stop),
`input_closed_to_server_done_ms` (server-side processing/network and
client completion), and `server_done_to_final_ms` (local dispatch).
They may be `None` when the server completes before the sender-side
completion callback; this is not a missing-recognition error.

Do not change the core Router, skip required tools, accept incomplete ASR,
or force a particular timeout until repeated traces identify the expensive
stage. In particular, a network search or second Router call can account
for a high identity-to-router value.
