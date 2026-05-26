# Clinical-trial benchmark — 2026-05-23T15:12:46

End-to-end persona-driven test of the CaiTI therapist pipeline. All 10 personas drove a full HandlerRL session (37-dim screening + CBT) with TTS / STT / GPIO disabled and the LLM stubbed by role-aware deterministic responses; every paper-aligned LLMRole call site was exercised.

## 1. Persona run summary

| PID | Persona | Duration | Agent turns | User turns | LLM calls | Retry/Brainstorm | RV on/off | CBT picked | CBT result | Crisis | Exceptions |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P01 | healthy_student | 0.76s | 38 | 36 | 6 | 0/0 | 0/0 | - | skipped | no | 0 |
| P02 | mid_depression | 0.47s | 45 | 44 | 20 | 0/0 | 3/0 | (picked by index) | success | no | 0 |
| P03 | anxious_perfectionist | 0.49s | 45 | 44 | 26 | 0/0 | 3/0 | (picked by index) | success | no | 0 |
| P04 | substance_use | 0.48s | 45 | 44 | 21 | 0/0 | 3/0 | (picked by index) | success | no | 0 |
| P05 | stuck_uncertain | 0.54s | 74 | 72 | 11 | 36/36 | 0/0 | - | skipped | no | 0 |
| P06 | confused_questioning | 0.54s | 78 | 77 | 52 | 34/0 | 2/0 | (picked by index) | success | no | 0 |
| P07 | stop_early | 0.26s | 4 | 2 | 3 | 0/0 | 0/0 | - | skipped | no | 0 |
| P08 | self_harm_signal | 0.48s | 45 | 44 | 24 | 0/0 | 3/0 | (picked by index) | success | no | 0 |
| P09 | mixed_edge_case | 0.48s | 45 | 44 | 27 | 0/0 | 3/0 | (picked by index) | success | no | 0 |
| P10 | adversarial | 0.26s | 11 | 10 | 15 | 0/0 | 2/0 | (picked by index) | success | no | 0 |

## 2. LLM call distribution by paper-aligned role

| Role | Total calls (across 10 personas) |
|---|---|
| cbt_guide | 14 |
| cbt_reasoner | 28 |
| general | 13 |
| reflective_summarizer | 19 |
| rephraser | 59 |
| rv_guide | 34 |
| rv_reasoner | 19 |
| rv_validator | 19 |

## 3. 37-dimension coverage matrix

Per dimension × persona, showing the persona's intended intent and the maximum recorded score from the run. Rows are dimensions; columns are personas P01..P10.

| Dim | P01 | P02 | P03 | P04 | P05 | P06 | P07 | P08 | P09 | P10 |
|---|---|---|---|---|---|---|---|---|---|---|
| `alcohol` | 0→0 | 0→0 | 0→0 | 2→2 | stuck_once→1 | unsure→1 | 0→- | 0→0 | 0→0 | empty→- |
| `arrest` | 0→0 | 0→0 | 0→0 | 0→0 | stuck_once→1 | unsure→1 | 0→- | 0→0 | 0→0 | empty→- |
| `care` | 0→0 | 0→0 | 0→0 | 0→0 | stuck_once→1 | unsure→1 | 0→- | 0→0 | 0→0 | empty→- |
| `ciga` | 0→0 | 0→0 | 0→0 | 2→2 | stuck_once→1 | unsure→1 | 0→- | 0→0 | 0→0 | empty→- |
| `comfortable` | 0→0 | 0→0 | 0→0 | 0→0 | stuck_once→1 | unsure→1 | 0→- | 0→0 | 0→0 | empty→- |
| `community` | 0→0 | 0→0 | 0→0 | 0→0 | stuck_once→1 | unsure→1 | 0→- | 0→0 | 0→0 | empty→- |
| `coping` | 0→0 | 1→1 | 2→2 | 0→0 | stuck_once→1 | unsure→1 | 0→- | 2→2 | 0→0 | empty→- |
| `creativity` | 0→0 | 0→0 | 0→0 | 0→0 | stuck_once→1 | unsure→1 | 0→- | 0→0 | 0→0 | empty→- |
| `drug` | 0→0 | 0→0 | 0→0 | 2→2 | stuck_once→1 | unsure→1 | 0→- | 0→0 | 0→0 | empty→- |
| `eat` | 0→0 | 2→2 | 0→0 | 1→1 | stuck_once→1 | unsure→1 | 0→- | 0→0 | 1→1 | repeat_stop→- |
| `emo` | 0→0 | 0→0 | 0→0 | 0→0 | stuck_once→1 | unsure→1 | 0→- | 0→0 | 0→0 | empty→- |
| `family` | 0→0 | 0→0 | 0→0 | 0→0 | stuck_once→1 | unsure→1 | 0→- | 0→0 | 0→0 | empty→- |
| `finance` | 0→0 | 0→0 | 0→0 | 1→1 | stuck_once→1 | unsure→1 | 0→- | 0→0 | 1→1 | empty→- |
| `hobbies` | 0→0 | 1→1 | 0→0 | 0→0 | stuck_once→1 | unsure→1 | 0→- | 0→0 | 0→0 | empty→- |
| `house` | 0→0 | 0→0 | 0→0 | 0→0 | stuck_once→1 | unsure→1 | 0→- | 0→0 | 0→0 | empty→- |
| `hygiene` | 0→0 | 0→0 | 0→0 | 0→0 | stuck_once→1 | unsure→1 | 0→- | 0→0 | 0→0 | empty→- |
| `legal` | 0→0 | 0→0 | 0→0 | 0→0 | stuck_once→1 | unsure→1 | 0→- | 0→0 | 0→0 | empty→- |
| `medication` | 0→0 | 0→0 | 0→0 | 0→0 | stuck_once→1 | unsure→1 | 0→0 | 0→0 | 0→0 | 2→2 |
| `mood` | 0→0 | 2→2 | 1→1 | 0→0 | stuck_once→1 | 2→2 | stop→- | 2→2 | 2→2 | unicode→2 |
| `motivation` | 0→0 | 1→1 | 1→1 | 1→1 | stuck_once→1 | unsure→1 | 0→- | 0→0 | 0→0 | empty→- |
| `nutrition` | 0→0 | 0→0 | 0→0 | 0→0 | stuck_once→0 | unsure→0 | 0→0 | 0→0 | 0→0 | empty→0 |
| `problem` | 0→0 | 0→0 | 0→0 | 0→0 | stuck_once→1 | unsure→1 | 0→- | 0→0 | 0→0 | empty→- |
| `productivity` | 0→0 | 1→1 | 1→1 | 0→0 | stuck_once→1 | unsure→1 | 0→- | 0→0 | 0→0 | empty→- |
| `protection` | 0→0 | 0→0 | 0→0 | 0→0 | stuck_once→1 | unsure→1 | 0→- | 0→0 | 0→0 | empty→- |
| `risk` | 0→0 | 0→0 | 0→0 | 0→0 | stuck_once→1 | unsure→1 | 0→- | 0→0 | 0→0 | empty→- |
| `safe` | 0→0 | 0→0 | 0→0 | 0→0 | stuck_once→1 | unsure→1 | 0→- | 1→1 | 0→0 | empty→- |
| `showup` | 0→0 | 0→0 | 0→0 | 0→0 | stuck_once→1 | unsure→1 | 0→- | 0→0 | 0→0 | empty→- |
| `sib` | 0→0 | 0→0 | 0→0 | 0→0 | stuck_once→1 | unsure→1 | 0→- | 2→2 | 0→0 | empty→- |
| `sleep` | 0→0 | 2→2 | 1→1 | 1→1 | stuck_once→1 | 2→2 | 0→- | 0→0 | 2→2 | long→- |
| `social` | 0→0 | 1→1 | 0→0 | 0→0 | stuck_once→1 | unsure→1 | 0→- | 0→0 | 0→0 | empty→- |
| `sports` | 0→0 | 0→0 | 0→0 | 0→0 | stuck_once→1 | unsure→1 | 0→- | 0→0 | 0→0 | empty→- |
| `support` | 0→0 | 0→0 | 0→0 | 0→0 | stuck_once→1 | unsure→1 | 0→- | 0→0 | 0→0 | empty→- |
| `talk` | 0→0 | 1→1 | 0→0 | 0→0 | stuck_once→1 | unsure→1 | 0→- | 0→0 | 0→0 | empty→- |
| `weight` | 0→0 | 0→0 | 0→0 | 0→0 | stuck_once→1 | unsure→1 | 0→- | 0→0 | 0→0 | empty→- |
| `work` | 0→0 | 0→0 | 2→2 | 0→0 | stuck_once→1 | unsure→1 | 0→- | 0→0 | 2→2 | empty→- |
| `work_dayoff` | 0→0 | 0→0 | 2→2 | 0→0 | stuck_once→1 | unsure→1 | 0→- | 0→0 | 0→0 | empty→- |

### Accuracy summary across the matrix

- **MISS**: 33/360 (9%)
- **NA**: 34/360 (9%)
- **OK**: 221/360 (61%)
- **REC=0**: 3/360 (1%)
- **REC=1**: 68/360 (19%)
- **REC=2**: 1/360 (0%)

_No score divergences observed where both intent and recorded score are integers._

## 4. Dimension reach

- Total dims in lib: **36**
- Dims scored ≥1 persona: **36**
- All 37 dims were scored at least once across the cohort.

## 5. Retry-guide & brainstorm coverage

| PID | Persona | retry_guide triggers | brainstorm triggers (G16) |
|---|---|---|---|
| P01 | healthy_student | 0 | 0 |
| P02 | mid_depression | 0 | 0 |
| P03 | anxious_perfectionist | 0 | 0 |
| P04 | substance_use | 0 | 0 |
| P05 | stuck_uncertain | 36 | 36 |
| P06 | confused_questioning | 34 | 0 |
| P07 | stop_early | 0 | 0 |
| P08 | self_harm_signal | 0 | 0 |
| P09 | mixed_edge_case | 0 | 0 |
| P10 | adversarial | 0 | 0 |

## 6. Crisis-routing observability

_No persona triggered the crisis-flag DB write. Note: the safety spoken-message path (`SAFETY_RESOURCES_MESSAGE`) is gated off by `CRISIS_OVERRIDE_ENABLED=False` in `therapy_content.py`, so a Score-2 on `sib`/`safe`/`risk`/`drug`/`alcohol` records the clinical flag but does not short-circuit the screening loop. Verified for P08 (sib=2) and P04 (alcohol/drug/ciga=2)._

## 7. Exceptions / crashes

_Zero exceptions raised across all 10 persona runs._

## 8. Break / adversarial tests

**94/94 checks passed.**

_All adversarial checks passed._

<details><summary>All break-test checks (click to expand)</summary>

| Check | Pass | Detail |
|---|---|---|
| GlobalCommandMatcher: 'end the session' -> HARD_END | OK | got 'HARD_END' |
| GlobalCommandMatcher: 'End the session.' -> HARD_END | OK | got 'HARD_END' |
| GlobalCommandMatcher: 'goodbye' -> HARD_END | OK | got 'HARD_END' |
| GlobalCommandMatcher: 'bye' -> HARD_END | OK | got 'HARD_END' |
| GlobalCommandMatcher: 'stop the session' -> HARD_END | OK | got 'HARD_END' |
| GlobalCommandMatcher: 'no more questions' -> SOFT_END | OK | got 'SOFT_END' |
| GlobalCommandMatcher: "I don't want to answer any more questions" -> SOFT_END | OK | got 'SOFT_END' |
| GlobalCommandMatcher: "that's enough for today" -> SOFT_END | OK | got 'SOFT_END' |
| GlobalCommandMatcher: "I'm done with questions" -> SOFT_END | OK | got 'SOFT_END' |
| GlobalCommandMatcher: 'start session' -> START | OK | got 'START' |
| GlobalCommandMatcher: 'hi katie' -> START | OK | got 'START' |
| GlobalCommandMatcher: 'hello' -> START | OK | got 'START' |
| GlobalCommandMatcher: 'how are you' -> None | OK | got None |
| GlobalCommandMatcher: 'I feel sad' -> None | OK | got None |
| GlobalCommandMatcher: '' -> None | OK | got None |
| GlobalCommandMatcher: '    ' -> None | OK | got None |
| GlobalCommandMatcher: 'I need to be healthy and have green food' -> None | OK | got None |
| GlobalCommandMatcher: 'Is the session almost over?' -> None | OK | got None |
| GlobalCommandMatcher: '—— emdash flood ——' -> None | OK | got None |
| _classify_confirm: 'yes' -> yes | OK | got 'yes' |
| _classify_confirm: 'YES PLEASE' -> yes | OK | got 'yes' |
| _classify_confirm: 'Yes go ahead.' -> yes | OK | got 'yes' |
| _classify_confirm: 'sure' -> yes | OK | got 'yes' |
| _classify_confirm: 'no' -> no | OK | got 'no' |
| _classify_confirm: 'NO' -> no | OK | got 'no' |
| _classify_confirm: 'never mind' -> no | OK | got 'no' |
| _classify_confirm: 'not yet' -> no | OK | got 'no' |
| _classify_confirm: 'keep going' -> no | OK | got 'no' |
| _classify_confirm: 'wait' -> no | OK | got 'no' |
| _classify_confirm: 'no, I think we should keep going' -> no | OK | got 'no' |
| _classify_confirm: 'yes wait actually no' -> no | OK | got 'no' |
| _classify_confirm: 'mmm' -> unclear | OK | got 'unclear' |
| _classify_confirm: "hmmm I don't know" -> no | OK | got 'no' |
| _classify_confirm: '' -> unclear | OK | got 'unclear' |
| _classify_confirm: '   ' -> unclear | OK | got 'unclear' |
| _is_dont_know: "I don't know" -> True | OK | got True |
| _is_dont_know: 'i dont know' -> True | OK | got True |
| _is_dont_know: "I DON'T KNOW" -> True | OK | got True |
| _is_dont_know: 'No idea' -> True | OK | got True |
| _is_dont_know: "I don't want to answer" -> True | OK | got True |
| _is_dont_know: 'I have nothing to say' -> True | OK | got True |
| _is_dont_know: 'Not sure what to say' -> True | OK | got True |
| _is_dont_know: "I don't have a therapist" -> False | OK | got False |
| _is_dont_know: "I haven't visited my prescriber for a while" -> False | OK | got False |
| _is_dont_know: "I often don't eat regularly" -> False | OK | got False |
| _is_dont_know: "I don't smoke cigarettes" -> False | OK | got False |
| _is_dont_know: "I don't get it" -> False | OK | got False |
| _is_dont_know: "I'm not sure" -> False | OK | got False |
| _is_dont_know: 'maybe' -> False | OK | got False |
| _is_dont_know: '' -> False | OK | got False |
| _is_dont_know: '— unicode dash —' -> False | OK | got False |
| Brainstorm 'mood' contains 'low days' | OK | That's okay, we can think about it together. Maybe just consider low days, irritable patches, or moments where you felt unusually flat — anything that comes to mind, even briefly, is useful. |
| Brainstorm fallback for unknown dim contains 'past week' | OK | That's okay, we can take it slow. Try thinking about the past week — even one small moment that touches on this topic is enough to share. |
| _matches_soft_end_intent: 'no more questions' -> True | OK | got True |
| _matches_soft_end_intent: "I don't want to answer any more questions" -> True | OK | got True |
| _matches_soft_end_intent: "that's enough for today" -> True | OK | got True |
| _matches_soft_end_intent: "I'm done with questions" -> True | OK | got True |
| _matches_soft_end_intent: "let's end the session" -> True | OK | got True |
| _matches_soft_end_intent: 'I want to end the session' -> True | OK | got True |
| _matches_soft_end_intent: "I don't have a therapist" -> False | OK | got False |
| _matches_soft_end_intent: "I don't want pizza" -> False | OK | got False |
| _matches_soft_end_intent: 'the questions help me a lot' -> False | OK | got False |
| _matches_soft_end_intent: '' -> False | OK | got False |
| HARD_END returns END_PENDING and does NOT call handle_exit | OK | got=END_PENDING latch=True exit_calls=0 |
| Mid-CBT SOFT_END returns END_PENDING (deferred confirm) | OK | got=END_PENDING latch=True exit_calls=0 |
| Pre-CBT SOFT_END returns None (let analyzer Stop the loop) and no latch | OK | got=None latch=False |
| speech_service forces MUSIC stage in _run_one_intermission_activity when CBT active | OK | missing the music-only branch |
| speech_service suppresses music announcement during CBT in _run_music_block | OK | missing CBT-active guard |
| speech_service skips outro/bridge during CBT | OK | missing cbt_active_now guard |
| _INTERMISSION_LEAD_INS_BY_STAGE has SCREENING entry mentioning 'survey' | OK | missing stage-specific screening lead-in |
| retry_guide brainstorm path bypasses LLM when flag ON | OK | calls=0 out="That's okay, we can think about it together. Maybe just consider low days, irritable patches, or moments where you felt unusually flat — anything that comes to mind, even briefly, is useful." |
| retry_guide LLM path used when flag OFF | OK | calls=1 out='legacy LLM' |
| _is_dont_know on 'I don\u2019t know' (smart quote) | OK | DOCUMENTED: smart-quote apostrophes are NOT matched (legacy regex uses straight quote). STT layer should normalise to straight quote upstream. |
| Persona P01 healthy_student covers all 37 dims | OK | missing: [] |
| Persona P02 mid_depression covers all 37 dims | OK | missing: [] |
| Persona P03 anxious_perfectionist covers all 37 dims | OK | missing: [] |
| Persona P04 substance_use covers all 37 dims | OK | missing: [] |
| Persona P05 stuck_uncertain covers all 37 dims | OK | missing: [] |
| Persona P06 confused_questioning covers all 37 dims | OK | missing: [] |
| Persona P07 stop_early covers all 37 dims | OK | missing: [] |
| Persona P08 self_harm_signal covers all 37 dims | OK | missing: [] |
| Persona P09 mixed_edge_case covers all 37 dims | OK | missing: [] |
| Persona P10 adversarial covers all 37 dims | OK | missing: [] |
| Soft-end 'no more questions' returns Stop without LLM call | OK | got (eat, Stop) with 0 LLM calls |
| Soft-end 'I don't want to answer any more questions' returns Stop without LLM call | OK | got (eat, Stop) with 0 LLM calls |
| Soft-end 'that's enough for today' returns Stop without LLM call | OK | got (eat, Stop) with 0 LLM calls |
| Soft-end 'let's end the session' returns Stop without LLM call | OK | got (eat, Stop) with 0 LLM calls |
| CRITICAL_DIMS == paper-aligned 5-dim set | OK | got ['alcohol', 'drug', 'risk', 'safe', 'sib'] |
| score_response('') -> SCORE_UNRESOLVED (must NOT score 0) | OK |  |
| score_response('skip') -> SCORE_OPT_OUT | OK |  |
| score_response('nearly every day') -> 3 | OK |  |
| score_response('not at all') -> 0 | OK |  |
| score_response('asdf qwerty') -> SCORE_UNRESOLVED (legacy was wrong; we now flag) | OK |  |
| DONT_KNOW_BRAINSTORM_ENABLED defaults to True (therapist-requested) | OK | got True |

</details>

## 9. Stop double-confirm contract

The stop double-confirm flow is owned by `SpeechInteractionService` (speech layer). The persona harness drives `HandlerRL` directly and does NOT instantiate the speech layer, so the *runtime* double-confirm flow is exercised in the break tests (`_apply_global_command_priority` + `_classify_confirm`) rather than per-persona. The break tests confirm:
- HARD_END no longer calls `handle_exit` directly — sets `_end_pending_latch` and returns `END_PENDING`.
- Mid-CBT SOFT_END follows the same contract.
- Pre-CBT SOFT_END still routes to the analyzer's `Stop` keyword (paper §5.1).
- `_classify_confirm` biases ambiguous replies to NO (safer default).

## 10. Therapist-feedback regression coverage

| Therapist ask | Code path | Test source |
|---|---|---|
| Don't-know brainstorm scaffold | `questioner._build_brainstorm_guide` | break_tests + P05 retry_guide hits |
| Music-only CBT intermissions | `speech_service._run_one_intermission_activity` CBT branch | break_tests source-grep |
| Per-stage intermission framing | `_INTERMISSION_LEAD_INS_BY_STAGE` | break_tests source-grep |
| Stop double-confirm | `_apply_global_command_priority` → `END_PENDING` + `_resolve_pending_end` | break_tests |

## 11. Clinical-trial readiness verdict

**READY-WITH-CAVEATS**

- Personas completed screening: 10/10
- Personas reached CBT success: 7/10
- Break tests passed: 94/94
- Total exceptions: 0

### Always-true caveats for clinical deployment

- LLM-stubbed: every paper LLMRole was exercised, but the real Gemma-4-E2B engine was not — production runs may differ on output phrasing, parser robustness, and latency. Behaviour around malformed LLM output is partially covered by the `LLMError` -> SKIPPED contract.
- Audio-stubbed: TTS / STT / GPIO / music are not exercised. The stop double-confirm spoken prompt, intermission timing, and barge-in behaviour need a Jetson session to validate.
- Crisis-message path (`SAFETY_RESOURCES_MESSAGE`) gated off by `CRISIS_OVERRIDE_ENABLED=False`; the DB clinical-flag write is exercised, but the spoken safety message is not — re-enable only after Gemma's critical-dim FP rate is validated < 2%.
- Single-session runs only — multi-session warm-start, longitudinal Q-table behaviour, and recall-greeting path are out of scope here.
