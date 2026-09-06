# Task 14: Transcriber vocabulary

## Goal
Her transcripts spell the house's names wrong ("living room apple tv", playlist names, artists). Clicky hands its transcriber a keyterms list. The installed openai SDK's AudioTranscription has a prompt field (verified). See docs/CLICKY-REVIEW-2026-09-04.md, item 5.

## Behavior
- Build a vocabulary string per session: her name, the owner's name, Home Assistant area and device names, media players, the owner's playlist names and recent artists (from the music library cache if one exists, otherwise skip), open task titles. Deduplicated, capped at about 1,000 characters, most useful names first.
- Set it as audio.input.transcription.prompt in the session config (verify the exact nesting against the installed SDK types); the model still hears audio directly, this only improves the written transcript.
- The transcript feeds the stop-phrase match, the journal, reflection and the session log, so all of those get cleaner names.
- Tests: the session config carries the prompt; it is capped; missing sources are skipped without error.

## Voice test
Say a playlist name and an artist you own, then "Alexa, what did I just ask?" and check the journal spelling with journal_search.

## Out of scope
Changing what the model itself understands.
