# Task 8: Fix lint gate fallback

Goal
Make the merge lint/test gate reliable when the 'uv' command is not available in the environment.

Behavior
- When running lint/tests for merge approval, attempt to use the configured runner.
- If 'uv' is not available, fall back to a standard, available command to run lint/tests (or a documented equivalent), and surface a clear message indicating the fallback was used.
- If neither the primary runner nor fallback can run, fail with a clear, actionable error that points to the missing dependency/setup.

Voice test plan
- Create or use a trivial change (like a small docs-only update) and run the merge gate in an environment without 'uv'. Confirm it uses the fallback and passes if lint/tests pass.
- Confirm that in a properly configured environment, the primary runner path is used.
- Confirm that when both primary and fallback are missing, the error message explains what to install or configure.

Out of scope
- Changing lint rules or test suites.
- Adding new dependencies beyond what is needed for the fallback mechanism.

## Revision 2 — 2026-09-02

Merge approval still fails when 'uv' is missing; the approval lint/test gate invoked 'uv' directly and did not use the fallback, so the fix did not apply to the merge approval path.
