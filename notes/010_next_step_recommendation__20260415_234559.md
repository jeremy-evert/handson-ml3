# 010 Next Step Recommendation

Chosen next move: `environment diagnosis prompt`

Why it should happen next:
The V1 runner candidate already produces the intended execution record shape closely enough, and the validation run failed during Codex session initialization with a read-only filesystem error. That makes environment diagnosis the smallest step that can unblock meaningful validation without broadening the build.

What it should produce:
A short diagnosis artifact that identifies where session startup is trying to write, whether the failure reproduces independently of `tools/codex/run_prompt.py`, and one bounded follow-up action.

What should explicitly wait:
Runner polish, review write-back helpers, and any broader workflow or runner build-out.
