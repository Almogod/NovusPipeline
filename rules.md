# NovusPipeline Refactoring Rules & Constraints

- **Determinism**: Code transformations must strictly preserve logical parity.
- **Strict Typing**: Add type hints (Python 3.10+ syntax / TypeScript strict mode).
- **Security**: Mandatory workspace path boundary checks to prevent arbitrary system file reads.
- **Local Testing**: Verify changes via local test suite execution after every modernization step.
- **API Stability**: Never remove or rename symbols that other modules import (see `get_file_context`); the pipeline rolls back changes that do.
- **Scoped Commits**: Commit only the files a modernization changed, never unrelated workspace changes.
