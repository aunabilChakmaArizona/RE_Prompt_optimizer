# OpenAI PRM800K grader

`grader.py`, `math_normalize.py`, and `LICENSE` were copied from:

- Repository: `https://github.com/openai/prm800k`
- Revision: `7ecc794703b2877f63226f2477a49b34f9b25163`
- Original directory: `prm800k/grading/`

The only source adaptation is changing `from grading import math_normalize` to
the package-relative import `from . import math_normalize`.

Original SHA-256 checksums before that import adaptation:

- `grader.py`: `9e8bbb6f504ee0d8068e1eca031d78174f1c925cfec391996fdf45c21bb016a9`
- `math_normalize.py`: `13998bf8c35bdc8a76868c84e1704a78567e2552557fda69027cc186568d3a4d`
- `LICENSE`: `dcf7927ec000e0bea021baf556c82ad7ac2e96cdd8285a1c8b2a7014e8300c6d`
