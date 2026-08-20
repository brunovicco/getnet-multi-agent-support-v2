"""getnet-multi-agent-support-v2 package."""

import os

# Must run before `import gradio` anywhere in this package: disables Gradio's
# phone-home version/usage check so tests and offline runs never make an
# unexpected outbound call (security-privacy.md: least privilege, no
# unexpected external calls).
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
