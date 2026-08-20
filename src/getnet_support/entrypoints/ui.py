"""Gradio UI, mounted in the same process as the API (REQ-04).

The UI calls :class:`ChatApplicationService` directly — never over HTTP — and
shows, per message, the execution panel required by REQ-27: route, agents,
tools, clickable sources, handoff, and latency.
"""

import gradio as gr

from getnet_support.application.chat_service import ChatApplicationService
from getnet_support.application.dto import ChatResult

_MARKET_CHOICES = ["", "BR", "GLOBAL"]
_LOCALE_CHOICES = ["", "pt-BR", "en"]

# Getnet's brand orange, approximated (P1.2 — this UI has no network access to Getnet's
# official brand guidelines, so treat this as "in the spirit of," not a verified hex).
_GETNET_ORANGE = "#FF6A13"
_GETNET_ORANGE_DARK = "#C94F00"

CUSTOM_CSS = f"""
.getnet-header {{
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 1rem 1.25rem;
    margin-bottom: 0.5rem;
    border-radius: 12px;
    background: linear-gradient(135deg, {_GETNET_ORANGE} 0%, {_GETNET_ORANGE_DARK} 100%);
    color: white;
}}
.getnet-badge {{
    display: flex;
    align-items: center;
    justify-content: center;
    width: 2.25rem;
    height: 2.25rem;
    border-radius: 8px;
    background: white;
    color: {_GETNET_ORANGE_DARK};
    font-weight: 800;
    font-size: 1.1rem;
    flex-shrink: 0;
}}
.getnet-title {{
    font-weight: 700;
    font-size: 1.15rem;
    line-height: 1.2;
}}
.getnet-subtitle {{
    font-size: 0.85rem;
    opacity: 0.9;
}}
.getnet-footer {{
    text-align: center;
    font-size: 0.75rem;
    opacity: 0.6;
    margin-top: 0.5rem;
}}
"""

_HEADER_HTML = """
<div class="getnet-header">
    <div class="getnet-badge">G</div>
    <div>
        <div class="getnet-title">Getnet Multi-Agent Support</div>
        <div class="getnet-subtitle">Router · Knowledge (RAG) · Customer Support · Escalation</div>
    </div>
</div>
"""

_FOOTER_HTML = (
    '<div class="getnet-footer">Multi-agent reference implementation for Getnet — '
    "not an official Getnet product.</div>"
)


def build_theme() -> gr.Theme:
    """Build the Getnet-orange theme.

    Applied at mount time (`gr.mount_gradio_app(..., theme=..., css=...)` in
    `entrypoints/http.py`), not on the `Blocks` constructor — Gradio 6 moved
    theme/css there for apps that are mounted rather than `.launch()`-ed.
    """
    return gr.themes.Soft(primary_hue="orange", neutral_hue="slate")


def _render_panel(result: ChatResult) -> str:
    """Render the REQ-27 execution panel as Markdown."""
    lines = [
        f"**Route:** `{result.route.value}`",
        f"**Agents:** {', '.join(result.agents) or '—'}",
        f"**Tools:** {', '.join(result.tools) or '—'}",
        f"**Grounding:** `{result.grounding.value}`",
        f"**Handoff required:** {'yes' if result.handoff_required else 'no'}",
        f"**Web search attempted:** {'yes' if result.web_search_attempted else 'no'}",
        f"**Decision source:** `{result.decision_source.value}` "
        f"({result.classifier_latency_ms} ms)",
        f"**Latency:** {result.latency_ms} ms",
        f"**Trace id:** `{result.trace_id}`",
    ]
    return "\n\n".join(lines)


def _render_sources(result: ChatResult) -> str:
    """Render clickable sources as Markdown (REQ-27)."""
    if not result.sources:
        return "_No sources cited._"
    lines = []
    for source in result.sources:
        market = f", {source.market.value}" if source.market else ""
        lines.append(
            f"- [{source.title}]({source.url}) — `{source.origin.value}`, "
            f"volatility `{source.volatility.value}`{market}"
        )
    return "\n".join(lines)


def build_ui(chat_service: ChatApplicationService) -> gr.Blocks:
    """Build the Gradio chat UI backed by the given application service."""

    def respond(
        message: str,
        history: list[dict[str, str]],
        user_id: str,
        market: str,
        locale: str,
    ) -> tuple[list[dict[str, str]], str, str, str]:
        result = chat_service.handle(
            message=message,
            user_id=user_id or "cliente1988",
            market=market or None,
            locale=locale or None,
        )
        updated_history = [
            *history,
            {"role": "user", "content": message},
            {"role": "assistant", "content": result.answer},
        ]
        return updated_history, "", _render_panel(result), _render_sources(result)

    demo = gr.Blocks(title="Getnet Support")
    with demo:
        gr.HTML(_HEADER_HTML)
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Chat", height=420)
                message_box = gr.Textbox(
                    label="Message", placeholder="Ask about Getnet products or your account..."
                )
                with gr.Row():
                    user_id_box = gr.Textbox(label="user_id", value="cliente1988")
                    market_box = gr.Dropdown(label="market", choices=_MARKET_CHOICES, value="")
                    locale_box = gr.Dropdown(label="locale", choices=_LOCALE_CHOICES, value="")
                send_button = gr.Button("Send", variant="primary")
            with gr.Column(scale=1):
                gr.Markdown("### Execution")
                execution_panel = gr.Markdown()
                gr.Markdown("### Sources")
                sources_panel = gr.Markdown()
        gr.HTML(_FOOTER_HTML)

        inputs = [message_box, chatbot, user_id_box, market_box, locale_box]
        outputs = [chatbot, message_box, execution_panel, sources_panel]
        send_button.click(respond, inputs=inputs, outputs=outputs)
        message_box.submit(respond, inputs=inputs, outputs=outputs)

    return demo
