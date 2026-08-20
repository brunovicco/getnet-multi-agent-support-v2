"""Gradio Blocks UI: calls the same ChatApplicationService instance as the HTTP API.

No internal HTTP hop — the UI event handler calls `ChatApplicationService.handle` directly, the
same call the `POST /chat` route makes. Visual identity (accent red, wordmark) is inspired by the
dominant color observed on the official https://www.getnet.net/en page (icon/asset naming and
overall palette read as red-on-white with dark text) — no brand-guideline document was available,
so this is a best-effort, publicly-observable approximation, not an exact reproduction.
"""

from typing import cast

import gradio as gr

from getnet_support.application.chat_service import ChatApplicationService
from getnet_support.domain.models import Locale, Market

_USER_CHOICES = ["cliente1988", "cliente2001", "cliente_desconhecido"]
_LOCALE_CHOICES = [("Português (BR)", Locale.PT_BR.value), ("English", Locale.EN.value)]
_MARKET_CHOICES = [Market.BR.value, Market.GLOBAL.value]

_CSS = """
.getnet-header { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.25rem; }
.getnet-wordmark {
    font-weight: 800; font-size: 1.6rem; color: #ffffff; background: #d6001c;
    padding: 0.2rem 0.6rem; border-radius: 0.4rem; letter-spacing: -0.02em;
}
.getnet-subtitle { color: var(--body-text-color-subdued); font-size: 0.95rem; }
.getnet-panel {
    border: 1px solid var(--border-color-primary); border-radius: 0.5rem; padding: 0.75rem;
}
"""

_EMPTY_SOURCES_MD = "_(sem fontes / no sources)_"

# Gradio 6 moved theme/css from the Blocks constructor to mount_gradio_app()/launch(); exported
# so entrypoints/http.py can pass them when mounting this UI on the FastAPI app.
THEME = gr.themes.Soft(primary_hue="red", secondary_hue="slate", neutral_hue="slate")
CSS = _CSS


def build_blocks(chat_service: ChatApplicationService) -> gr.Blocks:
    """Build the Gradio Blocks UI bound to one ChatApplicationService instance."""

    async def handle_submit(
        message: str, user_id: str, locale_value: str, market_value: str
    ) -> tuple[str, str, str]:
        if not message.strip():
            return "", _EMPTY_SOURCES_MD, ""
        try:
            result = await chat_service.handle(
                message=message,
                user_id=user_id,
                market=Market(market_value),
                locale=Locale(locale_value),
            )
        except Exception:
            error_md = (
                "⚠️ **Erro inesperado / Unexpected error.** Tente novamente ou "
                "acione o suporte humano. / Please retry or contact human support."
            )
            return "", _EMPTY_SOURCES_MD, error_md

        sources_md = (
            "\n".join(
                f"- [{source.title}]({source.url}) — {source.market.value}, "
                f"atualizado em / updated {source.retrieved_at}"
                + (" ⚠️ dado volátil / volatile data" if source.volatility == "high" else "")
                for source in result.sources
            )
            or _EMPTY_SOURCES_MD
        )
        handoff_badge = (
            "🔴 **Handoff necessário / Human handoff required**"
            if result.handoff_required
            else "🟢 Resolvido pelo assistente / Resolved by the assistant"
        )
        trace_md = (
            f"{handoff_badge}  \n"
            f"**Rota / Route:** {result.route.value}  \n"
            f"**Agentes / Agents:** {' → '.join(agent.value for agent in result.agents)}  \n"
            f"**Tools:** {', '.join(result.tools) or '—'}  \n"
            f"**Latência / Latency:** {result.latency_ms} ms  \n"
            f"**Trace ID:** `{result.trace_id}`"
        )
        return result.answer, sources_md, trace_md

    def clear_conversation() -> tuple[str, str, str, str]:
        return "", "", _EMPTY_SOURCES_MD, ""

    with gr.Blocks(title="Getnet AI Support", analytics_enabled=False) as blocks:
        with gr.Row(elem_classes="getnet-header"):
            gr.HTML('<span class="getnet-wordmark">Getnet</span>')
            gr.Markdown(
                '<span class="getnet-subtitle">AI Support — Router → Knowledge / '
                "Customer Support / Escalation</span>"
            )
        with gr.Row():
            user_id = gr.Dropdown(
                choices=_USER_CHOICES,
                value="cliente1988",
                label="Usuário / User ID",
                allow_custom_value=True,
            )
            locale = gr.Dropdown(
                choices=_LOCALE_CHOICES, value=Locale.PT_BR.value, label="Idioma / Language"
            )
            market = gr.Dropdown(
                choices=_MARKET_CHOICES, value=Market.BR.value, label="Mercado / Market"
            )

        message = gr.Textbox(
            label="Mensagem / Message",
            placeholder="Ex.: Minha maquininha não conecta à internet",
            lines=2,
        )
        with gr.Row():
            send = gr.Button("Enviar / Send", variant="primary")
            clear = gr.Button("Nova conversa / New conversation")

        answer = gr.Textbox(label="Resposta / Answer", lines=6, interactive=False)
        with gr.Row():
            sources = gr.Markdown(
                label="Fontes / Sources", value=_EMPTY_SOURCES_MD, elem_classes="getnet-panel"
            )
            trace = gr.Markdown(label="Execução / Execution", value="", elem_classes="getnet-panel")

        send.click(
            handle_submit,
            inputs=[message, user_id, locale, market],
            outputs=[answer, sources, trace],
        )
        clear.click(clear_conversation, outputs=[message, answer, sources, trace])

    return cast(gr.Blocks, blocks)
