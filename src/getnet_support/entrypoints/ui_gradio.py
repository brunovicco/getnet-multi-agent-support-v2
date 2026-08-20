"""Gradio Blocks UI: calls the same ChatApplicationService instance as the HTTP API.

No internal HTTP hop — the UI event handler calls `ChatApplicationService.handle` directly, the
same call the `POST /chat` route makes.
"""

from typing import cast

import gradio as gr

from getnet_support.application.chat_service import ChatApplicationService
from getnet_support.domain.models import Locale, Market

_USER_CHOICES = ["cliente1988", "cliente2001", "cliente_desconhecido"]
_LOCALE_CHOICES = [("Português (BR)", Locale.PT_BR.value), ("English", Locale.EN.value)]
_MARKET_CHOICES = [Market.BR.value, Market.GLOBAL.value]


def build_blocks(chat_service: ChatApplicationService) -> gr.Blocks:
    """Build the Gradio Blocks UI bound to one ChatApplicationService instance."""

    async def handle_submit(
        message: str, user_id: str, locale_value: str, market_value: str
    ) -> tuple[str, str, str]:
        if not message.strip():
            return "", "", ""
        result = await chat_service.handle(
            message=message,
            user_id=user_id,
            market=Market(market_value),
            locale=Locale(locale_value),
        )
        sources_md = (
            "\n".join(
                f"- [{source.title}]({source.url}) — {source.market.value}, "
                f"atualizado em {source.retrieved_at}"
                + (" ⚠️ dado volátil" if source.volatility == "high" else "")
                for source in result.sources
            )
            or "_(sem fontes)_"
        )
        trace_md = (
            f"**Rota:** {result.route.value}  \n"
            f"**Agentes:** {' → '.join(agent.value for agent in result.agents)}  \n"
            f"**Tools:** {', '.join(result.tools) or '—'}  \n"
            f"**Handoff necessário:** {'sim' if result.handoff_required else 'não'}  \n"
            f"**Latência:** {result.latency_ms} ms  \n"
            f"**Trace ID:** `{result.trace_id}`"
        )
        return result.answer, sources_md, trace_md

    def clear_conversation() -> tuple[str, str, str, str]:
        return "", "", "", ""

    with gr.Blocks(title="Getnet AI Support", analytics_enabled=False) as blocks:
        gr.Markdown(
            "# Getnet AI Support\n"
            "Multi-agent: **Router → Knowledge / Customer Support / Escalation**"
        )
        with gr.Row():
            user_id = gr.Dropdown(
                choices=_USER_CHOICES,
                value="cliente1988",
                label="Usuário (user_id)",
                allow_custom_value=True,
            )
            locale = gr.Dropdown(choices=_LOCALE_CHOICES, value=Locale.PT_BR.value, label="Idioma")
            market = gr.Dropdown(choices=_MARKET_CHOICES, value=Market.BR.value, label="Mercado")

        message = gr.Textbox(
            label="Mensagem",
            placeholder="Ex.: Minha maquininha não conecta à internet",
            lines=2,
        )
        with gr.Row():
            send = gr.Button("Enviar", variant="primary")
            clear = gr.Button("Nova conversa")

        answer = gr.Textbox(label="Resposta", lines=6, interactive=False)
        with gr.Row():
            sources = gr.Markdown(label="Fontes", value="_(sem fontes)_")
            trace = gr.Markdown(label="Execução", value="")

        send.click(
            handle_submit,
            inputs=[message, user_id, locale, market],
            outputs=[answer, sources, trace],
        )
        clear.click(clear_conversation, outputs=[message, answer, sources, trace])

    return cast(gr.Blocks, blocks)
