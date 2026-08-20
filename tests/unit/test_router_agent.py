"""Behavior tests for the deterministic Router Agent."""

from getnet_support.application.router_agent import RouterAgent
from getnet_support.domain.models import Route


def test_routes_product_question_to_knowledge() -> None:
    router = RouterAgent()
    route = router.route("Qual a diferença entre Get Clássica e Get Smart?")
    assert route is Route.KNOWLEDGE


def test_routes_terminal_problem_to_customer_support() -> None:
    router = RouterAgent()
    route = router.route("Minha maquininha não conecta à internet.")
    assert route is Route.CUSTOMER_SUPPORT


def test_routes_settlement_question_to_customer_support() -> None:
    router = RouterAgent()
    route = router.route("When will the money from yesterday's sales be deposited?")
    assert route is Route.CUSTOMER_SUPPORT


def test_routes_unsupported_financial_operation_to_escalation() -> None:
    router = RouterAgent()
    route = router.route("Quero fazer um estorno da minha última venda.")
    assert route is Route.ESCALATION


def test_routes_explicit_human_request_to_escalation() -> None:
    router = RouterAgent()
    route = router.route("Eu quero falar com um atendente humano agora.")
    assert route is Route.ESCALATION
