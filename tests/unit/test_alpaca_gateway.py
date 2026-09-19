"""Unit tests for AlpacaExecutionGateway implementing the ExecutionGateway protocol."""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from quant.execution.alpaca_gateway import AlpacaExecutionGateway
from quant.execution.gateway import ExecutionGateway
from quant.execution.models import (
    ERR_GW_DISCONNECTED,
    ERR_GW_INSUFFICIENT_MARGIN,
    ERR_GW_RATE_LIMIT_EXCEEDED,
    GatewayDisconnectedException,
    GatewayError,
    InsufficientMarginException,
    Order,
    OrderSide,
    OrderState,
    OrderType,
    TimeInForce,
)


def _mock_transport(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


@pytest.fixture
def sample_buy_order() -> Order:
    return Order(
        cl_ord_id="ord-alpaca-001",
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=10.0,
        time_in_force=TimeInForce.DAY,
    )


@pytest.fixture
def sample_limit_order() -> Order:
    return Order(
        cl_ord_id="ord-alpaca-002",
        symbol="NVDA",
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        quantity=5.0,
        price=130.0,
        time_in_force=TimeInForce.DAY,
    )


class TestAlpacaGatewayProtocol:
    """Verify structural subtyping and protocol compliance."""

    def test_implements_execution_gateway_protocol(self) -> None:
        gw = AlpacaExecutionGateway(
            api_key="test-key",
            secret_key="test-secret",
            base_url="https://paper-api.alpaca.markets",
        )
        assert isinstance(gw, ExecutionGateway)
        assert not gw.is_connected


class TestAlpacaConnectionLifecycle:
    """Verify session connection, authentication, and disconnection."""

    @pytest.mark.asyncio
    async def test_connect_success(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.headers["APCA-API-KEY-ID"] == "test-key"
            assert request.headers["APCA-API-SECRET-KEY"] == "test-secret"
            if request.url.path == "/v2/account":
                return httpx.Response(200, json={"status": "ACTIVE", "cash": "100000.0"})
            return httpx.Response(404)

        transport = _mock_transport(handler)
        gw = AlpacaExecutionGateway(
            api_key="test-key",
            secret_key="test-secret",
            transport=transport,
        )
        await gw.connect()
        assert gw.is_connected
        await gw.disconnect()
        assert not gw.is_connected

    @pytest.mark.asyncio
    async def test_connect_auth_failure(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"message": "forbidden"})

        transport = _mock_transport(handler)
        gw = AlpacaExecutionGateway(
            api_key="invalid-key",
            secret_key="invalid-secret",
            transport=transport,
        )
        with pytest.raises(GatewayDisconnectedException) as exc_info:
            await gw.connect()
        assert exc_info.value.code == ERR_GW_DISCONNECTED
        assert not gw.is_connected

    @pytest.mark.asyncio
    async def test_action_while_disconnected_raises_exception(
        self, sample_buy_order: Order
    ) -> None:
        gw = AlpacaExecutionGateway(api_key="key", secret_key="sec")
        assert not gw.is_connected
        with pytest.raises(GatewayDisconnectedException) as exc_info:
            await gw.submit_order(sample_buy_order)
        assert exc_info.value.code == ERR_GW_DISCONNECTED


class TestAlpacaOrderSubmission:
    """Verify order mapping and execution report parsing."""

    @pytest.mark.asyncio
    async def test_submit_market_order_new(self, sample_buy_order: Order) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v2/account":
                return httpx.Response(200, json={"status": "ACTIVE"})
            if request.url.path == "/v2/orders" and request.method == "POST":
                data = request.read().decode()
                assert "AAPL" in data
                assert "ord-alpaca-001" in data
                return httpx.Response(
                    200,
                    json={
                        "id": "alpaca-uuid-1",
                        "client_order_id": "ord-alpaca-001",
                        "status": "new",
                        "symbol": "AAPL",
                        "qty": "10",
                        "filled_qty": "0",
                        "filled_avg_price": None,
                    },
                )
            return httpx.Response(404)

        transport = _mock_transport(handler)
        gw = AlpacaExecutionGateway("key", "sec", transport=transport)
        await gw.connect()

        report = await gw.submit_order(sample_buy_order)
        assert report.cl_ord_id == "ord-alpaca-001"
        assert report.exec_type == OrderState.NEW
        assert report.cum_quantity == 0.0
        assert report.leaves_quantity == 10.0
        assert report.last_price == 0.0

    @pytest.mark.asyncio
    async def test_submit_limit_order_immediate_fill(self, sample_limit_order: Order) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v2/account":
                return httpx.Response(200, json={"status": "ACTIVE"})
            if request.url.path == "/v2/orders":
                return httpx.Response(
                    200,
                    json={
                        "id": "alpaca-uuid-2",
                        "client_order_id": "ord-alpaca-002",
                        "status": "filled",
                        "symbol": "NVDA",
                        "qty": "5",
                        "filled_qty": "5",
                        "filled_avg_price": "130.5",
                    },
                )
            return httpx.Response(404)

        transport = _mock_transport(handler)
        gw = AlpacaExecutionGateway("key", "sec", transport=transport)
        await gw.connect()

        report = await gw.submit_order(sample_limit_order)
        assert report.cl_ord_id == "ord-alpaca-002"
        assert report.exec_type == OrderState.FILLED
        assert report.cum_quantity == 5.0
        assert report.leaves_quantity == 0.0
        assert report.last_price == 130.5
        assert report.average_price == 130.5

    @pytest.mark.asyncio
    async def test_submit_insufficient_margin_error(self, sample_buy_order: Order) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v2/account":
                return httpx.Response(200, json={"status": "ACTIVE"})
            if request.url.path == "/v2/orders":
                return httpx.Response(
                    403,
                    json={"message": "insufficient buying power for order"},
                )
            return httpx.Response(404)

        transport = _mock_transport(handler)
        gw = AlpacaExecutionGateway("key", "sec", transport=transport)
        await gw.connect()

        with pytest.raises(InsufficientMarginException) as exc_info:
            await gw.submit_order(sample_buy_order)
        assert exc_info.value.code == ERR_GW_INSUFFICIENT_MARGIN

    @pytest.mark.asyncio
    async def test_submit_rate_limit_error(self, sample_buy_order: Order) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v2/account":
                return httpx.Response(200, json={"status": "ACTIVE"})
            if request.url.path == "/v2/orders":
                return httpx.Response(429, json={"message": "too many requests"})
            return httpx.Response(404)

        transport = _mock_transport(handler)
        gw = AlpacaExecutionGateway("key", "sec", transport=transport)
        await gw.connect()

        with pytest.raises(GatewayError) as exc_info:
            await gw.submit_order(sample_buy_order)
        assert exc_info.value.code == ERR_GW_RATE_LIMIT_EXCEEDED


class TestAlpacaOrderManagement:
    """Verify cancellation, order retrieval, balance, and positions."""

    @pytest.mark.asyncio
    async def test_cancel_order_success(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v2/account":
                return httpx.Response(200, json={"status": "ACTIVE"})
            if "by_client_order_id" in request.url.path and request.method == "DELETE":
                return httpx.Response(204)
            return httpx.Response(404)

        transport = _mock_transport(handler)
        gw = AlpacaExecutionGateway("key", "sec", transport=transport)
        await gw.connect()

        report = await gw.cancel_order("ord-alpaca-001")
        assert report.cl_ord_id == "ord-alpaca-001"
        assert report.exec_type == OrderState.CANCELLED

    @pytest.mark.asyncio
    async def test_get_positions(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v2/account":
                return httpx.Response(200, json={"status": "ACTIVE"})
            if request.url.path == "/v2/positions":
                return httpx.Response(
                    200,
                    json=[
                        {"symbol": "AAPL", "qty": "15.0"},
                        {"symbol": "MSFT", "qty": "-5.0"},
                    ],
                )
            return httpx.Response(404)

        transport = _mock_transport(handler)
        gw = AlpacaExecutionGateway("key", "sec", transport=transport)
        await gw.connect()

        positions = await gw.get_positions()
        assert positions == {"AAPL": 15.0, "MSFT": -5.0}

    @pytest.mark.asyncio
    async def test_get_account_balance(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v2/account":
                return httpx.Response(
                    200,
                    json={
                        "cash": "45250.75",
                        "portfolio_value": "125000.50",
                        "buying_power": "250000.00",
                    },
                )
            return httpx.Response(404)

        transport = _mock_transport(handler)
        gw = AlpacaExecutionGateway("key", "sec", transport=transport)
        await gw.connect()

        bal = await gw.get_account_balance()
        assert bal["cash"] == 45250.75
        assert bal["equity"] == 125000.50
        assert bal["buying_power"] == 250000.00
