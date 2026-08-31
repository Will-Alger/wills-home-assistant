"""Per-command cost and latency meter.

Every LLM hop is recorded and priced; per-command and session totals are
printed by the REPL, and every hop is appended to `.usage.jsonl` (gitignored)
so real usage data — not estimates — drives model/effort decisions.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from assistant.llm.base import TurnResult


@dataclass(frozen=True)
class Pricing:
    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float
    cache_write_1h_per_mtok: float  # 1h-TTL writes bill at 2x input


PRICES: dict[str, Pricing] = {
    "claude-opus-5": Pricing(5.00, 25.00, 0.50, 10.00),
    "claude-sonnet-5": Pricing(2.00, 10.00, 0.20, 4.00),
    "claude-haiku-4-5": Pricing(1.00, 5.00, 0.10, 2.00),
}


def _price(model: str) -> Pricing | None:
    for known, pricing in PRICES.items():
        if model.startswith(known):
            return pricing
    return None


def hop_cost_usd(result: TurnResult) -> float:
    pricing = _price(result.model)
    if pricing is None:
        return 0.0
    usage = result.usage
    return (
        usage.input_tokens * pricing.input_per_mtok
        + usage.output_tokens * pricing.output_per_mtok
        + usage.cache_read_tokens * pricing.cache_read_per_mtok
        + usage.cache_write_tokens * pricing.cache_write_1h_per_mtok
    ) / 1_000_000


@dataclass
class CommandStats:
    hops: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0


class Meter:
    def __init__(self, log_path: Path | None = None) -> None:
        self._log_path = log_path
        self.session_cost_usd = 0.0
        self.session_hops = 0
        self.current = CommandStats()

    def start_command(self) -> None:
        self.current = CommandStats()

    def record(self, result: TurnResult) -> float:
        cost = hop_cost_usd(result)
        self.current.hops += 1
        self.current.cost_usd += cost
        self.current.latency_ms += result.latency_ms
        self.current.output_tokens += result.usage.output_tokens
        self.current.cache_read_tokens += result.usage.cache_read_tokens
        self.session_cost_usd += cost
        self.session_hops += 1
        if self._log_path is not None:
            entry = {
                "ts": time.time(),
                "model": result.model,
                "stop_reason": result.stop_reason,
                "latency_ms": result.latency_ms,
                "input_tokens": result.usage.input_tokens,
                "output_tokens": result.usage.output_tokens,
                "cache_read_tokens": result.usage.cache_read_tokens,
                "cache_write_tokens": result.usage.cache_write_tokens,
                "cost_usd": round(cost, 6),
            }
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        return cost

    def command_line(self) -> str:
        c = self.current
        cache_note = "warm" if c.cache_read_tokens else "cold"
        return (
            f"{c.hops} hop(s) · {c.latency_ms} ms · {c.output_tokens} out-tok · "
            f"cache {cache_note} · ${c.cost_usd:.4f} cmd · ${self.session_cost_usd:.4f} session"
        )
