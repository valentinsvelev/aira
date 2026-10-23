"""Experimental TEV1 NLI-style verifier using Ollama's /v1/systemone API.

All edits also pass deterministic evidence-substring, length, and number checks.
TEV1 is a probabilistic classifier, NOT a proof of entailment; any error fails
closed. Never make untrusted source text part of system instructions.
"""
from __future__ import annotations

import re

import httpx

from aira.context import Verdict, normalize


class Tev1Checker:
    def __init__(self, base_url: str, model: str = "tev1:0.8b", threshold: float = 0.75,
                 transport=None):
        if not 0.0 < threshold <= 1.0:
            raise ValueError("TEV1 probability threshold must be between 0 and 1")
        self.url = base_url.rstrip("/") + "/v1/systemone"
        self.model = model
        self.threshold = threshold
        self.transport = transport

    def classify(self, original: str, edited: str) -> dict:
        if len(original.split()) > 180 or len(edited.split()) > 100:
            raise ValueError("TEV1 inputs must be short excerpts (source <=180, claim <=100 words)")
        payload = {
            "model": self.model,
            "state": {"source_evidence": original, "edited_claim": edited},
            "questions": {"faithfulness": {
                "type": "choice",
                "instructions": ("Does the source evidence logically entail the entire edited claim, "
                                 "including quantities, negation, conditions, scope and caveats? "
                                 "Judge source support, not topical similarity."),
                "criteria": {
                    "supported": "Every assertion in the edited claim is explicitly supported by the source.",
                    "contradicted": "At least one assertion conflicts with the source evidence.",
                    "insufficient": "Source lacks support for at least one part of the claim, or omits key qualifiers."
                }
            }}
        }
        with httpx.Client(timeout=90, transport=self.transport) as client:
            response = client.post(self.url, json=payload)
            response.raise_for_status()
            answer = response.json().get("answers", {}).get("faithfulness", {})
        choice = answer.get("choice")
        probabilities = answer.get("probabilities", {})
        if choice not in ("supported", "contradicted", "insufficient"):
            raise ValueError("Unexpected TEV1 decision response")
        if not isinstance(probabilities, dict):
            raise ValueError("TEV1 did not return option probabilities")
        # Probability is a model estimate, not a calibrated accuracy guarantee.
        supported = probabilities.get("supported")
        if not isinstance(supported, (int, float)) or not 0 <= supported <= 1:
            raise ValueError("TEV1 response lacks a valid supported probability")
        return {"choice": choice, "supported_probability": float(supported)}

    def check(self, claim: str, quote: str, evidence: str) -> Verdict:
        c, q, e = normalize(claim), normalize(quote), normalize(evidence)
        if not c or not q or not e or len(c) < 20:
            return Verdict(False, "Nonempty source and claim of at least 20 characters required", method="tev1_decision")
        if q not in e:
            return Verdict(False, "Quote must match immutable source evidence", method="tev1_decision")
        nums_claim = set(re.findall(r"\d+(?:[.,]\d+)?", c))
        nums_quote = set(re.findall(r"\d+(?:[.,]\d+)?", q))
        if not nums_claim.issubset(nums_quote):
            return Verdict(False, "Edited claim introduces a number absent from source", method="tev1_decision")
        try:
            result = self.classify(quote, claim)
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            return Verdict(False, f"TEV1 verification unavailable: {type(exc).__name__}; rejected safely",
                           method="tev1_decision")
        prob = result["supported_probability"]
        accepted = result["choice"] == "supported" and prob >= self.threshold
        return Verdict(accepted,
                       f"TEV1: {result['choice']}; supported probability={prob:.3f}; "
                       f"threshold={self.threshold:.2f} (experimental, not proof)",
                       similarity=round(prob, 6), method="tev1_decision")
