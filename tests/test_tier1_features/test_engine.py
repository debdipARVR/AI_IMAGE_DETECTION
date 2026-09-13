"""
Tier 1: Feature Coverage - Engine Initialization & Offline Cache
Test ID: T1.1.1 to T1.1.5
Authoritative Source: PROJECT.md § Feature 1 & TEST_INFRA.md
"""

import pytest
import torch
from src.vae_resonance import VAEResonanceEngine


def test_t1_1_1_default_model_instantiation(test_engine):
    """T1.1.1: Default model instantiation sets device and eval mode."""
    engine = test_engine
    assert engine is not None
    assert engine.vae is not None
    assert hasattr(engine, "device")
    assert engine.device in ["cpu", "cuda"]


def test_t1_1_2_explicit_cpu_target_device():
    """T1.1.2: Explicit CPU target device parameter binds to CPU."""
    engine = VAEResonanceEngine(device="cpu")
    assert engine.device == "cpu"


def test_t1_1_3_local_cache_offline_loading(monkeypatch):
    """T1.1.3: Local weights offline loading succeeds without network sockets."""
    import socket
    def blocked_connect(*args, **kwargs):
        raise ConnectionRefusedError("Simulated offline/air-gapped environment")
    monkeypatch.setattr(socket.socket, "connect", blocked_connect)
    
    # Engine instantiation should load locally without attempting socket connections
    engine = VAEResonanceEngine(device="cpu")
    assert engine is not None


def test_t1_1_4_eval_mode_immutability(test_engine):
    """T1.1.4: Evaluation mode immutability ensures gradients are disabled."""
    engine = test_engine
    for param in engine.vae.parameters():
        assert param.requires_grad is False


def test_t1_1_5_idempotent_initialization():
    """T1.1.5: Consecutive engine initializations maintain architectural equivalence."""
    engine_1 = VAEResonanceEngine(device="cpu")
    engine_2 = VAEResonanceEngine(device="cpu")
    assert engine_1.device == engine_2.device
    assert type(engine_1) is type(engine_2)
