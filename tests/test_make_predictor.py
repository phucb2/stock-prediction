"""Tests for utils.make_predictor and predictor registry."""

from __future__ import annotations

import importlib
import unittest

import numpy as np

from main import DefaultPredictor, make_predictor, parse_algo_params, registered_predictor_names, resolve_predictor_class
from utils import _predictor_classes_in_module
from SES import SimpleExponentialSmoothing


class TestMakePredictor(unittest.TestCase):
    def test_resolve_by_full_class_name(self) -> None:
        self.assertIs(resolve_predictor_class("default"), DefaultPredictor)
        self.assertIs(resolve_predictor_class("DEFAULT"), DefaultPredictor)
        self.assertIs(resolve_predictor_class("DefaultPredictor"), DefaultPredictor)
        self.assertIs(
            resolve_predictor_class("SimpleExponentialSmoothing"),
            SimpleExponentialSmoothing,
        )

    def test_ses_short_name_rejected(self) -> None:
        with self.assertRaises(ValueError):
            resolve_predictor_class("ses")

    def test_registered_names_include_discovered_class(self) -> None:
        names = registered_predictor_names()
        self.assertIn("default", names)
        self.assertIn("SimpleExponentialSmoothing", names)
        self.assertIn("DefaultPredictor", names)

    def test_discovery_finds_subclasses_in_ses_module(self) -> None:
        mod = importlib.import_module("SES")
        classes = _predictor_classes_in_module(mod)
        self.assertIn(SimpleExponentialSmoothing, classes)

    def test_make_predictor_ses_params(self) -> None:
        label, fn = make_predictor("SimpleExponentialSmoothing", {"alpha": 0.5})
        self.assertIn("SimpleExponentialSmoothing", label)
        self.assertIn("0.5", label)
        p = np.array([100.0, 101.0, 102.0])
        out = fn(p, np.zeros(3), 5)
        self.assertEqual(len(out), 3)

    def test_parse_algo_params(self) -> None:
        d = parse_algo_params(["alpha=0.2", "flag=true"])
        self.assertAlmostEqual(d["alpha"], 0.2)
        self.assertTrue(d["flag"])

    def test_unknown_init_keys_ignored(self) -> None:
        """Params not in ``__init__`` are dropped unless the ctor has ``**kwargs``."""
        label, fn = make_predictor("default", {"foo": 1})
        self.assertEqual(label, "DefaultPredictor")
        fn(np.array([1.0, 2.0]), np.zeros(2), 5)

    def test_ses_rejects_invalid_alpha(self) -> None:
        with self.assertRaises(ValueError):
            make_predictor("SimpleExponentialSmoothing", {"alpha": 0.0})


if __name__ == "__main__":
    unittest.main()
