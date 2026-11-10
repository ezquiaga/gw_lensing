"""Regression tests for gw_lensing.utils.gwutils.snr_from_psd.

Reference SNRs were computed with the code at commit 6648769 (before the
duplicated aS() call was removed), for detector 'A#', fmin=10 Hz, Tobs=1 yr,
based='ground'. Any refactor of snr_from_psd must reproduce them exactly.
"""
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import pytest

from gw_lensing.utils import gwutils
from gw_lensing.detectors import sensitivity_curves as sc

FMIN, TOBS, DETECTOR, BASED = 10.0, 1.0, "A#", "ground"

# (m1z [Msun], m2z [Msun], dL [Mpc], reference SNR)
REFERENCE = [
    (10.0, 10.0, 1000.0, 55.848945502071736),
    (36.0, 29.0, 410.0, 330.24221854564547),
    (80.0, 40.0, 5000.0, 38.648279849098024),
    (300.0, 250.0, 20000.0, 20.255606232810056),
    (1.4, 1.4, 100.0, 118.27585109994111),
    (1000.0, 900.0, 3000.0, 13.207416743327158),
]


@pytest.fixture(scope="module")
def psd():
    return sc.detector_psd(DETECTOR)


@pytest.mark.parametrize("m1, m2, dl, ref", REFERENCE)
def test_snr_from_psd_matches_reference(psd, m1, m2, dl, ref):
    Sn, fmin_d, fmax_d = psd
    val = gwutils.snr_from_psd(m1, m2, dl, FMIN, TOBS, Sn, fmin_d, fmax_d, BASED)
    assert val == pytest.approx(ref, rel=1e-12)


@pytest.mark.parametrize("m1, m2, dl, ref", REFERENCE[:3])
def test_snr_wrapper_matches_reference(m1, m2, dl, ref):
    assert gwutils.snr(m1, m2, dl, FMIN, TOBS, DETECTOR, BASED) == pytest.approx(ref, rel=1e-12)


def test_snr_scales_inversely_with_distance(psd):
    Sn, fmin_d, fmax_d = psd
    s1 = gwutils.snr_from_psd(36.0, 29.0, 1000.0, FMIN, TOBS, Sn, fmin_d, fmax_d, BASED)
    s5 = gwutils.snr_from_psd(36.0, 29.0, 5000.0, FMIN, TOBS, Sn, fmin_d, fmax_d, BASED)
    assert s1 / s5 == pytest.approx(5.0, rel=1e-10)


def test_snr_from_psd_generates_waveform_once(psd, monkeypatch):
    """The waveform (aS -> pycbc get_fd_waveform) is the expensive step and
    must be computed only once per SNR evaluation."""
    Sn, fmin_d, fmax_d = psd
    calls = {"n": 0}
    original = gwutils.aS

    def counting_aS(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(gwutils, "aS", counting_aS)
    gwutils.snr_from_psd(36.0, 29.0, 410.0, FMIN, TOBS, Sn, fmin_d, fmax_d, BASED)
    assert calls["n"] == 1


def test_snr_from_psd_evaluates_psd_once(psd):
    """The PSD interpolator must be evaluated once on the frequency array
    (not separately for the integrand and the NaN mask)."""
    Sn, fmin_d, fmax_d = psd
    calls = {"n": 0}

    def counting_Sn(f):
        calls["n"] += 1
        return Sn(f)

    val = gwutils.snr_from_psd(36.0, 29.0, 410.0, FMIN, TOBS, counting_Sn, fmin_d, fmax_d, BASED)
    assert calls["n"] == 1
    assert val == pytest.approx(REFERENCE[1][3], rel=1e-12)
