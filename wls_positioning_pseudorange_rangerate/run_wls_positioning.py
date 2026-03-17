from __future__ import annotations

import argparse
import json
import math
import re
import sys
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import jpype
import matplotlib
import numpy as np
import orekit_jpype
import pymap3d as pm
import requests

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STEP1_INDEX_FILE = PROJECT_ROOT / "step1_data_generation" / "output" / "passes_index.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OREKIT_DATA_ZIP = PROJECT_ROOT / "step1_data_generation" / "orekit-data.zip"
OREKIT_DATA_URL = "https://gitlab.orekit.org/orekit/orekit-data/-/archive/main/orekit-data-main.zip"


@dataclass(frozen=True)
class ReceiverConfig:
    latitude_deg: float = 45.772625
    longitude_deg: float = 126.682625
    altitude_m: float = 154.0


@dataclass(frozen=True)
class SelectionConfig:
    satellite_name: str = "ORBCOMM FM34"
    pass_index: int = 1
    orbit_source: str = "HPOP"


@dataclass(frozen=True)
class DataConfig:
    use_full_pass: bool = True
    start_sec: float = 100.0
    duration_sec: float = 150.0
    measurement_step_sec: float = 0.1


@dataclass(frozen=True)
class TruthConfig:
    cb0_true_m: float = 1200.0
    cdot_true_mps: float = 0.02


@dataclass(frozen=True)
class InitConfig:
    initial_offset_km: float = 10.0
    cb0_m: float = 0.0
    cdot0_mps: float = 0.0


@dataclass(frozen=True)
class WlsConfig:
    batch_size: int = 100
    use_cumulative: bool = True
    final_partial_update: bool = True
    max_iter: int = 20
    tol_pos_m: float = 1.0e-4
    tol_cb_m: float = 1.0e-4
    tol_cdot_mps: float = 1.0e-8
    lambda0: float = 1.0e-6


@dataclass(frozen=True)
class MonteCarloConfig:
    trial_count: int = 100
    base_seed: int = 20260314
    stop_on_failure: bool = False


@dataclass(frozen=True)
class AtmosphereConfig:
    enable_klobuchar: bool = True
    enable_hopfield: bool = True
    signal_frequency_hz: float = 12_000_000_000.0
    klobuchar_nav_file: str = ""
    klobuchar_alpha: tuple[float, float, float, float] = (2.4214e-08, 1.4901e-08, -1.1921e-07, 0.0)
    klobuchar_beta: tuple[float, float, float, float] = (1.1674e05, -2.2938e05, -1.3107e05, 1.0486e06)


@dataclass(frozen=True)
class RunConfig:
    single_trial_only: bool = False
    single_trial_seed: int = 20260315
    save_results: bool = True


RECEIVER = ReceiverConfig()
SELECTION = SelectionConfig()
DATA = DataConfig()
TRUTH = TruthConfig()
INIT = InitConfig()
WLS = WlsConfig()
MC = MonteCarloConfig()
ATM = AtmosphereConfig()
RUN = RunConfig()
NOISE_PROFILES = ("orbcomm_like", "optimistic_pnt")


def scalar_string(value: np.ndarray | str | bytes) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8")
    array = np.asarray(value)
    return str(array.item() if array.shape == () else array.reshape(-1)[0])


def scalar_number(value: np.ndarray | float | int) -> float:
    array = np.asarray(value)
    return float(array.item() if array.shape == () else array.reshape(-1)[0])


def parse_cli_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-satellite WLS positioning from step1 pass data.")
    parser.add_argument("--list-passes", action="store_true", help="List available satellite passes and exit.")
    parser.add_argument(
        "--satellite-name",
        default=SELECTION.satellite_name,
        help="Satellite name to use for positioning.",
    )
    parser.add_argument(
        "--pass-index",
        type=int,
        default=SELECTION.pass_index,
        help="Pass index for the selected satellite.",
    )
    parser.add_argument(
        "--orbit-source",
        choices=("HPOP", "SGP4"),
        default=SELECTION.orbit_source,
        help="Orbit source used to synthesize measurements.",
    )
    return parser.parse_args(argv)


def sanitize_tag(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def resolve_signal_frequency_hz(satellite_name: str) -> float:
    _ = satellite_name
    return ATM.signal_frequency_hz


def receiver_ecef(receiver: ReceiverConfig) -> np.ndarray:
    x, y, z = pm.geodetic2ecef(receiver.latitude_deg, receiver.longitude_deg, receiver.altitude_m, deg=True)
    return np.array([x, y, z], dtype=float)


def download_file(url: str, destination: Path, max_retries: int = 3) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for _ in range(max_retries):
        try:
            with requests.get(url, stream=True, timeout=60) as response:
                response.raise_for_status()
                with destination.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            with zipfile.ZipFile(destination) as archive:
                if archive.testzip() is not None:
                    raise zipfile.BadZipFile("Corrupted orekit-data.zip entry detected.")
            return
        except Exception as exc:  # pragma: no cover - network retry path
            last_error = exc
            if destination.exists():
                destination.unlink()
    raise RuntimeError(f"Failed to download Orekit data from {url}") from last_error


def ensure_orekit_ready(orekit_data_zip: Path) -> None:
    if not orekit_data_zip.exists():
        download_file(OREKIT_DATA_URL, orekit_data_zip)
    else:
        try:
            with zipfile.ZipFile(orekit_data_zip) as archive:
                if archive.testzip() is not None:
                    raise zipfile.BadZipFile("Corrupted archive entry detected.")
        except zipfile.BadZipFile:
            download_file(OREKIT_DATA_URL, orekit_data_zip)

    if not jpype.isJVMStarted():
        orekit_jpype.initVM()

    from orekit_jpype.pyhelpers import setup_orekit_curdir

    setup_orekit_curdir(str(orekit_data_zip))


def resolve_klobuchar_coefficients() -> tuple[list[float], list[float], str]:
    if ATM.klobuchar_nav_file:
        nav_path = Path(ATM.klobuchar_nav_file)
        if not nav_path.is_absolute():
            nav_path = PROJECT_ROOT / nav_path
        if not nav_path.exists():
            raise FileNotFoundError(f"Klobuchar navigation file not found: {nav_path}")

        from java.io import BufferedInputStream, FileInputStream
        from org.orekit.models.earth.ionosphere import KlobucharIonoCoefficientsLoader

        loader = KlobucharIonoCoefficientsLoader()
        stream = BufferedInputStream(FileInputStream(str(nav_path)))
        try:
            loader.loadData(stream, nav_path.name)
        finally:
            stream.close()
        return (
            [float(value) for value in loader.getAlpha()],
            [float(value) for value in loader.getBeta()],
            str(nav_path.resolve()),
        )

    return list(ATM.klobuchar_alpha), list(ATM.klobuchar_beta), "config"


def build_atmosphere_context(receiver: ReceiverConfig) -> dict[str, Any]:
    if not (ATM.enable_klobuchar or ATM.enable_hopfield):
        return {}

    ensure_orekit_ready(OREKIT_DATA_ZIP)

    from orekit_jpype.pyhelpers import datetime_to_absolutedate
    from org.orekit.bodies import GeodeticPoint
    from org.orekit.models.earth.ionosphere import KlobucharIonoModel
    from org.orekit.models.earth.troposphere import ModifiedHopfieldModel, TroposphericModelUtils

    klobuchar_alpha, klobuchar_beta, klobuchar_source = resolve_klobuchar_coefficients()
    return {
        "receiver_gp": GeodeticPoint(
            math.radians(receiver.latitude_deg),
            math.radians(receiver.longitude_deg),
            receiver.altitude_m,
        ),
        "datetime_to_absolutedate": datetime_to_absolutedate,
        "iono_model": KlobucharIonoModel(klobuchar_alpha, klobuchar_beta) if ATM.enable_klobuchar else None,
        "tropo_model": ModifiedHopfieldModel(TroposphericModelUtils.STANDARD_ATMOSPHERE_PROVIDER)
        if ATM.enable_hopfield
        else None,
        "klobuchar_alpha": np.asarray(klobuchar_alpha, dtype=float),
        "klobuchar_beta": np.asarray(klobuchar_beta, dtype=float),
        "klobuchar_source": klobuchar_source,
    }


def finite_difference(values: np.ndarray, t_sec: np.ndarray) -> np.ndarray:
    if values.size < 2:
        return np.zeros_like(values)
    edge_order = 2 if values.size >= 3 else 1
    return np.gradient(values, t_sec, edge_order=edge_order)


def compute_tracking_angles(
    receiver: ReceiverConfig,
    rs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    az_deg, el_deg, slant_range_m = pm.ecef2aer(
        rs[0],
        rs[1],
        rs[2],
        receiver.latitude_deg,
        receiver.longitude_deg,
        receiver.altitude_m,
        deg=True,
    )
    return (
        np.asarray(az_deg, dtype=float),
        np.asarray(el_deg, dtype=float),
        np.asarray(slant_range_m, dtype=float),
    )


def build_observation_atmosphere(
    receiver: ReceiverConfig,
    observation_start_utc: str,
    t_sec: np.ndarray,
    rs: np.ndarray,
    signal_frequency_hz: float,
) -> dict[str, np.ndarray]:
    azimuth_deg, elevation_deg, slant_range_m = compute_tracking_angles(receiver, rs)
    iono_delay_m = np.zeros_like(t_sec)
    tropo_delay_m = np.zeros_like(t_sec)

    if ATM.enable_klobuchar or ATM.enable_hopfield:
        context = build_atmosphere_context(receiver)
        start_date = context["datetime_to_absolutedate"](datetime.fromisoformat(observation_start_utc))

        from org.orekit.utils import TrackingCoordinates

        for index in range(t_sec.size):
            if elevation_deg[index] <= 0.0:
                continue
            date = start_date.shiftedBy(float(t_sec[index]))
            azimuth_rad = math.radians(float(azimuth_deg[index]))
            elevation_rad = math.radians(float(elevation_deg[index]))

            if context["iono_model"] is not None:
                iono_delay_m[index] = context["iono_model"].pathDelay(
                    date,
                    context["receiver_gp"],
                    elevation_rad,
                    azimuth_rad,
                    signal_frequency_hz,
                    context["iono_model"].getParameters(date),
                )
            if context["tropo_model"] is not None:
                tracking = TrackingCoordinates(
                    azimuth_rad,
                    elevation_rad,
                    float(slant_range_m[index]),
                )
                tropo_delay_m[index] = context["tropo_model"].pathDelay(
                    tracking,
                    context["receiver_gp"],
                    context["tropo_model"].getParameters(date),
                    date,
                ).getDelay()

    total_delay_m = iono_delay_m + tropo_delay_m
    return {
        "azimuth_deg": azimuth_deg,
        "elevation_deg": elevation_deg,
        "slant_range_m": slant_range_m,
        "iono_delay_m": iono_delay_m,
        "tropo_delay_m": tropo_delay_m,
        "total_delay_m": total_delay_m,
        "total_delay_rate_mps": finite_difference(total_delay_m, t_sec),
        "signal_frequency_hz": np.array(signal_frequency_hz, dtype=float),
        "klobuchar_alpha": context["klobuchar_alpha"] if ATM.enable_klobuchar or ATM.enable_hopfield else np.asarray(ATM.klobuchar_alpha, dtype=float),
        "klobuchar_beta": context["klobuchar_beta"] if ATM.enable_klobuchar or ATM.enable_hopfield else np.asarray(ATM.klobuchar_beta, dtype=float),
        "klobuchar_source": np.array(
            context["klobuchar_source"] if ATM.enable_klobuchar or ATM.enable_hopfield else "config"
        ),
    }


def summarize_observation_atmosphere(obs_atmosphere: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        "iono_mean_m": float(np.mean(obs_atmosphere["iono_delay_m"])),
        "iono_max_m": float(np.max(obs_atmosphere["iono_delay_m"])),
        "tropo_mean_m": float(np.mean(obs_atmosphere["tropo_delay_m"])),
        "tropo_max_m": float(np.max(obs_atmosphere["tropo_delay_m"])),
        "total_mean_m": float(np.mean(obs_atmosphere["total_delay_m"])),
        "total_max_m": float(np.max(obs_atmosphere["total_delay_m"])),
        "total_rate_rms_mps": rms(obs_atmosphere["total_delay_rate_mps"]),
    }


def load_pass_index(index_file: Path) -> dict[str, Any]:
    if not index_file.exists():
        raise FileNotFoundError(
            f"Step 1 summary file not found: {index_file}. Run step1_data_generation/generate_pass_data.py first."
        )
    return json.loads(index_file.read_text(encoding="utf-8"))


def list_available_passes(index_file: Path) -> None:
    summary = load_pass_index(index_file)
    passes = summary.get("passes", [])
    if not passes:
        print("No passes found in passes_index.json")
        return

    print("Available passes:")
    for record in passes:
        frequency_mhz = resolve_signal_frequency_hz(record["satellite_name"]) / 1.0e6
        print(
            f"  satellite = {record['satellite_name']:<24} | "
            f"pass = {int(record['pass_index']):02d} | "
            f"duration = {float(record['duration_sec']):7.1f} s | "
            f"start = {record['pass_start_utc']} | "
            f"freq = {frequency_mhz:.3f} MHz"
        )


def load_pass_record(index_file: Path, satellite_name: str, pass_index: int) -> dict[str, Any]:
    summary = load_pass_index(index_file)
    matches = [
        record
        for record in summary.get("passes", [])
        if record["satellite_name"] == satellite_name and int(record["pass_index"]) == int(pass_index)
    ]
    if not matches:
        available = [f"{record['satellite_name']}#pass{record['pass_index']}" for record in summary.get("passes", [])]
        raise ValueError(
            f"Requested pass not found for satellite '{satellite_name}' and pass index {pass_index}. "
            f"Available passes: {available}. Use --list-passes to inspect the current step1 catalog."
        )
    return matches[0]


def load_step1_pass(pass_file: Path, orbit_source: str) -> dict[str, Any]:
    if not pass_file.exists():
        raise FileNotFoundError(f"Step 1 pass file not found: {pass_file}")

    with np.load(pass_file, allow_pickle=False) as data:
        orbit_key = orbit_source.strip().upper()
        if orbit_key == "HPOP":
            pos_key = "hpop_ecef_pos_m"
            vel_key = "hpop_ecef_vel_mps"
        elif orbit_key == "SGP4":
            pos_key = "sgp4_ecef_pos_m"
            vel_key = "sgp4_ecef_vel_mps"
        else:
            raise ValueError(f"Unsupported orbit source '{orbit_source}'. Use 'HPOP' or 'SGP4'.")

        return {
            "satellite_name": scalar_string(data["satellite_name"]),
            "catalog_id": int(scalar_number(data["catalog_id"])),
            "pass_index": int(scalar_number(data["pass_index"])),
            "pass_start_utc": scalar_string(data["pass_start_utc"]),
            "pass_end_utc": scalar_string(data["pass_end_utc"]),
            "time_seconds": np.asarray(data["time_seconds"], dtype=float).reshape(-1),
            "receiver_lla_deg_m": np.asarray(data["receiver_lla_deg_m"], dtype=float).reshape(-1),
            "receiver_ecef_m": np.asarray(data["receiver_ecef_m"], dtype=float).reshape(-1),
            "rs_ecef": np.asarray(data[pos_key], dtype=float).T,
            "vs_ecef": np.asarray(data[vel_key], dtype=float).T,
        }


def slice_observation_window(
    time_seconds: np.ndarray,
    rs: np.ndarray,
    vs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if DATA.use_full_pass:
        idx = np.arange(time_seconds.size)
    else:
        stop_sec = DATA.start_sec + DATA.duration_sec
        idx = np.where((time_seconds >= DATA.start_sec) & (time_seconds <= stop_sec))[0]
    if idx.size == 0:
        raise ValueError("No observation epochs remain after applying DataConfig.")

    if time_seconds.size > 1 and DATA.measurement_step_sec > 0.0:
        native_step = float(np.median(np.diff(time_seconds)))
        stride = max(1, int(round(DATA.measurement_step_sec / native_step)))
        idx = idx[::stride]
        if idx.size == 0:
            raise ValueError("Measurement decimation removed all observation epochs.")

    start_offset_sec = float(time_seconds[idx[0]])
    t_sec = time_seconds[idx].copy()
    t_sec -= start_offset_sec
    return t_sec, rs[:, idx], vs[:, idx], start_offset_sec


def rho_jac_ecef(pos: np.ndarray, rs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dr = pos.reshape(3, 1) - rs
    rho = np.linalg.norm(dr, axis=0)
    jac = (dr / rho).T
    return rho, jac


def rhodot_jac_ecef(pos: np.ndarray, rs: np.ndarray, vs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dr = pos.reshape(3, 1) - rs
    rho = np.linalg.norm(dr, axis=0)
    rd = np.sum(-vs * dr, axis=0) / rho
    jac = (-vs / rho.reshape(1, -1) - dr * (rd / (rho**2)).reshape(1, -1)).T
    return rd, jac


def safe_solve(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    matrix = 0.5 * (matrix + matrix.T)
    if np.linalg.cond(matrix) > 1.0e12:
        return np.linalg.pinv(matrix) @ rhs
    return np.linalg.solve(matrix, rhs)


def build_posterior_covariance_5state(
    state: np.ndarray,
    rs: np.ndarray,
    vs: np.ndarray,
    t_sec: np.ndarray,
    sigma_rho: float,
    sigma_rhodot: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, jac_rho = rho_jac_ecef(state[:3], rs)
    _, jac_rd = rhodot_jac_ecef(state[:3], rs, vs)
    h_rho = np.column_stack((jac_rho, np.ones(t_sec.size), t_sec))
    h_rhodot = np.column_stack((jac_rd, np.zeros(t_sec.size), np.ones(t_sec.size)))
    h_whitened = np.vstack((h_rho / sigma_rho, h_rhodot / sigma_rhodot))
    normal_matrix = h_whitened.T @ h_whitened
    normal_matrix = 0.5 * (normal_matrix + normal_matrix.T)
    covariance = np.linalg.pinv(normal_matrix) if np.linalg.cond(normal_matrix) > 1.0e12 else np.linalg.inv(normal_matrix)
    covariance = 0.5 * (covariance + covariance.T)
    return h_rho, h_rhodot, covariance


def solve_one_wls_ecef_5state(
    x_init: np.ndarray,
    y_rho: np.ndarray,
    y_rhodot: np.ndarray,
    rs: np.ndarray,
    vs: np.ndarray,
    t_sec: np.ndarray,
    sigma_rho: float,
    sigma_rhodot: float,
) -> tuple[np.ndarray, np.ndarray, int, float, bool]:
    state = x_init.astype(float).copy()
    lambda_lm = WLS.lambda0
    converged = False
    final_cost = math.nan

    for iteration in range(1, WLS.max_iter + 1):
        rho, jac_rho = rho_jac_ecef(state[:3], rs)
        rd, jac_rd = rhodot_jac_ecef(state[:3], rs, vs)
        h_rho = np.column_stack((jac_rho, np.ones(y_rho.size), t_sec))
        h_rhodot = np.column_stack((jac_rd, np.zeros(y_rhodot.size), np.ones(y_rhodot.size)))
        v_rho = y_rho - (rho + state[3] + state[4] * t_sec)
        v_rhodot = y_rhodot - (rd + state[4])

        h_whitened = np.vstack((h_rho / sigma_rho, h_rhodot / sigma_rhodot))
        v_whitened = np.concatenate((v_rho / sigma_rho, v_rhodot / sigma_rhodot))
        normal_matrix = h_whitened.T @ h_whitened
        gradient = h_whitened.T @ v_whitened
        normal_matrix = 0.5 * (normal_matrix + normal_matrix.T)

        damping = lambda_lm * np.diag(np.maximum(np.abs(np.diag(normal_matrix)), 1.0))
        delta = safe_solve(normal_matrix + damping, gradient)
        trial_state = state + delta

        rho_trial, _ = rho_jac_ecef(trial_state[:3], rs)
        rd_trial, _ = rhodot_jac_ecef(trial_state[:3], rs, vs)
        v_rho_trial = y_rho - (rho_trial + trial_state[3] + trial_state[4] * t_sec)
        v_rhodot_trial = y_rhodot - (rd_trial + trial_state[4])

        cost_now = float(np.sum((v_rho / sigma_rho) ** 2) + np.sum((v_rhodot / sigma_rhodot) ** 2))
        cost_trial = float(np.sum((v_rho_trial / sigma_rho) ** 2) + np.sum((v_rhodot_trial / sigma_rhodot) ** 2))

        if cost_trial <= cost_now:
            state = trial_state
            lambda_lm = max(lambda_lm / 10.0, 1.0e-12)
            if (
                np.linalg.norm(delta[:3]) < WLS.tol_pos_m
                and abs(delta[3]) < WLS.tol_cb_m
                and abs(delta[4]) < WLS.tol_cdot_mps
            ):
                converged = True
                final_cost = cost_trial
                break
        else:
            lambda_lm = min(lambda_lm * 10.0, 1.0e12)

    _, _, covariance = build_posterior_covariance_5state(state, rs, vs, t_sec, sigma_rho, sigma_rhodot)
    rho, _ = rho_jac_ecef(state[:3], rs)
    rd, _ = rhodot_jac_ecef(state[:3], rs, vs)
    residual_rho = y_rho - (rho + state[3] + state[4] * t_sec)
    residual_rhodot = y_rhodot - (rd + state[4])
    final_cost = float(np.sum((residual_rho / sigma_rho) ** 2) + np.sum((residual_rhodot / sigma_rhodot) ** 2))
    return state, covariance, iteration, final_cost, converged


def ecef_to_enu_matrix(reference_ecef: np.ndarray) -> np.ndarray:
    lat_deg, lon_deg, _ = pm.ecef2geodetic(reference_ecef[0], reference_ecef[1], reference_ecef[2], deg=True)
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    e_hat = np.array([-math.sin(lon), math.cos(lon), 0.0], dtype=float)
    n_hat = np.array(
        [-math.sin(lat) * math.cos(lon), -math.sin(lat) * math.sin(lon), math.cos(lat)],
        dtype=float,
    )
    u_hat = np.array(
        [math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)],
        dtype=float,
    )
    return np.vstack((e_hat / np.linalg.norm(e_hat), n_hat / np.linalg.norm(n_hat), u_hat / np.linalg.norm(u_hat)))


def enu_error_components(dpos_ecef: np.ndarray, reference_ecef: np.ndarray) -> tuple[float, float, float]:
    enu = ecef_to_enu_matrix(reference_ecef) @ dpos_ecef
    return float(enu[0]), float(enu[1]), float(enu[2])


def project_to_fixed_height(pos_ecef: np.ndarray, target_height_m: float) -> np.ndarray:
    lat_deg, lon_deg, _ = pm.ecef2geodetic(pos_ecef[0], pos_ecef[1], pos_ecef[2], deg=True)
    x, y, z = pm.geodetic2ecef(lat_deg, lon_deg, target_height_m, deg=True)
    return np.array([x, y, z], dtype=float)


def make_horizontal_initial_guess_unprojected(rr_true: np.ndarray, offset_m: float, rng: np.random.Generator) -> np.ndarray:
    transform = ecef_to_enu_matrix(rr_true)
    theta = rng.uniform(0.0, 2.0 * math.pi)
    return rr_true + offset_m * (math.cos(theta) * transform[0] + math.sin(theta) * transform[1])


def solve_single_sat_ecefpos_cb_cdot_wls_batches(
    x0: np.ndarray,
    y_rho: np.ndarray,
    y_rhodot: np.ndarray,
    rs: np.ndarray,
    vs: np.ndarray,
    t_sec: np.ndarray,
    rr_true: np.ndarray,
    sigma_rho: float,
    sigma_rhodot: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    state = x0.astype(float).copy()
    x_hist: list[np.ndarray] = []
    pdiag_hist: list[np.ndarray] = []
    update_times: list[float] = []
    enu_err_hist: list[np.ndarray] = []
    iter_hist: list[int] = []
    cost_hist: list[float] = []
    num_obs_hist: list[int] = []
    converged_updates: list[bool] = []

    if WLS.use_cumulative:
        update_idx = list(range(WLS.batch_size, t_sec.size + 1, WLS.batch_size))
        if WLS.final_partial_update and (not update_idx or update_idx[-1] != t_sec.size):
            update_idx.append(t_sec.size)
        for end_idx in update_idx:
            state, covariance, iter_now, cost_now, converged_now = solve_one_wls_ecef_5state(
                state,
                y_rho[:end_idx],
                y_rhodot[:end_idx],
                rs[:, :end_idx],
                vs[:, :end_idx],
                t_sec[:end_idx],
                sigma_rho,
                sigma_rhodot,
            )
            dpos = state[:3] - rr_true
            x_hist.append(state.copy())
            pdiag_hist.append(np.diag(covariance).copy())
            update_times.append(float(t_sec[end_idx - 1]))
            enu_err_hist.append(np.array(enu_error_components(dpos, rr_true), dtype=float))
            iter_hist.append(iter_now)
            cost_hist.append(cost_now)
            num_obs_hist.append(end_idx)
            converged_updates.append(converged_now)
    else:
        start_idx = 0
        while start_idx < t_sec.size:
            end_idx = min(start_idx + WLS.batch_size, t_sec.size)
            state, covariance, iter_now, cost_now, converged_now = solve_one_wls_ecef_5state(
                state,
                y_rho[start_idx:end_idx],
                y_rhodot[start_idx:end_idx],
                rs[:, start_idx:end_idx],
                vs[:, start_idx:end_idx],
                t_sec[start_idx:end_idx],
                sigma_rho,
                sigma_rhodot,
            )
            dpos = state[:3] - rr_true
            x_hist.append(state.copy())
            pdiag_hist.append(np.diag(covariance).copy())
            update_times.append(float(t_sec[end_idx - 1]))
            enu_err_hist.append(np.array(enu_error_components(dpos, rr_true), dtype=float))
            iter_hist.append(iter_now)
            cost_hist.append(cost_now)
            num_obs_hist.append(end_idx - start_idx)
            converged_updates.append(converged_now)
            start_idx = end_idx

    rho_all, jac_rho_all = rho_jac_ecef(state[:3], rs)
    rd_all, jac_rd_all = rhodot_jac_ecef(state[:3], rs, vs)
    h_rho_all = np.column_stack((jac_rho_all, np.ones(t_sec.size), t_sec))
    h_rhodot_all = np.column_stack((jac_rd_all, np.zeros(t_sec.size), np.ones(t_sec.size)))
    residual_rho_all = y_rho - (rho_all + state[3] + state[4] * t_sec)
    residual_rhodot_all = y_rhodot - (rd_all + state[4])
    _, _, p_final = build_posterior_covariance_5state(state, rs, vs, t_sec, sigma_rho, sigma_rhodot)

    xhat = {"state": state, "pos": state[:3].copy(), "cb0": float(state[3]), "cdot": float(state[4])}
    info = {
        "P": p_final,
        "H_rho_all": h_rho_all,
        "H_rhodot_all": h_rhodot_all,
        "v_rho_all": residual_rho_all,
        "v_rhodot_all": residual_rhodot_all,
        "x_hist": np.vstack(x_hist) if x_hist else np.empty((0, 5)),
        "Pdiag_hist": np.vstack(pdiag_hist) if pdiag_hist else np.empty((0, 5)),
        "update_times": np.asarray(update_times, dtype=float),
        "enu_err_hist": np.vstack(enu_err_hist) if enu_err_hist else np.empty((0, 3)),
        "iter_hist": np.asarray(iter_hist, dtype=int),
        "cost_hist": np.asarray(cost_hist, dtype=float),
        "num_obs_hist": np.asarray(num_obs_hist, dtype=int),
        "num_updates": len(update_times),
        "converged_updates": np.asarray(converged_updates, dtype=bool),
        "all_updates_converged": bool(converged_updates) and all(converged_updates),
        "converged": bool(converged_updates) and converged_updates[-1],
    }
    return xhat, info


def compute_single_sat_crlb_ecef_cb_cdot(
    rr_ref: np.ndarray,
    rs: np.ndarray,
    vs: np.ndarray,
    t_sec: np.ndarray,
    sigma_rho: float,
    sigma_rhodot: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    _, jac_rho = rho_jac_ecef(rr_ref, rs)
    _, jac_rd = rhodot_jac_ecef(rr_ref, rs, vs)
    h_rho = np.column_stack((jac_rho, np.ones(t_sec.size), t_sec))
    h_rhodot = np.column_stack((jac_rd, np.zeros(t_sec.size), np.ones(t_sec.size)))
    h_whitened = np.vstack((h_rho / sigma_rho, h_rhodot / sigma_rhodot))
    fim = h_whitened.T @ h_whitened
    fim = 0.5 * (fim + fim.T)
    rank = int(np.linalg.matrix_rank(fim))
    cond = float(np.linalg.cond(fim))
    rcond = 1.0 / cond if np.isfinite(cond) and cond != 0.0 else 0.0
    if rank < fim.shape[0]:
        raise ValueError(f"FIM is rank-deficient: rank={rank} < {fim.shape[0]}")
    covariance = np.linalg.pinv(fim) if np.linalg.cond(fim) > 1.0e12 else np.linalg.inv(fim)
    covariance = 0.5 * (covariance + covariance.T)
    return covariance, {"J": fim, "Hrho": h_rho, "Hrd": h_rhodot, "rank": rank, "cond": cond, "rcond": rcond}


def make_obs_cfg(profile_name: str) -> dict[str, float | str]:
    if profile_name == "orbcomm_like":
        sigma_rho_m = 50.0
        sigma_rhodot_mps = 0.05
    elif profile_name == "optimistic_pnt":
        sigma_rho_m = 5.0
        sigma_rhodot_mps = 0.05
    else:
        raise ValueError(f"Unsupported noise profile: {profile_name}")
    return {
        "noise_profile": profile_name,
        "sigma_rho_m": sigma_rho_m,
        "sigma_rhodot_mps": sigma_rhodot_mps,
        "cb0_true_m": TRUTH.cb0_true_m,
        "cdot_true_mps": TRUTH.cdot_true_mps,
    }


def build_profile_geometry_summary(obs_cfg: dict[str, float | str], rr_true: np.ndarray, rs: np.ndarray, vs: np.ndarray, t_sec: np.ndarray) -> dict[str, Any]:
    crlb, crlb_info = compute_single_sat_crlb_ecef_cb_cdot(
        rr_true,
        rs,
        vs,
        t_sec,
        float(obs_cfg["sigma_rho_m"]),
        float(obs_cfg["sigma_rhodot_mps"]),
    )
    p_xyz = crlb[:3, :3]
    transform = ecef_to_enu_matrix(rr_true)
    p_enu = transform @ p_xyz @ transform.T
    sigma_state = np.sqrt(np.maximum(np.diag(crlb), 0.0))
    return {
        "CRLB": crlb,
        "crlb_info": crlb_info,
        "P_enu_crlb": p_enu,
        "crlb_sigma_state": sigma_state,
        "sigmaE_crlb": float(np.sqrt(max(p_enu[0, 0], 0.0))),
        "sigmaN_crlb": float(np.sqrt(max(p_enu[1, 1], 0.0))),
        "sigmaU_crlb": float(np.sqrt(max(p_enu[2, 2], 0.0))),
        "horiz_rms_crlb": float(np.sqrt(np.trace(p_enu[:2, :2]))),
        "rms3d_crlb": float(np.sqrt(np.trace(p_enu))),
    }


def build_error_ellipse_enu(
    east_north_samples_m: np.ndarray,
    probability: float = 0.90,
    point_count: int = 361,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(east_north_samples_m, axis=0)
    covariance = np.cov(east_north_samples_m, rowvar=False, bias=True)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    scale = math.sqrt(-2.0 * math.log(max(1.0 - probability, np.finfo(float).tiny)))
    angles = np.linspace(0.0, 2.0 * math.pi, point_count)
    unit_circle = np.vstack((np.cos(angles), np.sin(angles)))
    ellipse = mean.reshape(2, 1) + eigenvectors @ np.diag(scale * np.sqrt(eigenvalues)) @ unit_circle
    return ellipse, mean, covariance


def symmetric_plot_limit(*arrays: np.ndarray, padding: float = 0.15, minimum_half_span: float = 5.0) -> float:
    max_abs = 0.0
    for array in arrays:
        if array.size:
            max_abs = max(max_abs, float(np.max(np.abs(array))))
    return max(max_abs * (1.0 + padding), minimum_half_span)


def zoom_window_limits(
    east_north_samples_m: np.ndarray,
    ellipse_m: np.ndarray,
    padding: float = 0.35,
    minimum_span_m: float = 2.0,
) -> tuple[tuple[float, float], tuple[float, float]]:
    cloud = np.vstack((east_north_samples_m, ellipse_m.T))
    xmin, ymin = np.min(cloud, axis=0)
    xmax, ymax = np.max(cloud, axis=0)
    x_center = 0.5 * (xmin + xmax)
    y_center = 0.5 * (ymin + ymax)
    x_half = max(0.5 * (xmax - xmin) * (1.0 + padding), 0.5 * minimum_span_m)
    y_half = max(0.5 * (ymax - ymin) * (1.0 + padding), 0.5 * minimum_span_m)
    return (x_center - x_half, x_center + x_half), (y_center - y_half, y_center + y_half)


def add_zoom_inset(
    axis: Any,
    east_north_samples_m: np.ndarray,
    ellipse_m: np.ndarray,
    mean_en_m: np.ndarray,
    title: str,
) -> None:
    xlim, ylim = zoom_window_limits(east_north_samples_m, ellipse_m)
    inset = axis.inset_axes([0.57, 0.07, 0.38, 0.38])
    inset.scatter(east_north_samples_m[:, 0], east_north_samples_m[:, 1], s=20, alpha=0.7, color="#1f77b4", linewidths=0.0)
    inset.plot(ellipse_m[0], ellipse_m[1], color="#d62728", linewidth=1.8)
    inset.scatter([mean_en_m[0]], [mean_en_m[1]], marker="x", s=75, color="#2ca02c", linewidths=1.6)
    inset.set_xlim(*xlim)
    inset.set_ylim(*ylim)
    inset.set_aspect("equal", adjustable="box")
    inset.set_title(title, fontsize=9, pad=2.0)
    inset.grid(True, alpha=0.25)
    inset.tick_params(labelsize=8)
    axis.indicate_inset_zoom(inset, edgecolor="0.4", alpha=0.9)


def save_monte_carlo_scatter_plot(
    figure_path: Path,
    result: dict[str, Any],
    satellite_name: str,
    pass_index: int,
) -> None:
    valid = result["ok"]
    east_north = result["err_enu_all"][valid, :2]
    ellipse, mean_en, _ = build_error_ellipse_enu(east_north, probability=0.90)
    limit = symmetric_plot_limit(east_north, ellipse.T)

    figure, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)

    axes[0].scatter(east_north[:, 0], east_north[:, 1], s=18, alpha=0.55, color="#1f77b4", linewidths=0.0)
    axes[0].plot(ellipse[0], ellipse[1], color="#d62728", linewidth=2.0, label="90% ellipse")
    axes[0].scatter([0.0], [0.0], marker="*", s=180, color="#ff7f0e", label="True receiver")
    axes[0].scatter([mean_en[0]], [mean_en[1]], marker="x", s=90, color="#2ca02c", label="Sample mean")
    axes[0].axhline(0.0, color="0.75", linewidth=0.8)
    axes[0].axvline(0.0, color="0.75", linewidth=0.8)
    axes[0].set_title("Monte Carlo EN Error Scatter")
    axes[0].set_xlabel("East Error (m)")
    axes[0].set_ylabel("North Error (m)")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)
    add_zoom_inset(axes[0], east_north, ellipse, mean_en, "Zoomed error cloud")

    axes[1].scatter(east_north[:, 0], east_north[:, 1], s=18, alpha=0.45, color="#1f77b4", linewidths=0.0)
    axes[1].plot(ellipse[0], ellipse[1], color="#d62728", linewidth=2.0, label="90% ellipse")
    axes[1].scatter([0.0], [0.0], marker="*", s=180, color="#ff7f0e", label="Receiver")
    axes[1].scatter([mean_en[0]], [mean_en[1]], marker="x", s=90, color="#2ca02c", label="Mean estimate")
    axes[1].set_title("Receiver and EN Positioning Scatter")
    axes[1].set_xlabel("East in Local Receiver Frame (m)")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)
    add_zoom_inset(axes[1], east_north, ellipse, mean_en, "Zoomed estimates")

    for axis in axes:
        axis.set_xlim(-limit, limit)
        axis.set_ylim(-limit, limit)
        axis.set_aspect("equal", adjustable="box")

    figure.suptitle(f"{satellite_name} Pass {pass_index:02d} | {result['obs_cfg']['noise_profile']}")
    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)


def run_single_trial_case(
    obs_cfg: dict[str, float | str],
    geom_case: dict[str, Any],
    obs_atmosphere: dict[str, np.ndarray],
    x0: np.ndarray,
    rr_true: np.ndarray,
    rs: np.ndarray,
    vs: np.ndarray,
    t_sec: np.ndarray,
    rho_geom: np.ndarray,
    rd_geom: np.ndarray,
) -> dict[str, Any]:
    rng = np.random.default_rng(RUN.single_trial_seed)
    noise_rho = float(obs_cfg["sigma_rho_m"]) * rng.standard_normal(t_sec.size)
    noise_rhodot = float(obs_cfg["sigma_rhodot_mps"]) * rng.standard_normal(t_sec.size)
    y_rho = (
        rho_geom
        + float(obs_cfg["cb0_true_m"])
        + float(obs_cfg["cdot_true_mps"]) * t_sec
        + obs_atmosphere["total_delay_m"]
        + noise_rho
    )
    y_rhodot = rd_geom + float(obs_cfg["cdot_true_mps"]) + obs_atmosphere["total_delay_rate_mps"] + noise_rhodot

    xhat, info = solve_single_sat_ecefpos_cb_cdot_wls_batches(
        x0,
        y_rho,
        y_rhodot,
        rs,
        vs,
        t_sec,
        rr_true,
        float(obs_cfg["sigma_rho_m"]),
        float(obs_cfg["sigma_rhodot_mps"]),
    )
    if not info["converged"]:
        raise RuntimeError(f"Single-trial solver did not converge for profile {obs_cfg['noise_profile']}.")

    x_est = xhat["state"]
    dpos = x_est[:3] - rr_true
    dcb0 = float(x_est[3] - obs_cfg["cb0_true_m"])
    dcdot = float(x_est[4] - obs_cfg["cdot_true_mps"])
    dE, dN, dU = enu_error_components(dpos, rr_true)
    err_horiz = float(math.hypot(dE, dN))
    err_3d = float(np.linalg.norm(dpos))
    post_sigma = np.sqrt(np.maximum(np.diag(info["P"]), 0.0))

    return {
        "mode": "single_trial",
        "obs_cfg": obs_cfg,
        "geom_case": geom_case,
        "x_est": x_est,
        "dpos": dpos,
        "dE": dE,
        "dN": dN,
        "dU": dU,
        "dcb0": dcb0,
        "dcdot": dcdot,
        "err_horiz": err_horiz,
        "err_3d": err_3d,
        "post_sigma": post_sigma,
        "obs_atmosphere": obs_atmosphere,
        "info": info,
    }


def run_monte_carlo_case(
    obs_cfg: dict[str, float | str],
    geom_case: dict[str, Any],
    obs_atmosphere: dict[str, np.ndarray],
    x0: np.ndarray,
    rr_true: np.ndarray,
    rs: np.ndarray,
    vs: np.ndarray,
    t_sec: np.ndarray,
    rho_geom: np.ndarray,
    rd_geom: np.ndarray,
) -> dict[str, Any]:
    trial_count = MC.trial_count
    x_est_all = np.full((trial_count, 5), np.nan, dtype=float)
    err_state_all = np.full((trial_count, 5), np.nan, dtype=float)
    err_enu_all = np.full((trial_count, 3), np.nan, dtype=float)
    err_horiz_all = np.full(trial_count, np.nan, dtype=float)
    err_3d_all = np.full(trial_count, np.nan, dtype=float)
    cost_all = np.full(trial_count, np.nan, dtype=float)
    iter_last_all = np.full(trial_count, np.nan, dtype=float)
    fail_flag = np.zeros(trial_count, dtype=bool)
    conv_flag_all = np.zeros(trial_count, dtype=bool)

    print(f"\n====== Start Monte Carlo: N = {trial_count} | profile = {obs_cfg['noise_profile']} ======")
    for index in range(trial_count):
        rng = np.random.default_rng(MC.base_seed + index + 1)
        noise_rho = float(obs_cfg["sigma_rho_m"]) * rng.standard_normal(t_sec.size)
        noise_rhodot = float(obs_cfg["sigma_rhodot_mps"]) * rng.standard_normal(t_sec.size)
        y_rho = (
            rho_geom
            + float(obs_cfg["cb0_true_m"])
            + float(obs_cfg["cdot_true_mps"]) * t_sec
            + obs_atmosphere["total_delay_m"]
            + noise_rho
        )
        y_rhodot = rd_geom + float(obs_cfg["cdot_true_mps"]) + obs_atmosphere["total_delay_rate_mps"] + noise_rhodot

        try:
            xhat, info = solve_single_sat_ecefpos_cb_cdot_wls_batches(
                x0,
                y_rho,
                y_rhodot,
                rs,
                vs,
                t_sec,
                rr_true,
                float(obs_cfg["sigma_rho_m"]),
                float(obs_cfg["sigma_rhodot_mps"]),
            )
            conv_flag_all[index] = info["converged"]
            if not info["converged"]:
                fail_flag[index] = True
                continue

            x_est = xhat["state"]
            dpos = x_est[:3] - rr_true
            dcb0 = float(x_est[3] - obs_cfg["cb0_true_m"])
            dcdot = float(x_est[4] - obs_cfg["cdot_true_mps"])
            dE, dN, dU = enu_error_components(dpos, rr_true)

            x_est_all[index] = x_est
            err_state_all[index] = np.array([dpos[0], dpos[1], dpos[2], dcb0, dcdot], dtype=float)
            err_enu_all[index] = np.array([dE, dN, dU], dtype=float)
            err_horiz_all[index] = float(math.hypot(dE, dN))
            err_3d_all[index] = float(np.linalg.norm(dpos))
            if info["cost_hist"].size:
                cost_all[index] = float(info["cost_hist"][-1])
            if info["iter_hist"].size:
                iter_last_all[index] = float(info["iter_hist"][-1])
        except Exception as exc:
            fail_flag[index] = True
            if MC.stop_on_failure:
                raise RuntimeError(f"Monte Carlo failed at trial {index + 1}") from exc

        if (index + 1) in {1, trial_count} or (index + 1) % max(1, round(trial_count / 10)) == 0:
            print(f"[{obs_cfg['noise_profile']}] progress: {index + 1:4d} / {trial_count:4d}")

    ok = (~fail_flag) & conv_flag_all & np.all(np.isfinite(err_state_all), axis=1)
    ok_count = int(np.sum(ok))
    if ok_count < 5:
        raise RuntimeError(f"Too few valid Monte Carlo samples for profile {obs_cfg['noise_profile']}: {ok_count}")

    e_state = err_state_all[ok]
    e_enu = err_enu_all[ok]
    e_h = err_horiz_all[ok]
    e_3d = err_3d_all[ok]

    bias_state = np.mean(e_state, axis=0)
    cov_state = np.cov(e_state, rowvar=False, bias=True)
    rmse_state = np.sqrt(np.mean(e_state**2, axis=0))
    bias_enu = np.mean(e_enu, axis=0)
    cov_enu = np.cov(e_enu, rowvar=False, bias=True)
    rmse_enu = np.sqrt(np.mean(e_enu**2, axis=0))
    rms_horiz = float(np.sqrt(np.mean(e_h**2)))
    rms_3d = float(np.sqrt(np.mean(e_3d**2)))

    return {
        "mode": "monte_carlo",
        "obs_cfg": obs_cfg,
        "geom_case": geom_case,
        "ok": ok,
        "Nok": ok_count,
        "Nfail": int(trial_count - ok_count),
        "x_est_all": x_est_all,
        "err_state_all": err_state_all,
        "err_enu_all": err_enu_all,
        "err_horiz_all": err_horiz_all,
        "err_3d_all": err_3d_all,
        "cost_all": cost_all,
        "iter_last_all": iter_last_all,
        "fail_flag": fail_flag,
        "conv_flag_all": conv_flag_all,
        "bias_state": bias_state,
        "cov_state": cov_state,
        "rmse_state": rmse_state,
        "bias_enu": bias_enu,
        "cov_enu": cov_enu,
        "rmse_enu": rmse_enu,
        "rms_horiz": rms_horiz,
        "rms_3d": rms_3d,
        "mean_horiz": float(np.mean(e_h)),
        "mean_3d": float(np.mean(e_3d)),
        "ratio_rmse_state": rmse_state / geom_case["crlb_sigma_state"],
        "ratio_rmse_E": float(rmse_enu[0] / geom_case["sigmaE_crlb"]),
        "ratio_rmse_N": float(rmse_enu[1] / geom_case["sigmaN_crlb"]),
        "ratio_rmse_U": float(rmse_enu[2] / geom_case["sigmaU_crlb"]),
        "ratio_horiz": float(rms_horiz / geom_case["horiz_rms_crlb"]),
        "ratio_3d": float(rms_3d / geom_case["rms3d_crlb"]),
        "obs_atmosphere": obs_atmosphere,
    }


def json_summary(result: dict[str, Any], satellite_name: str, pass_index: int, selection: SelectionConfig) -> dict[str, Any]:
    obs_atmosphere_summary = summarize_observation_atmosphere(result["obs_atmosphere"])
    summary = {
        "satellite_name": satellite_name,
        "pass_index": pass_index,
        "orbit_source": selection.orbit_source,
        "mode": result["mode"],
        "noise_profile": result["obs_cfg"]["noise_profile"],
        "atmosphere": {
            "enable_klobuchar": ATM.enable_klobuchar,
            "enable_hopfield": ATM.enable_hopfield,
            "signal_frequency_hz": scalar_number(result["obs_atmosphere"]["signal_frequency_hz"]),
            "klobuchar_nav_file": ATM.klobuchar_nav_file,
            "klobuchar_source": scalar_string(result["obs_atmosphere"]["klobuchar_source"]),
            "klobuchar_alpha": np.asarray(result["obs_atmosphere"]["klobuchar_alpha"], dtype=float).tolist(),
            "klobuchar_beta": np.asarray(result["obs_atmosphere"]["klobuchar_beta"], dtype=float).tolist(),
            **obs_atmosphere_summary,
        },
    }
    if result["mode"] == "single_trial":
        summary.update(
            {
                "err_horiz_m": result["err_horiz"],
                "err_3d_m": result["err_3d"],
                "dE_m": result["dE"],
                "dN_m": result["dN"],
                "dU_m": result["dU"],
                "dcb0_m": result["dcb0"],
                "dcdot_mps": result["dcdot"],
                "post_sigma": result["post_sigma"].tolist(),
                "postfit_rho_rms_m": rms(result["info"]["v_rho_all"]),
                "postfit_rhodot_rms_mps": rms(result["info"]["v_rhodot_all"]),
                "last_iterations": int(result["info"]["iter_hist"][-1]),
                "final_cost": float(result["info"]["cost_hist"][-1]),
            }
        )
    else:
        summary.update(
            {
                "valid_trials": int(result["Nok"]),
                "failed_trials": int(result["Nfail"]),
                "rms_horiz_m": result["rms_horiz"],
                "rms_3d_m": result["rms_3d"],
                "mean_horiz_m": result["mean_horiz"],
                "mean_3d_m": result["mean_3d"],
                "ratio_horiz": result["ratio_horiz"],
                "ratio_3d": result["ratio_3d"],
                "ratio_rmse_state": result["ratio_rmse_state"].tolist(),
                "ratio_rmse_E": result["ratio_rmse_E"],
                "ratio_rmse_N": result["ratio_rmse_N"],
                "ratio_rmse_U": result["ratio_rmse_U"],
            }
        )
    return summary


def save_result_files(result: dict[str, Any], satellite_name: str, pass_index: int, selection: SelectionConfig) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    profile_tag = sanitize_tag(str(result["obs_cfg"]["noise_profile"]))
    satellite_tag = sanitize_tag(satellite_name)
    stem = f"{satellite_tag}_pass_{pass_index:02d}_{profile_tag}_{result['mode']}"
    summary_path = OUTPUT_DIR / f"{stem}.json"
    summary_path.write_text(json.dumps(json_summary(result, satellite_name, pass_index, selection), indent=2), encoding="utf-8")

    if result["mode"] == "single_trial":
        raw_arrays = {
            "x_est": result["x_est"],
            "dpos": result["dpos"],
            "post_sigma": result["post_sigma"],
            "azimuth_deg": result["obs_atmosphere"]["azimuth_deg"],
            "elevation_deg": result["obs_atmosphere"]["elevation_deg"],
            "signal_frequency_hz": result["obs_atmosphere"]["signal_frequency_hz"],
            "klobuchar_alpha": result["obs_atmosphere"]["klobuchar_alpha"],
            "klobuchar_beta": result["obs_atmosphere"]["klobuchar_beta"],
            "iono_delay_m": result["obs_atmosphere"]["iono_delay_m"],
            "tropo_delay_m": result["obs_atmosphere"]["tropo_delay_m"],
            "total_delay_m": result["obs_atmosphere"]["total_delay_m"],
            "total_delay_rate_mps": result["obs_atmosphere"]["total_delay_rate_mps"],
            "v_rho_all": result["info"]["v_rho_all"],
            "v_rhodot_all": result["info"]["v_rhodot_all"],
            "iter_hist": result["info"]["iter_hist"],
            "cost_hist": result["info"]["cost_hist"],
        }
    else:
        raw_arrays = {
            "x_est_all": result["x_est_all"],
            "err_state_all": result["err_state_all"],
            "err_enu_all": result["err_enu_all"],
            "err_horiz_all": result["err_horiz_all"],
            "err_3d_all": result["err_3d_all"],
            "cost_all": result["cost_all"],
            "iter_last_all": result["iter_last_all"],
            "azimuth_deg": result["obs_atmosphere"]["azimuth_deg"],
            "elevation_deg": result["obs_atmosphere"]["elevation_deg"],
            "signal_frequency_hz": result["obs_atmosphere"]["signal_frequency_hz"],
            "klobuchar_alpha": result["obs_atmosphere"]["klobuchar_alpha"],
            "klobuchar_beta": result["obs_atmosphere"]["klobuchar_beta"],
            "iono_delay_m": result["obs_atmosphere"]["iono_delay_m"],
            "tropo_delay_m": result["obs_atmosphere"]["tropo_delay_m"],
            "total_delay_m": result["obs_atmosphere"]["total_delay_m"],
            "total_delay_rate_mps": result["obs_atmosphere"]["total_delay_rate_mps"],
            "ok": result["ok"].astype(np.int8),
        }
    np.savez_compressed(OUTPUT_DIR / f"{stem}.npz", **raw_arrays)

    if result["mode"] == "monte_carlo":
        save_monte_carlo_scatter_plot(
            OUTPUT_DIR / f"{stem}_scatter.png",
            result,
            satellite_name,
            pass_index,
        )


def main() -> int:
    try:
        args = parse_cli_args()
        if args.list_passes:
            list_available_passes(STEP1_INDEX_FILE)
            return 0

        selection = SelectionConfig(
            satellite_name=args.satellite_name,
            pass_index=int(args.pass_index),
            orbit_source=args.orbit_source.upper(),
        )
        pass_record = load_pass_record(STEP1_INDEX_FILE, selection.satellite_name, selection.pass_index)
        pass_data = load_step1_pass(Path(pass_record["file"]), selection.orbit_source)

        receiver_cfg = ReceiverConfig(
            latitude_deg=float(pass_data["receiver_lla_deg_m"][0]),
            longitude_deg=float(pass_data["receiver_lla_deg_m"][1]),
            altitude_m=float(pass_data["receiver_lla_deg_m"][2]),
        )
        rr_true = np.asarray(pass_data["receiver_ecef_m"], dtype=float).reshape(3)
        lat_deg, lon_deg, h_true = pm.ecef2geodetic(rr_true[0], rr_true[1], rr_true[2], deg=True)
        t_sec, rs, vs, obs_start_offset_sec = slice_observation_window(
            pass_data["time_seconds"],
            pass_data["rs_ecef"],
            pass_data["vs_ecef"],
        )
        obs_start_dt = datetime.fromisoformat(pass_data["pass_start_utc"]) + timedelta(seconds=obs_start_offset_sec)
        obs_start_utc = obs_start_dt.isoformat()
        rho_geom, _ = rho_jac_ecef(rr_true, rs)
        rd_geom, _ = rhodot_jac_ecef(rr_true, rs, vs)
        signal_frequency_hz = resolve_signal_frequency_hz(pass_data["satellite_name"])
        obs_atmosphere = build_observation_atmosphere(receiver_cfg, obs_start_utc, t_sec, rs, signal_frequency_hz)
        obs_atmosphere_summary = summarize_observation_atmosphere(obs_atmosphere)

        print(f"Selected satellite: {pass_data['satellite_name']} | pass {pass_data['pass_index']:02d}")
        print(f"Orbit source for synthetic observations: {selection.orbit_source}")
        print(f"Signal frequency: {signal_frequency_hz / 1.0e6:.3f} MHz")
        print(f"Receiver LLA [deg, deg, m]: [{lat_deg:.6f}, {lon_deg:.6f}, {h_true:.3f}]")
        print(f"Number of epochs used: {t_sec.size}")
        print(f"Pass interval: {pass_data['pass_start_utc']} -> {pass_data['pass_end_utc']}")
        print(f"Observation interval start: {obs_start_utc}")
        print(f"Klobuchar coefficient source: {scalar_string(obs_atmosphere['klobuchar_source'])}")
        print(f"RMS geometry self-check rho = {rms(rho_geom - rho_geom):.3e} m")
        print(f"RMS geometry self-check rhodot = {rms(rd_geom - rd_geom):.3e} m/s")
        print(
            "Atmospheric delays | "
            f"iono mean/max = {obs_atmosphere_summary['iono_mean_m']:.3f}/{obs_atmosphere_summary['iono_max_m']:.3f} m | "
            f"tropo mean/max = {obs_atmosphere_summary['tropo_mean_m']:.3f}/{obs_atmosphere_summary['tropo_max_m']:.3f} m | "
            f"total rate RMS = {obs_atmosphere_summary['total_rate_rms_mps']:.6f} m/s"
        )

        init_rng = np.random.default_rng(12345)
        pos0 = make_horizontal_initial_guess_unprojected(rr_true, INIT.initial_offset_km * 1.0e3, init_rng)
        pos0 = project_to_fixed_height(pos0, h_true)
        x0 = np.array([pos0[0], pos0[1], pos0[2], INIT.cb0_m, INIT.cdot0_mps], dtype=float)
        dE0, dN0, dU0 = enu_error_components(pos0 - rr_true, rr_true)
        print(f"Initial ENU error [m]: [{dE0:+.3f}, {dN0:+.3f}, {dU0:+.6f}]")
        print(f"Initial horizontal error [km]: {math.hypot(dE0, dN0) / 1.0e3:.3f}")

        results: list[dict[str, Any]] = []
        for profile_name in NOISE_PROFILES:
            obs_cfg = make_obs_cfg(profile_name)
            geom_case = build_profile_geometry_summary(obs_cfg, rr_true, rs, vs, t_sec)

            print("\n============================================================")
            print(f"Noise profile: {profile_name}")
            print(
                f"sigma_rho = {obs_cfg['sigma_rho_m']:.3f} m | "
                f"sigma_rhodot = {obs_cfg['sigma_rhodot_mps']:.3f} m/s"
            )
            print(
                "CRLB sigma_xyzcbcdot = "
                f"{geom_case['crlb_sigma_state'][0]:.6f}, "
                f"{geom_case['crlb_sigma_state'][1]:.6f}, "
                f"{geom_case['crlb_sigma_state'][2]:.6f}, "
                f"{geom_case['crlb_sigma_state'][3]:.6f}, "
                f"{geom_case['crlb_sigma_state'][4]:.6e}"
            )

            if RUN.single_trial_only:
                result = run_single_trial_case(obs_cfg, geom_case, obs_atmosphere, x0, rr_true, rs, vs, t_sec, rho_geom, rd_geom)
                print(
                    f"Single trial | horiz = {result['err_horiz']:.6f} m | "
                    f"3D = {result['err_3d']:.6f} m | "
                    f"rho RMS = {rms(result['info']['v_rho_all']):.6f} m | "
                    f"rhodot RMS = {rms(result['info']['v_rhodot_all']):.6f} m/s"
                )
            else:
                result = run_monte_carlo_case(
                    obs_cfg,
                    geom_case,
                    obs_atmosphere,
                    x0,
                    rr_true,
                    rs,
                    vs,
                    t_sec,
                    rho_geom,
                    rd_geom,
                )
                print(
                    f"Monte Carlo | valid = {result['Nok']:4d}/{MC.trial_count:4d} | "
                    f"horiz RMS = {result['rms_horiz']:.6f} m | "
                    f"3D RMS = {result['rms_3d']:.6f} m | "
                    f"ratio_h = {result['ratio_horiz']:.4f} | ratio_3d = {result['ratio_3d']:.4f}"
                )

            results.append(result)
            if RUN.save_results:
                save_result_files(result, pass_data["satellite_name"], pass_data["pass_index"], selection)

        if RUN.save_results:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            sweep_summary = {
                "receiver": asdict(receiver_cfg),
                "selection": asdict(selection),
                "data": asdict(DATA),
                "truth": asdict(TRUTH),
                "init": asdict(INIT),
                "wls": asdict(WLS),
                "monte_carlo": asdict(MC),
                "atmosphere": asdict(ATM),
                "run": asdict(RUN),
                "profiles": [
                    json_summary(result, pass_data["satellite_name"], pass_data["pass_index"], selection)
                    for result in results
                ],
            }
            summary_file = OUTPUT_DIR / f"{sanitize_tag(pass_data['satellite_name'])}_pass_{pass_data['pass_index']:02d}_summary.json"
            summary_file.write_text(json.dumps(sweep_summary, indent=2), encoding="utf-8")
            print(f"\nSaved WLS summaries to {OUTPUT_DIR}")

        return 0
    except Exception as exc:  # pragma: no cover - command line entry point
        print(f"WLS positioning failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
