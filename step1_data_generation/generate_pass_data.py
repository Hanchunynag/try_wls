from __future__ import annotations

import json
import math
import re
import sys
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import jpype
import numpy as np
import orekit_jpype
import requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TLE_FILE = PROJECT_ROOT / "tle.tle"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OREKIT_DATA_ZIP = Path(__file__).resolve().parent / "orekit-data.zip"
OREKIT_DATA_URL = "https://gitlab.orekit.org/orekit/orekit-data/-/archive/main/orekit-data-main.zip"


@dataclass(frozen=True)
class ScenarioConfig:
    start_time_utc: datetime = datetime(2026, 1, 15, 3, 30, 0, tzinfo=timezone.utc)
    end_time_utc: datetime = datetime(2026, 1, 15, 4, 0, 0, tzinfo=timezone.utc)
    coarse_step_sec: float = 1.0
    fine_step_sec: float = 0.01
    buffer_sec: float = 10.0
    min_elevation_deg: float = 10.0
    min_pass_duration_sec: float = 3.0


@dataclass(frozen=True)
class ReceiverConfig:
    latitude_deg: float = 45.772625
    longitude_deg: float = 126.682625
    altitude_m: float = 154.0


@dataclass(frozen=True)
class HpopConfig:
    mass_kg: float = 260.0
    drag_area_m2: float = 1.5
    drag_coefficient: float = 2.2
    gravity_degree: int = 70
    gravity_order: int = 70
    position_tolerance_m: float = 10.0
    min_step_sec: float = 1.0e-3
    max_step_sec: float = 300.0


SCENARIO = ScenarioConfig()
RECEIVER = ReceiverConfig()
HPOP = HpopConfig()


def download_file(url: str, destination: Path, max_retries: int = 3) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            with requests.get(url, stream=True, timeout=60) as response:
                response.raise_for_status()
                with destination.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            with zipfile.ZipFile(destination) as archive:
                if archive.testzip() is not None:
                    raise zipfile.BadZipFile("Corrupted entry found while validating orekit-data.zip.")
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


def absolute_date_helpers() -> tuple[Any, Any]:
    from orekit_jpype.pyhelpers import absolutedate_to_datetime, datetime_to_absolutedate

    return absolutedate_to_datetime, datetime_to_absolutedate


def read_tle_catalog(tle_path: Path) -> list[dict[str, str]]:
    lines = [line.rstrip("\n") for line in tle_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) % 3 != 0:
        raise ValueError(f"TLE file {tle_path} does not contain complete 3-line sets.")

    catalog: list[dict[str, str]] = []
    for index in range(0, len(lines), 3):
        catalog.append(
            {
                "name": lines[index].strip(),
                "line1": lines[index + 1].strip(),
                "line2": lines[index + 2].strip(),
            }
        )
    return catalog


def make_datetime_grid(start_time: datetime, end_time: datetime, step_sec: float) -> list[datetime]:
    total_seconds = (end_time - start_time).total_seconds()
    sample_count = int(round(total_seconds / step_sec)) + 1
    return [start_time + timedelta(seconds=index * step_sec) for index in range(sample_count)]


def vector3_to_numpy(vector: Any) -> np.ndarray:
    return np.array([vector.getX(), vector.getY(), vector.getZ()], dtype=float)


def pv_to_numpy(pv_coordinates: Any) -> tuple[np.ndarray, np.ndarray]:
    return vector3_to_numpy(pv_coordinates.getPosition()), vector3_to_numpy(pv_coordinates.getVelocity())


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


def find_pass_windows(visible_mask: np.ndarray, coarse_step_sec: float, min_duration_sec: float) -> list[tuple[int, int]]:
    windows: list[tuple[int, int]] = []
    in_window = False
    start_idx = 0

    for idx, is_visible in enumerate(visible_mask):
        if is_visible and not in_window:
            in_window = True
            start_idx = idx
        elif not is_visible and in_window:
            end_idx = idx - 1
            duration = (end_idx - start_idx) * coarse_step_sec
            if duration >= min_duration_sec:
                windows.append((start_idx, end_idx))
            in_window = False

    if in_window:
        end_idx = len(visible_mask) - 1
        duration = (end_idx - start_idx) * coarse_step_sec
        if duration >= min_duration_sec:
            windows.append((start_idx, end_idx))

    return windows


def create_frames(receiver: ReceiverConfig) -> dict[str, Any]:
    from org.orekit.bodies import GeodeticPoint, OneAxisEllipsoid
    from org.orekit.frames import FramesFactory, TopocentricFrame
    from org.orekit.utils import Constants, IERSConventions

    itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
    eme2000 = FramesFactory.getEME2000()
    earth = OneAxisEllipsoid(
        Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
        Constants.WGS84_EARTH_FLATTENING,
        itrf,
    )
    site = GeodeticPoint(
        math.radians(receiver.latitude_deg),
        math.radians(receiver.longitude_deg),
        receiver.altitude_m,
    )
    station = TopocentricFrame(earth, site, "receiver")
    receiver_ecef = vector3_to_numpy(earth.transform(site))
    return {
        "earth_fixed": itrf,
        "inertial": eme2000,
        "earth": earth,
        "station": station,
        "receiver_ecef_m": receiver_ecef,
    }


def create_tle(name: str, line1: str, line2: str) -> Any:
    from org.orekit.propagation.analytical.tle import TLE

    _ = name
    return TLE(line1, line2)


def create_sgp4_propagator(tle: Any) -> Any:
    from org.orekit.propagation.analytical.tle import TLEPropagator

    return TLEPropagator.selectExtrapolator(tle)


def create_hpop_propagator(initial_state: Any, earth_fixed_frame: Any, config: HpopConfig) -> Any:
    from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
    from org.orekit.bodies import CelestialBodyFactory, OneAxisEllipsoid
    from org.orekit.forces.drag import DragForce, IsotropicDrag
    from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel
    from org.orekit.forces.gravity.potential import GravityFieldFactory
    from org.orekit.models.earth.atmosphere import HarrisPriester
    from org.orekit.orbits import OrbitType
    from org.orekit.propagation.numerical import NumericalPropagator
    from org.orekit.utils import Constants

    orbit = initial_state.getOrbit()
    tolerances = NumericalPropagator.tolerances(config.position_tolerance_m, orbit, OrbitType.CARTESIAN)
    integrator = DormandPrince853Integrator(config.min_step_sec, config.max_step_sec, tolerances[0], tolerances[1])
    propagator = NumericalPropagator(integrator)
    propagator.setOrbitType(OrbitType.CARTESIAN)
    propagator.setInitialState(initial_state)

    gravity_provider = GravityFieldFactory.getNormalizedProvider(config.gravity_degree, config.gravity_order)
    propagator.addForceModel(HolmesFeatherstoneAttractionModel(earth_fixed_frame, gravity_provider))

    earth = OneAxisEllipsoid(
        Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
        Constants.WGS84_EARTH_FLATTENING,
        earth_fixed_frame,
    )
    atmosphere = HarrisPriester(CelestialBodyFactory.getSun(), earth)
    spacecraft = IsotropicDrag(config.drag_area_m2, config.drag_coefficient)
    propagator.addForceModel(DragForce(atmosphere, spacecraft))
    return propagator


def generate_satellite_pass_files() -> list[dict[str, Any]]:
    ensure_orekit_ready(OREKIT_DATA_ZIP)
    absolutedate_to_datetime, datetime_to_absolutedate = absolute_date_helpers()
    frames = create_frames(RECEIVER)
    output_records: list[dict[str, Any]] = []

    start_buffered = SCENARIO.start_time_utc - timedelta(seconds=SCENARIO.buffer_sec)
    end_buffered = SCENARIO.end_time_utc + timedelta(seconds=SCENARIO.buffer_sec)
    coarse_datetimes = make_datetime_grid(start_buffered, end_buffered, SCENARIO.coarse_step_sec)
    coarse_dates = [datetime_to_absolutedate(value) for value in coarse_datetimes]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tle_catalog = read_tle_catalog(TLE_FILE)
    print(f"Loaded {len(tle_catalog)} TLE records from {TLE_FILE}")

    from org.orekit.orbits import CartesianOrbit
    from org.orekit.propagation import SpacecraftState
    from org.orekit.utils import Constants

    for sat_idx, tle_entry in enumerate(tle_catalog, start=1):
        tle = create_tle(tle_entry["name"], tle_entry["line1"], tle_entry["line2"])
        sgp4 = create_sgp4_propagator(tle)

        coarse_elevations = np.zeros(len(coarse_dates), dtype=float)
        for index, date in enumerate(coarse_dates):
            pv_ecef = sgp4.getPVCoordinates(date, frames["earth_fixed"])
            coarse_elevations[index] = math.degrees(
                frames["station"].getElevation(pv_ecef.getPosition(), frames["earth_fixed"], date)
            )

        visible_windows = find_pass_windows(
            coarse_elevations > SCENARIO.min_elevation_deg,
            SCENARIO.coarse_step_sec,
            SCENARIO.min_pass_duration_sec,
        )
        if not visible_windows:
            print(f"[{sat_idx:02d}/{len(tle_catalog):02d}] {tle_entry['name']}: no valid pass in the scenario window")
            continue

        tle_epoch = tle.getDate()
        tle_epoch_dt = absolutedate_to_datetime(tle_epoch, tz_aware=True)
        tle_age_hours = max((SCENARIO.start_time_utc - tle_epoch_dt).total_seconds() / 3600.0, 0.0)

        for pass_number, (start_idx, end_idx) in enumerate(visible_windows, start=1):
            pass_start_dt = coarse_datetimes[start_idx]
            pass_end_dt = coarse_datetimes[end_idx]
            pass_duration_sec = (pass_end_dt - pass_start_dt).total_seconds()

            segment_start_dt = max(start_buffered, pass_start_dt - timedelta(seconds=SCENARIO.buffer_sec))
            segment_end_dt = min(end_buffered, pass_end_dt + timedelta(seconds=SCENARIO.buffer_sec))

            segment_start_date = datetime_to_absolutedate(segment_start_dt)
            segment_end_date = datetime_to_absolutedate(segment_end_dt)
            pass_start_date = datetime_to_absolutedate(pass_start_dt)
            pass_end_date = datetime_to_absolutedate(pass_end_dt)

            initial_pv = sgp4.getPVCoordinates(tle_epoch, frames["inertial"])
            initial_orbit = CartesianOrbit(initial_pv, frames["inertial"], tle_epoch, Constants.WGS84_EARTH_MU)
            initial_state = SpacecraftState(initial_orbit, HPOP.mass_kg)
            hpop_bridge = create_hpop_propagator(initial_state, frames["earth_fixed"], HPOP)
            segment_start_state = hpop_bridge.propagate(segment_start_date)

            segment_hpop = create_hpop_propagator(segment_start_state, frames["earth_fixed"], HPOP)
            ephemeris_generator = segment_hpop.getEphemerisGenerator()
            segment_hpop.propagate(segment_end_date)
            hpop_ephemeris = ephemeris_generator.getGeneratedEphemeris()

            fine_datetimes = make_datetime_grid(pass_start_dt, pass_end_dt, SCENARIO.fine_step_sec)
            fine_dates = [datetime_to_absolutedate(value) for value in fine_datetimes]
            time_seconds = np.array(
                [(value - pass_start_dt).total_seconds() for value in fine_datetimes],
                dtype=float,
            )

            hpop_eci_pos = np.zeros((len(fine_dates), 3), dtype=float)
            hpop_eci_vel = np.zeros((len(fine_dates), 3), dtype=float)
            hpop_ecef_pos = np.zeros((len(fine_dates), 3), dtype=float)
            hpop_ecef_vel = np.zeros((len(fine_dates), 3), dtype=float)
            sgp4_eci_pos = np.zeros((len(fine_dates), 3), dtype=float)
            sgp4_eci_vel = np.zeros((len(fine_dates), 3), dtype=float)
            sgp4_ecef_pos = np.zeros((len(fine_dates), 3), dtype=float)
            sgp4_ecef_vel = np.zeros((len(fine_dates), 3), dtype=float)
            elevation_deg = np.zeros(len(fine_dates), dtype=float)

            for index, date in enumerate(fine_dates):
                hpop_state = hpop_ephemeris.propagate(date)
                hpop_eci_pv = hpop_state.getPVCoordinates(frames["inertial"])
                hpop_ecef_pv = hpop_state.getPVCoordinates(frames["earth_fixed"])
                sgp4_eci_pv = sgp4.getPVCoordinates(date, frames["inertial"])
                sgp4_ecef_pv = sgp4.getPVCoordinates(date, frames["earth_fixed"])

                hpop_eci_pos[index], hpop_eci_vel[index] = pv_to_numpy(hpop_eci_pv)
                hpop_ecef_pos[index], hpop_ecef_vel[index] = pv_to_numpy(hpop_ecef_pv)
                sgp4_eci_pos[index], sgp4_eci_vel[index] = pv_to_numpy(sgp4_eci_pv)
                sgp4_ecef_pos[index], sgp4_ecef_vel[index] = pv_to_numpy(sgp4_ecef_pv)
                elevation_deg[index] = math.degrees(
                    frames["station"].getElevation(hpop_ecef_pv.getPosition(), frames["earth_fixed"], date)
                )

            file_stem = f"{safe_name(tle_entry['name'])}_pass_{pass_number:02d}"
            output_file = OUTPUT_DIR / f"{file_stem}.npz"
            np.savez_compressed(
                output_file,
                satellite_name=np.array(tle_entry["name"]),
                catalog_id=np.array(int(tle.getSatelliteNumber())),
                pass_index=np.array(pass_number),
                tle_line1=np.array(tle_entry["line1"]),
                tle_line2=np.array(tle_entry["line2"]),
                tle_epoch_utc=np.array(tle_epoch_dt.isoformat()),
                tle_age_hours=np.array(tle_age_hours),
                scenario_start_utc=np.array(SCENARIO.start_time_utc.isoformat()),
                scenario_end_utc=np.array(SCENARIO.end_time_utc.isoformat()),
                pass_start_utc=np.array(pass_start_dt.isoformat()),
                pass_end_utc=np.array(pass_end_dt.isoformat()),
                receiver_lla_deg_m=np.array(
                    [RECEIVER.latitude_deg, RECEIVER.longitude_deg, RECEIVER.altitude_m],
                    dtype=float,
                ),
                receiver_ecef_m=frames["receiver_ecef_m"],
                time_seconds=time_seconds,
                elevation_deg=elevation_deg,
                hpop_eci_pos_m=hpop_eci_pos,
                hpop_eci_vel_mps=hpop_eci_vel,
                hpop_ecef_pos_m=hpop_ecef_pos,
                hpop_ecef_vel_mps=hpop_ecef_vel,
                sgp4_eci_pos_m=sgp4_eci_pos,
                sgp4_eci_vel_mps=sgp4_eci_vel,
                sgp4_ecef_pos_m=sgp4_ecef_pos,
                sgp4_ecef_vel_mps=sgp4_ecef_vel,
            )

            record = {
                "satellite_name": tle_entry["name"],
                "catalog_id": int(tle.getSatelliteNumber()),
                "pass_index": pass_number,
                "pass_start_utc": pass_start_dt.isoformat(),
                "pass_end_utc": pass_end_dt.isoformat(),
                "duration_sec": pass_duration_sec,
                "tle_epoch_utc": tle_epoch_dt.isoformat(),
                "tle_age_hours": tle_age_hours,
                "file": str(output_file.resolve()),
            }
            output_records.append(record)
            print(
                f"[{sat_idx:02d}/{len(tle_catalog):02d}] {tle_entry['name']}: "
                f"saved pass {pass_number:02d} ({pass_duration_sec:.1f} s) -> {output_file.name}"
            )

    scenario_summary = {
        "scenario": {
            "start_time_utc": SCENARIO.start_time_utc.isoformat(),
            "end_time_utc": SCENARIO.end_time_utc.isoformat(),
            "coarse_step_sec": SCENARIO.coarse_step_sec,
            "fine_step_sec": SCENARIO.fine_step_sec,
            "buffer_sec": SCENARIO.buffer_sec,
            "min_elevation_deg": SCENARIO.min_elevation_deg,
            "min_pass_duration_sec": SCENARIO.min_pass_duration_sec,
        },
        "receiver": asdict(RECEIVER),
        "hpop": asdict(HPOP),
        "tle_file": str(TLE_FILE.resolve()),
        "orekit_data_zip": str(OREKIT_DATA_ZIP.resolve()),
        "passes": output_records,
    }
    summary_file = OUTPUT_DIR / "passes_index.json"
    summary_file.write_text(json.dumps(scenario_summary, indent=2), encoding="utf-8")
    print(f"Wrote summary index -> {summary_file}")
    return output_records


def main() -> int:
    try:
        records = generate_satellite_pass_files()
        if not records:
            print("No pass files were generated for the configured time window.")
        return 0
    except Exception as exc:  # pragma: no cover - command line entry point
        print(f"Step 1 failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
