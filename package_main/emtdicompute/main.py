import time
import shutil
from pathlib import Path

import pandas as pd
import torch

from emtdicompute.batch import pipeline
from emtdicompute.camera_handler import camera
from emtdicompute.scene import sample_mesh
from emtdicompute.embodied_3d_isovist._parallel import validate_num_workers
from emtdicompute.utils.device import resolve_device
import emtdicompute.utils.datatypes as dc


def _process_record_file(
    file_path: Path,
    mesh: dc.SampleMesh,
    chunk_size: int,
    hfov: float,
    vfov: float,
    device: str,
    temp_dir: str = None,
    metrics: list[str] | tuple[str, ...] | None = None,
    num_workers: int = 1,
) -> tuple[pd.DataFrame | None, dict]:
    df = pd.read_csv(file_path)
    if df.empty:
        print(f"[em3disovist] {file_path.name} is empty; skipping.")
        return None, {}

    pid = df["PlayerID"].iloc[0]
    bldg = df["Scene"].iloc[0]
    task = df["Task"].iloc[0]

    cam_orig, cam_forward = camera.extract_camera_from_record(df)
    cam_records = camera.combine_camera_pos_and_dir(cam_orig, cam_forward)

    if "Timestamp" not in df.columns:
        raise KeyError(f"[em3disovist] Timestamp column missing in {file_path.name}")
    timestamps = df["Timestamp"].to_numpy()

    results = pipeline.compute_em3di_metrics_batch(
        mesh,
        cam_records,
        chunk_size,
        hfov=hfov,
        vfov=vfov,
        device=device,
        timestamps=timestamps,
        temp_dir = temp_dir,
        save_each_chunk=True,
        metrics=metrics,
        num_workers=num_workers,
    )
    return results, {"pid": pid, "bldg": bldg, "task": task}


def run_batch_job(
    records_dir: Path = None,
    mesh_path: str = None,
    sampling_density: int = 100,
    chunk_size: int = 10,
    hfov: float = 90,
    vfov: float = 60,
    temp_dir: str = None,
    output_base: str = None,
    metrics: list[str] | tuple[str, ...] | None = None,
    num_workers: int = 1,
    device: str | None = None,
) -> None:
    """Process CSV recordings with up to num_workers CPU intersection threads.

    The default 1 keeps serial queries. This option does not change PyTorch's
    thread settings. device=None selects CUDA when available, otherwise CPU;
    device='cpu' explicitly keeps all PyTorch calculations on the CPU.
    """
    num_workers = validate_num_workers(num_workers)

    if records_dir is None:
        records_dir = Path(input("[Enter records directory]"))
    else:
        records_dir = Path(records_dir)
    if mesh_path is None:
        mesh_path = str(input("[Enter mesh obj/fbx path]"))
    else:
        mesh_path = Path(mesh_path)
    

    if not records_dir.exists():
        raise FileNotFoundError(f"Record directory not found: {records_dir}")

    record_files = sorted(f for f in records_dir.iterdir() if f.suffix.lower() == ".csv")
    if not record_files:
        print(f"[em3disovist] No CSV files found in {records_dir}")
        return

    device = str(resolve_device(device))
    if device == "cpu":
        print("[em3disovist] Using CPU for PyTorch computations.")

    mesh = sample_mesh.mesh_sampling(mesh_path, sampling_density)

    if output_base == None:
        output_base = Path("./outputs/em3di_metrics/trials/")
    else:
        output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    for file_path in record_files:
        print(file_path.name)
        print("==== ID ====")

        results, meta = _process_record_file(
            file_path, mesh, chunk_size, hfov, vfov, device, temp_dir = temp_dir, metrics = metrics, num_workers = num_workers,
        )
        if results is None:
            continue

        print(meta["pid"], meta["bldg"], meta["task"], "\n")
        result_path = output_base / f"em3dimets_{file_path.name}"
        results.to_csv(result_path, index=False)
        print(f"[em3disovist] Combined results saved to {result_path}")
        print(f"[em3disovist] Finished processing {file_path.name}")


if __name__ == "__main__":
    run_batch_job()
