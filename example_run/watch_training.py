import argparse
import os
import subprocess
import sys
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from tensorboard.backend.event_processing import event_accumulator

from gops.utils.tensorboard_setup import DEFAULT_TB_PORT, start_tensorboard, tb_tags


DEFAULT_METRICS = [
    tb_tags["TAR of RL iteration"],
    tb_tags["TAR of total time"],
    tb_tags["loss_actor"],
    tb_tags["loss_critic"],
    tb_tags["alg_time"],
    tb_tags["sampler_time"],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch GOPS result folders and automatically attach a terminal monitor to active training runs."
    )
    parser.add_argument("--result-root", type=str, default="results", help="Root directory to watch.")
    parser.add_argument("--tensorboard-port", type=int, default=DEFAULT_TB_PORT, help="TensorBoard port.")
    parser.add_argument("--poll-interval", type=float, default=3.0, help="Polling interval in seconds.")
    parser.add_argument(
        "--active-window",
        type=float,
        default=20.0,
        help="A run is considered active if an event file changed within this many seconds.",
    )
    parser.add_argument(
        "--metric",
        action="append",
        default=None,
        help="TensorBoard scalar tag to print. Repeat to add more.",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable auto-starting TensorBoard when an active run is detected.",
    )
    parser.add_argument(
        "--no-popup",
        action="store_true",
        help="Disable auto-opening a dedicated monitor window for detected training runs.",
    )
    return parser.parse_args()


def clear_terminal() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def format_value(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 1000:
        return f"{value:.2f}"
    if abs_value >= 1:
        return f"{value:.4f}"
    return f"{value:.6f}"


def iter_event_dirs(root_dir: str) -> Iterable[Tuple[str, float, str]]:
    if not os.path.isdir(root_dir):
        return []

    output: List[Tuple[str, float, str]] = []
    for dirpath, _, filenames in os.walk(root_dir):
        event_files = [name for name in filenames if name.startswith("events.out.tfevents")]
        if not event_files:
            continue
        latest_file = max(
            event_files,
            key=lambda name: os.path.getmtime(os.path.join(dirpath, name)),
        )
        latest_path = os.path.join(dirpath, latest_file)
        output.append((dirpath, os.path.getmtime(latest_path), latest_path))
    output.sort(key=lambda item: item[1], reverse=True)
    return output


def find_active_logdir(root_dir: str, active_window: float) -> Tuple[Optional[str], List[Tuple[str, float]]]:
    now = time.time()
    candidates: List[Tuple[str, float]] = []
    active_logdir: Optional[str] = None

    for logdir, last_modified, _ in iter_event_dirs(root_dir):
        age = now - last_modified
        candidates.append((logdir, age))
        if active_logdir is None and age <= active_window:
            active_logdir = logdir

    return active_logdir, candidates[:5]


def read_latest_scalars(logdir: str) -> Dict[str, Tuple[int, float]]:
    acc = event_accumulator.EventAccumulator(logdir, size_guidance={"scalars": 0})
    acc.Reload()
    output: Dict[str, Tuple[int, float]] = {}
    for tag in acc.scalars.Keys():
        items = acc.scalars.Items(tag)
        if items:
            last = items[-1]
            output[tag] = (int(last.step), float(last.value))
    return output


def choose_metric_tags(all_scalars: Dict[str, Tuple[int, float]], requested: Sequence[str]) -> List[str]:
    if requested:
        return [tag for tag in requested if tag in all_scalars]

    chosen = [tag for tag in DEFAULT_METRICS if tag in all_scalars]
    if chosen:
        return chosen
    return sorted(all_scalars.keys())[:6]


def render_dashboard(
    result_root: str,
    active_logdir: Optional[str],
    candidates: Sequence[Tuple[str, float]],
    scalars: Dict[str, Tuple[int, float]],
    metric_tags: Sequence[str],
    tensorboard_port: int,
    tb_started: bool,
    last_error: Optional[str],
) -> None:
    clear_terminal()
    print("GOPS Auto Training Watcher")
    print("=" * 72)
    print(f"watch root   : {os.path.abspath(result_root)}")
    print(f"active run   : {active_logdir or 'none'}")
    print(f"tensorboard  : {f'http://localhost:{tensorboard_port}' if tb_started else 'not started'}")
    print(f"updated      : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 72)

    if last_error:
        print(f"note         : {last_error}")
        print("-" * 72)

    if active_logdir and scalars and metric_tags:
        width = max(len(tag) for tag in metric_tags)
        print(f"{'metric':<{width}}  {'step':>10}  {'value':>16}")
        print("-" * (width + 30))
        for tag in metric_tags:
            step, value = scalars[tag]
            print(f"{tag:<{width}}  {step:>10d}  {format_value(value):>16}")
        print("-" * (width + 30))
    elif active_logdir:
        print("Active run detected, waiting for scalar data...")
        print("-" * 72)
    else:
        print("No active training detected.")
        print("-" * 72)

    print("Recent candidate runs")
    print("-" * 72)
    if candidates:
        for logdir, age in candidates:
            status = "ACTIVE" if active_logdir == logdir else ""
            print(f"{age:8.1f}s  {logdir} {status}")
    else:
        print("(no TensorBoard event directories found)")
    print("-" * 72)
    print("Keep this watcher running. Start any GOPS training in another terminal.")


def open_monitor_window(logdir: str, result_root: str, tensorboard_port: int) -> None:
    script_path = os.path.join(os.path.dirname(__file__), "monitor_training.py")
    command = (
        f'cd /d "{os.getcwd()}" && '
        f'"{sys.executable}" "{script_path}" --attach-only --logdir "{logdir}" '
        f'--result-root "{result_root}" --tensorboard-port {tensorboard_port} --no-tensorboard'
    )
    subprocess.Popen(
        [
            "cmd",
            "/c",
            "start",
            "GOPS Training Monitor",
            "cmd",
            "/k",
            command,
        ]
    )


def main() -> int:
    args = parse_args()
    result_root = args.result_root
    current_logdir: Optional[str] = None
    tb_started_for: Optional[str] = None
    last_scalars: Dict[str, Tuple[int, float]] = {}
    last_error: Optional[str] = None
    launched_monitors = set()

    try:
        while True:
            active_logdir, candidates = find_active_logdir(result_root, args.active_window)

            if active_logdir != current_logdir:
                current_logdir = active_logdir
                last_scalars = {}
                last_error = None

            if current_logdir and (current_logdir not in launched_monitors) and (not args.no_popup) and os.name == "nt":
                open_monitor_window(current_logdir, result_root, args.tensorboard_port)
                launched_monitors.add(current_logdir)

            if current_logdir and (tb_started_for != current_logdir) and (not args.no_tensorboard):
                start_tensorboard(current_logdir, port=args.tensorboard_port, autoopen=False)
                tb_started_for = current_logdir

            if current_logdir:
                try:
                    last_scalars = read_latest_scalars(current_logdir)
                    last_error = None
                except Exception as exc:
                    last_error = f"waiting for scalar data: {exc}"
            else:
                last_error = None

            render_dashboard(
                result_root=result_root,
                active_logdir=current_logdir,
                candidates=candidates,
                scalars=last_scalars,
                metric_tags=choose_metric_tags(last_scalars, args.metric or []),
                tensorboard_port=args.tensorboard_port,
                tb_started=tb_started_for == current_logdir and current_logdir is not None,
                last_error=last_error,
            )
            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print("\nWatcher stopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
