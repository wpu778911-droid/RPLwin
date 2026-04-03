import argparse
import collections
import os
import subprocess
import sys
import threading
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from tensorboard.backend.event_processing import event_accumulator

from gops.utils.tensorboard_setup import DEFAULT_TB_PORT, save_tb_to_csv, start_tensorboard, tb_tags


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
        description="Launch a GOPS training job and monitor it with TensorBoard plus terminal summaries."
    )
    parser.add_argument(
        "--result-root",
        type=str,
        default="results",
        help="Root directory that contains training result folders.",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default=None,
        help="Explicit TensorBoard log directory. If omitted, the script auto-detects the newest run folder.",
    )
    parser.add_argument(
        "--tensorboard-port",
        type=int,
        default=DEFAULT_TB_PORT,
        help="TensorBoard port.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=10.0,
        help="Seconds between metric refreshes.",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=300.0,
        help="Seconds to wait for a new log directory to appear.",
    )
    parser.add_argument(
        "--metric",
        action="append",
        default=None,
        help="TensorBoard scalar tag to print. Repeat this flag to add more tags.",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable auto-starting TensorBoard.",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Export TensorBoard scalars to CSV after training finishes.",
    )
    parser.add_argument(
        "--attach-only",
        action="store_true",
        help="Attach to an existing TensorBoard logdir without launching a training command.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Training command to run. Use `--` before the command.",
    )
    return parser.parse_args()


def strip_command_prefix(command: Sequence[str]) -> List[str]:
    if command and command[0] == "--":
        return list(command[1:])
    return list(command)


def format_value(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 1000:
        return f"{value:.2f}"
    if abs_value >= 1:
        return f"{value:.4f}"
    return f"{value:.6f}"


def clear_terminal() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def stream_pipe(pipe, prefix: str, sink) -> None:
    try:
        for line in iter(pipe.readline, ""):
            if not line:
                break
            sink.append(f"[{prefix}] {line.rstrip()}")
    finally:
        pipe.close()


def has_event_file(path: str) -> bool:
    try:
        for name in os.listdir(path):
            if name.startswith("events.out.tfevents"):
                return True
    except OSError:
        return False
    return False


def iter_candidate_logdirs(root_dir: str) -> Iterable[Tuple[float, str]]:
    if not os.path.isdir(root_dir):
        return []

    candidates: List[Tuple[float, str]] = []
    for dirpath, _, filenames in os.walk(root_dir):
        if any(name.startswith("events.out.tfevents") for name in filenames):
            try:
                modified = max(os.path.getmtime(os.path.join(dirpath, name)) for name in filenames)
            except OSError:
                continue
            candidates.append((modified, dirpath))
    candidates.sort(reverse=True)
    return candidates


def snapshot_run_dirs(root_dir: str) -> Dict[str, float]:
    snapshot: Dict[str, float] = {}
    if not os.path.isdir(root_dir):
        return snapshot

    for name in os.listdir(root_dir):
        path = os.path.join(root_dir, name)
        if os.path.isdir(path):
            try:
                snapshot[path] = os.path.getmtime(path)
            except OSError:
                continue
    return snapshot


def discover_logdir(root_dir: str, baseline: Dict[str, float], timeout: float) -> Optional[str]:
    deadline = time.time() + timeout
    newest_existing = next((path for _, path in iter_candidate_logdirs(root_dir)), None)
    if newest_existing is not None:
        return newest_existing

    while time.time() < deadline:
        current = snapshot_run_dirs(root_dir)
        new_dirs = [path for path in current if path not in baseline]
        for path in sorted(new_dirs, key=current.get, reverse=True):
            if has_event_file(path):
                return path

        newest_existing = next((path for _, path in iter_candidate_logdirs(root_dir)), None)
        if newest_existing is not None:
            return newest_existing

        time.sleep(2.0)
    return None


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


def print_metric_table(logdir: str, scalars: Dict[str, Tuple[int, float]], metric_tags: Sequence[str]) -> None:
    print(f"\n[monitor] logdir: {logdir}")
    if not metric_tags:
        print("[monitor] no scalar data found yet")
        return

    width = max(len(tag) for tag in metric_tags)
    for tag in metric_tags:
        step, value = scalars[tag]
        print(f"[monitor] {tag:<{width}}  step={step:<8d} value={format_value(value)}")


def render_dashboard(
    command: Sequence[str],
    logdir: Optional[str],
    tensorboard_port: int,
    tb_started: bool,
    process: Optional[subprocess.Popen],
    scalars: Dict[str, Tuple[int, float]],
    metric_tags: Sequence[str],
    last_error: Optional[str],
    recent_logs: Sequence[str],
) -> None:
    clear_terminal()
    print("GOPS Training Monitor")
    print("=" * 72)
    print(f"command     : {' '.join(command) if command else '(attach only)'}")
    print(f"pid         : {process.pid if process is not None else '-'}")
    print(
        f"status      : "
        f"{'attached' if process is None else ('running' if process.poll() is None else f'exited({process.returncode})')}"
    )
    print(f"logdir      : {logdir or 'detecting...'}")
    print(f"tensorboard : {f'http://localhost:{tensorboard_port}' if tb_started else 'not started'}")
    print(f"updated     : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 72)

    if last_error:
        print(f"note        : {last_error}")
        print("-" * 72)

    if not scalars:
        print("No scalar data yet. Waiting for TensorBoard event files...")
    elif not metric_tags:
        print("No requested metric tags found in current scalar set.")
    else:
        width = max(len(tag) for tag in metric_tags)
        print(f"{'metric':<{width}}  {'step':>10}  {'value':>16}")
        print("-" * (width + 30))
        for tag in metric_tags:
            step, value = scalars[tag]
            print(f"{tag:<{width}}  {step:>10d}  {format_value(value):>16}")
        print("-" * (width + 30))

    print("Recent logs")
    print("-" * 72)
    if recent_logs:
        for line in recent_logs:
            print(line)
    else:
        print("(no logs yet)")
    print("-" * 72)
    print("Ctrl+C to stop monitoring and terminate training.")


def main() -> int:
    args = parse_args()
    command = strip_command_prefix(args.command)
    if args.attach_only and not args.logdir:
        print("--attach-only requires --logdir", file=sys.stderr)
        return 2
    if (not args.attach_only) and (not command):
        print("No training command was provided. Example:", file=sys.stderr)
        print(
            "python example_run/monitor_training.py -- python example_train/ppo/ppo_mlp_cartpoleconti_onserial.py",
            file=sys.stderr,
        )
        return 2

    result_root = os.path.abspath(args.result_root)
    baseline = snapshot_run_dirs(result_root)

    recent_logs = collections.deque(maxlen=12)
    process: Optional[subprocess.Popen] = None
    stdout_thread = None
    stderr_thread = None

    if not args.attach_only:
        print(f"[monitor] launching: {' '.join(command)}")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        stdout_thread = threading.Thread(target=stream_pipe, args=(process.stdout, "train", recent_logs), daemon=True)
        stderr_thread = threading.Thread(target=stream_pipe, args=(process.stderr, "train-err", recent_logs), daemon=True)
        stdout_thread.start()
        stderr_thread.start()

    logdir = os.path.abspath(args.logdir) if args.logdir else None
    tb_started = False
    last_snapshot: Dict[str, Tuple[int, float]] = {}
    last_print_time = 0.0
    last_error: Optional[str] = None

    try:
        while True:
            if logdir is None:
                logdir = discover_logdir(result_root, baseline, timeout=min(args.startup_timeout, 10.0))
                if logdir is not None:
                    last_error = None

            if logdir is not None and (not tb_started) and (not args.no_tensorboard):
                start_tensorboard(logdir, port=args.tensorboard_port, autoopen=False)
                tb_started = True
                last_error = None

            if logdir is not None and (time.time() - last_print_time >= args.poll_interval):
                try:
                    scalars = read_latest_scalars(logdir)
                    metric_tags = choose_metric_tags(scalars, args.metric or [])
                    last_snapshot = scalars
                    last_error = None
                    render_dashboard(
                        command,
                        logdir,
                        args.tensorboard_port,
                        tb_started,
                        process,
                        scalars,
                        metric_tags,
                        last_error,
                        list(recent_logs),
                    )
                    last_print_time = time.time()
                except Exception as exc:
                    last_error = f"waiting for scalar data: {exc}"
                    render_dashboard(
                        command,
                        logdir,
                        args.tensorboard_port,
                        tb_started,
                        process,
                        last_snapshot,
                        choose_metric_tags(last_snapshot, args.metric or []),
                        last_error,
                        list(recent_logs),
                    )
                    last_print_time = time.time()
            elif time.time() - last_print_time >= args.poll_interval:
                render_dashboard(
                    command,
                    logdir,
                    args.tensorboard_port,
                    tb_started,
                    process,
                    last_snapshot,
                    choose_metric_tags(last_snapshot, args.metric or []),
                    last_error,
                    list(recent_logs),
                )
                last_print_time = time.time()

            retcode = process.poll() if process is not None else None
            if process is not None and retcode is not None:
                break
            time.sleep(1.0)

        if process is not None:
            stdout_thread.join(timeout=2.0)
            stderr_thread.join(timeout=2.0)

        if logdir is not None:
            try:
                scalars = read_latest_scalars(logdir)
                metric_tags = choose_metric_tags(scalars, args.metric or [])
                render_dashboard(
                    command,
                    logdir,
                    args.tensorboard_port,
                    tb_started,
                    process,
                    scalars,
                    metric_tags,
                    None,
                    list(recent_logs),
                )
            except Exception as exc:
                print(f"[monitor] failed to read final scalar snapshot: {exc}")

            if args.export_csv:
                save_tb_to_csv(logdir)
                print(f"[monitor] exported csv files to: {os.path.join(logdir, 'data')}")

        if process is None:
            return 0
        if retcode == 0:
            print("[monitor] training finished successfully")
        else:
            print(f"[monitor] training failed with exit code {retcode}")
        return retcode
    except KeyboardInterrupt:
        if process is not None:
            print("[monitor] stopping training process")
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
