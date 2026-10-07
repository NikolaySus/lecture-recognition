"""Detached resumable worker, serializing GPU work across one service store."""
import argparse
import os
import traceback

import psutil
from filelock import FileLock, Timeout

from .jobs import JobStore
from .pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--store', required=True)
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    store = JobStore(args.store)
    try:
        with FileLock(str(store.folder(args.job) / 'job.lock'), timeout=0):
            store.state(args.job, status='queued', stage='waiting_gpu', worker_pid=os.getpid(),
                        worker_birth=psutil.Process().create_time())
            with FileLock(str(store.root / 'gpu.lock')):
                try:
                    run_pipeline(store, args.job)
                except Exception as exc:
                    store.state(args.job, status='failed', error=str(exc))
                    traceback.print_exc()
    except Timeout:
        return


if __name__ == '__main__':
    main()
