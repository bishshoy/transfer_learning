import argparse
import time
import os
import sys

from job_stat import check_stat, fetch_job_ids


def process(args):
    total_running = 0

    for job_id in fetch_job_ids():
        running, _ = check_stat(job_id, args.debug)

        if running:
            total_running += 1

    if total_running < args.queue_max:
        print('\ntotal jobs running:', total_running)

        to_launch = max(args.queue_max - total_running, 0)
        print('launching', to_launch, 'jobs')

        cmd = 'python job_launcher.py -n' + str(to_launch)
        if args.debug:
            cmd += ' --debug'
        print(cmd, '\n\n')

        os.system(cmd)
        print()

    else:
        sys.stdout.write('.')
        sys.stdout.flush()


def main(args):
    print('check interval', str(args.check_interval))
    print('queue max', str(args.queue_max))

    while True:
        process(args)
        time.sleep(args.check_interval)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-interval', type=int, default=60)
    parser.add_argument('--queue-max', type=int, default=10)

    parser.add_argument('--debug', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    print('daemon has started')
    main(parse())
