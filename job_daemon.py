import argparse
import time
import os

from job_stat import check_stat, fetch_job_ids


def process(args):
    total_running = 0

    for job_id in fetch_job_ids():
        running, _ = check_stat(job_id)

        if running:
            total_running += 1

    print('\ntotal jobs running:', total_running)

    if total_running == args.queue_max:
        print('queue full\n\n')

    else:
        to_launch = max(args.queue_max - total_running, 0)
        print('launching', to_launch, 'jobs')

        cmd = 'python job_launcher.py -n' + str(to_launch)
        if args.debug:
            cmd += ' --debug'
        print(cmd, '\n\n')

        os.system(cmd)
        print()


def main(args):
    print('check interval', str(args.check_interval))
    print('queue max', str(args.queue_max))

    while True:
        process(args)

        for i in range(args.check_interval-1, -1, -1):
            print('\rperforming next check in', i, 'seconds', end='    ')
            time.sleep(1)

        print()


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-interval', type=int, default=60)
    parser.add_argument('--queue-max', type=int, default=10)

    parser.add_argument('--debug', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    print('daemon has started')
    main(parse())
