import argparse
import subprocess
from glob import glob
import os


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--logs', action='store_true')
    parser.add_argument('--clean', action='store_true')
    parser.add_argument('--stop-all', action='store_true')

    args = parser.parse_args()
    return args


def watch_qstat():
    lines = open('.ids', 'r').read().strip().split('\n')

    outputs = []
    for l in lines:
        job_id = l.strip().split(',')[0]
        cmd = 'qstat ' + job_id

        try:
            out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).strip().decode('utf-8')
        except subprocess.CalledProcessError as e:
            out = 'Job id\n----------------\n'
            out += job_id + '\t\t'
            out += e.output.strip().decode('utf-8')

        outputs.append(out)
        outputs.append('\n')

    output = '\n'.join(outputs)
    print(output)


def view_logs():
    files = sorted(glob('logs/*'))

    for f in files:
        cmd = 'cat ' + f
        os.system(cmd)
    
    print()


def stop_all():
    lines = open('.ids', 'r').read().strip().split('\n')

    for l in lines:
        job_id = l.strip().split(',')[0]

        cmd = 'qdel '+job_id
        print(cmd)

        try:
            os.system(cmd)
        except:
            pass

        print()


def clean():
    file = open('.ids', 'r').read()
    lines = file.strip().split('\n')

    print()
    print('All')
    print('===')
    print(file)
    print()

    keep = []
    outdated = []

    for l in lines:
        job_id = l.strip().split(',')[0]
        cmd = 'qstat ' + job_id

        try:
            subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
            keep.append(l)
        except subprocess.CalledProcessError:
            outdated.append(l)

    print('Outdated')
    print('========')
    print('\n'.join(outdated))
    print()

    print('Keep')
    print('====')
    print('\n'.join(keep))
    print()

    file = open('.ids', 'w+')
    file.writelines('\n'.join(keep)+'\n')
    file.close()


def main(args):

    if args.watch:
        watch_qstat()
    elif args.logs:
        view_logs()
    elif args.clean:
        clean()
    elif args.stop_all:
        stop_all()


if __name__ == '__main__':
    args = parse()
    main(args)
