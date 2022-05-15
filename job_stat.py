import glob
from datetime import datetime
import random
from collections import OrderedDict
import argparse
import subprocess
import os


def fetch_job_ids():
    job_ids = []

    with open('.ids', 'r') as file:
        lines = file.read().strip().split('\n')

        for l in lines:
            if l != '':
                job_ids.append(l.strip().split(',')[0])

    return job_ids


def check_stat(job_id, debug_mode=False):
    if debug_mode:
        return random.randint(0, 1), ''

    cmd = 'qstat ' + job_id

    try:
        message = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).strip().decode('utf-8')
        return True, message
    except subprocess.CalledProcessError as e:
        return False, e.output.strip().decode('utf-8')


def watch_stat():
    outputs = []

    for job_id in fetch_job_ids():
        running, msg = check_stat(job_id)

        if running:
            out = msg
        else:
            out = 'Job id\n----------------\n'
            out += job_id + '\t\t'
            out += msg

        outputs.append(out)
        outputs.append('\n')

    output = '\n'.join(outputs)
    print(output)


def view_logs(dir, nice):
    files = glob.glob(dir + '/*')
    files = sorted([datetime.strptime(x.strip().split('/')[-1].split('.')[0], '%d-%b-%H-%M-%S') for x in files])
    files = [dir + '/' + x.strftime('%d-%b-%H-%M-%S') + '.txt' for x in files]

    print('found', len(files), 'logs')

    all_lines = []
    for f in files:
        all_lines.append('filename: ' + f)
        with open(f, 'rb') as stream:
            lines = stream.read().decode()
        all_lines.append(lines)
    all_lines = '\n\n'.join(all_lines)

    if not nice:
        print(all_lines)
    else:
        all_lines = all_lines.replace('\r', '\n')
        all_lines = all_lines.split('\n')

        models, datasets = [], []
        table = OrderedDict()

        for l in all_lines:
            if '###' in l:
                iden = l.strip().split(', ')
                model = iden[0][len('### model: ') :]
                dataset = iden[1][len('dataset: ') :]
                mode = iden[3][len('mode: ') :]
                acc = float(iden[4][len('best_acc: ') :])

                if model not in models:
                    models.append(model)

                if dataset not in datasets:
                    datasets.append(dataset)

                key = model + '_' + dataset + '_' + mode
                table[key] = max(acc, table.get(key, 0))

        for model in models:
            for dataset in datasets:
                print('|', end=' ')
                print(model, end=' ')
                print('|', end=' ')
                print(dataset, end=' ')
                print('|', end=' ')

                for i in range(3):
                    try:
                        print('{:.2f}'.format(table[model + '_' + dataset + '_' + str(i)]), end=' ')
                    except:
                        print('', end=' ')
                    print('|', end=' ')
                print()
            print('|\t' * 6)


def stop_all():
    for job_id in fetch_job_ids():
        cmd = 'qdel ' + job_id
        print(cmd)

        try:
            os.system(cmd)
        except:
            pass

        print()


def clean():
    with open('.ids', 'r') as file:
        lines = file.read().strip().split('\n')

    print()
    print('All')
    print('===')
    print('\n'.join(lines))
    print()

    keep = []
    outdated = []

    for l in lines:
        job_id = l.strip().split(',')[0]

        running, msg = check_stat(job_id)

        if running:
            keep.append(l)
        else:
            outdated.append(l)

    print('Outdated')
    print('========')
    print('\n'.join(outdated))
    print()

    print('Keep')
    print('====')
    print('\n'.join(keep))
    print()

    with open('.ids', 'w+') as file:
        file.writelines('\n'.join(keep).strip())


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--logs', action='store_true')
    parser.add_argument('--nice', action='store_true')
    parser.add_argument('--dir', type=str, default='logs')
    parser.add_argument('--clean', action='store_true')
    parser.add_argument('--stop-all', action='store_true')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()

    if args.watch:
        watch_stat()
    elif args.logs:
        view_logs(args.dir, args.nice)
    elif args.clean:
        clean()
    elif args.stop_all:
        stop_all()
