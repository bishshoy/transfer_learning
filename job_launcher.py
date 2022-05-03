from datetime import datetime
import yaml
import argparse
import subprocess
import time

from parsers import parse as script_parser


def create_script_args(config, launch_args):
    script_args = script_parser()

    # Fixed params
    # script_args.print_model
    # script_args.epochs
    # script_args.continuous
    # script_args.mom
    # script_args.wd
    # script_args.validate
    # script_args.root

    # For HPC, num_workers=1
    script_args.num_workers = 1

    # Must be supplied
    script_args.model = config['model']
    script_args.dataset = config['dataset']

    # Set or use default
    script_args.batch_size = config.get('batch_size', script_args.batch_size)

    # Boolean params
    script_args.pretrained = config.get('pretrained', False)
    script_args.freeze_conv = config.get('freeze_conv', False)

    # Hyperparameters to be searched
    script_args.lr = config['lr']

    return script_args


def create_launch_scripts(script_args, launch_args):
    time.sleep(1)
    script_id = datetime.now().strftime('%d-%b-%H-%M-%S')

    lines = []

    lines += ['export SCRATCH=/scratch/ee/phd/eez168482 \n']
    lines += ['export PATH=$SCRATCH/miniconda3/bin:$PATH \n']

    lines += ['cd $HOME/transfer_learning \n']

    lines += ['unbuffer python experiment.py \\']
    lines += ['--num-workers ' + str(script_args.num_workers) + ' \\']
    lines += ['--model ' + script_args.model + ' \\']
    lines += ['--dataset ' + script_args.dataset + ' \\']
    lines += ['--batch-size ' + str(script_args.batch_size) + ' \\']

    if script_args.pretrained:
        lines += ['--pretrained \\']

    if script_args.freeze_conv:
        lines += ['--freeze-conv \\']

    lines += ['--lr ' + str(script_args.lr) + ' \\']
    lines += ['| tee logs/' + str(script_id) + '.txt \\']
    lines += ['\n\n']

    script = '\n'.join(lines)

    with open('.temp_launch.sh', 'w+') as file:
        file.writelines(script)

    return script, script_id


def submit_job(script_id, launch_args):
    cmd = 'qsub -P ee -o /dev/null -e /dev/null -lselect=1:ncpus=1:ngpus=1:centos=skylake'
    cmd += ' -N ' + str(script_id)
    cmd += ' ' + '.temp_launch.sh'

    if launch_args.debug:
        print('debug:', cmd)
        return int(1e6)

    job_id = subprocess.check_output(cmd, shell=True).strip().decode('utf-8')
    return job_id


def store_job_id(job_id, script_id, script_args):
    lines = [
        str(job_id),
        str(script_id),
        script_args.model,
        script_args.dataset,
        'pretrained' if script_args.pretrained else '',
        'freeze_conv' if script_args.freeze_conv else '',
        'lr=' + str(script_args.lr),
    ]
    lines = [','.join(lines)]

    with open('.ids', 'r') as file:
        existing_ids = file.read().strip().split('\n')

    lines = existing_ids + lines

    with open('.ids', 'w+') as file:
        file.writelines('\n'.join(lines))


def launch_jobs(launch_args):
    with open('config.yaml', 'r') as stream:
        config = yaml.safe_load(stream)

    script_args = {}
    launched_jobs = []
    for job, conf in config.items():
        job_number = int(job.split('job')[-1])

        if conf['launched']:
            launched_jobs.append(job_number)
        else:
            script_args[job_number] = create_script_args(conf, launch_args)

    last_launched = max(launched_jobs, default=0)
    next_launch = last_launched + 1

    print('JOB_ID\t\t\t SCRIPT_ID')
    print('======\t\t\t =========')

    to_launch = launch_args.n

    for j in script_args.keys():
        to_launch -= 1

        if to_launch < 0:
            break

        else:
            _, script_id = create_launch_scripts(script_args[next_launch], launch_args)
            job_id = submit_job(script_id, launch_args)
            config['job' + str(next_launch)]['launched'] = True

            print(job_id, '\t\t', script_id)
            store_job_id(job_id, script_id, script_args[j])

            next_launch += 1

    with open('config.yaml', 'w+') as stream:
        yaml.safe_dump(config, stream, sort_keys=False)


def launch_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true')
    parser.add_argument('-n', type=int, default=1)

    launch_args = parser.parse_args()

    if launch_args.debug:
        print('DEBUG mode::')

    if launch_args.n < 0:
        launch_args.n = 0

    print('creating', launch_args.n, 'jobs')

    return launch_args


if __name__ == '__main__':
    launch_args = launch_parser()
    launch_jobs(launch_args)
