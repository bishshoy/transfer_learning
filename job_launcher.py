import numpy as np
from datetime import datetime
import yaml
import argparse
import subprocess
import time

from parsers import parse as script_parser


def create_arg_sets(config):
    arg_sets = []

    def hparam_lr(lr):
        script_args = script_parser()

        # Fixed params
        # script_args.print_model
        # script_args.epochs
        # script_args.continuous
        # script_args.num_workers
        # script_args.mom
        # script_args.wd
        # script_args.validate
        # script_args.upscale_images
        # script_args.check_hyp
        # script_args.root

        # Must be supplied
        script_args.model = config['model']
        script_args.dataset = config['dataset']

        # Set or use default
        script_args.batch_size = config.get('batch_size', script_args.batch_size)

        # Boolean params
        script_args.pretrained = config.get('pretrained', False)
        script_args.freeze_conv = config.get('freeze_conv', False)
        script_args.replace_fc = config.get('replace_fc', False)

        # Hyperparameters to be searched
        script_args.lr = lr

        return script_args

    for lr in np.float64(config['lr']):
        arg_sets.append(hparam_lr(lr))

    return arg_sets


def create_launch_scripts(script_args, launch_args):
    time.sleep(1)
    script_id = datetime.now().strftime('%d-%b-%H-%M-%S')

    lines = []

    if launch_args.hpc:
        lines += ['export SCRATCH=/scratch/ee/phd/eez168482 \n']
        lines += ['export PATH=$SCRATCH/miniconda3/bin:$PATH \n']

    lines += ['cd $HOME/transfer_learning \n']

    lines += ['unbuffer python experiment.py \\']
    lines += ['--model ' + script_args.model + ' \\']
    lines += ['--dataset ' + script_args.dataset + ' \\']
    lines += ['--batch-size ' + str(script_args.batch_size) + ' \\']

    if script_args.pretrained:
        lines += ['--pretrained \\']

    if script_args.freeze_conv:
        lines += ['--freeze-conv \\']

    if script_args.replace_fc:
        lines += ['--replace-fc \\']

    lines += ['--lr ' + str(script_args.lr) + ' \\']
    lines += ['| tee logs/'+str(script_id)+'.txt \\']
    lines += ['\n\n']

    script = '\n'.join(lines)
    file = open('.temp_launch.sh', 'w+')
    file.writelines(script)
    file.close()

    # print(script)
    return script, script_id


def create_job(launch_script, script_id):
    cmd = 'qsub -P ee -o /dev/null -e /dev/null -lselect=1:ncpus=1:ngpus=1:centos=skylake'
    cmd += ' -N '+str(script_id)
    cmd += ' '+launch_script

    # return 0
    job_id = subprocess.check_output(cmd, shell=True).strip().decode('utf-8')
    return job_id


def store_job_id(job_id, script_id):
    file = open('.ids', 'a+')
    file.writelines(str(job_id)+','+str(script_id)+'\n')
    file.close()


def launch_local():
    subprocess.Popen(['sh', '.temp_launch.sh'])


def main(launch_args):
    file = open('config.yaml', 'r')
    config = yaml.safe_load(file)

    jobs = {}
    for conf in config.keys():
        jobs[conf] = create_arg_sets(config[conf])

    if launch_args.hpc:
        print('JOB_ID\t\t\t SCRIPT_ID')
        print('======\t\t\t =========')

    scripts = []
    for job in jobs.keys():
        for script_args in jobs[job]:
            script, script_id = create_launch_scripts(script_args, launch_args)

            if launch_args.hpc:
                job_id = create_job('.temp_launch.sh', script_id)
                print(job_id, '\t\t', script_id)
                store_job_id(job_id, script_id)

            if launch_args.local:
                scripts += [script]

    if launch_args.local:
        final_script = '\n\n'.join(scripts)
        file = open('.temp_launch.sh', 'w+')
        file.writelines(final_script)
        file.close()

        launch_local()


def launch_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hpc', action='store_true')
    parser.add_argument('--local', action='store_true')

    launch_args = parser.parse_args()

    if launch_args.hpc and launch_args.local:
        raise ValueError('--hpc and --local are mutually exclusive')

    if not launch_args.local:
        launch_args.hpc = True

    if launch_args.hpc:
        print('Creating jobs for: HPC')
    if launch_args.local:
        print('Creating jobs for: local machine')

    return launch_args


if __name__ == '__main__':
    launch_args = launch_parser()
    main(launch_args)
