import numpy as np
from datetime import datetime
import yaml
import time
import subprocess

from parsers import *


def create_arg_sets(config):
    arg_sets = []

    def hparam_lr(lr):
        args = parse()

        # Fixed params
        # args.print_model
        # args.epochs
        # args.continuous
        # args.num_workers
        # args.mom
        # args.wd
        # args.validate
        # args.upscale_images
        # args.check_hyp
        # args.root

        # Must be supplied
        args.model = config['model']
        args.dataset = config['dataset']

        # Set or use default
        args.batch_size = config.get('batch_size', args.batch_size)

        # Boolean params
        args.pretrained = config.get('pretrained', False)
        args.freeze_conv = config.get('freeze_conv', False)
        args.replace_fc = config.get('replace_fc', False)

        # Hyperparameters to be searched
        args.lr = lr

        return args

    for lr in np.float64(config['lr']):
        arg_sets.append(hparam_lr(lr))

    return arg_sets


def create_launch_scripts(args):
    time.sleep(1)
    script_id = datetime.now().strftime('%d-%b-%H-%M-%S')

    lines = []
    lines += ['export SCRATCH=/scratch/ee/phd/eez168482 \n']
    lines += ['export PATH=$SCRATCH/miniconda3/bin:$PATH \n']
    lines += ['cd $HOME/transfer_learning \n']

    lines += ['unbuffer python experiment.py \\']
    lines += ['--model ' + args.model + ' \\']
    lines += ['--dataset ' + args.dataset + ' \\']
    lines += ['--batch-size ' + str(args.batch_size) + ' \\']

    if args.pretrained:
        lines += ['--pretrained \\']

    if args.freeze_conv:
        lines += ['--freeze-conv \\']

    if args.replace_fc:
        lines += ['--replace-fc \\']

    lines += ['--lr ' + str(args.lr) + ' \\']
    lines += ['| tee logs/'+str(script_id)+'.txt \\']
    lines += ['\n\n']

    file = open('.temp_launch.sh', 'w+')
    file.writelines('\n'.join(lines))
    file.close()

    # print('\n'.join(lines))
    return script_id


def create_job(launch_script, script_id):
    cmd = 'qsub -P ee -o /dev/null -e /dev/null -lselect=1:ncpus=1:ngpus=1'
    cmd += ' -N '+str(script_id)
    cmd += ' '+launch_script

    # return 0
    job_id = subprocess.check_output(cmd, shell=True).strip().decode('utf-8')
    return job_id


def store_job_id(job_id, script_id):
    file = open('.ids', 'a+')
    file.writelines(str(job_id)+','+str(script_id)+'\n')
    file.close()


def main():
    file = open('config.yaml', 'r')
    config = yaml.safe_load(file)

    jobs = {}
    for conf in config.keys():
        jobs[conf] = create_arg_sets(config[conf])

    print('JOB_ID\t\t\t SCRIPT_ID')
    print('======\t\t\t =========')
    for job in jobs.keys():
        for args in jobs[job]:
            script_id = create_launch_scripts(args)
            job_id = create_job('.temp_launch.sh', script_id)
            print(job_id, '\t\t', script_id)
            store_job_id(job_id, script_id)


if __name__ == '__main__':
    main()
