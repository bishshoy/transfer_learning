import yaml


def update_config(new_configs, overwrite):
    if overwrite:
        print('OVERWRITE MODE')
        print('==============')

        ans = input('Are you sure you want to continue? (y/n): ')

        if ans == 'n':
            print('Exiting...')
            return

        curr_config = {}
        last_job = 0

    else:
        print('APPEND MODE')
        print('============')

        with open('config.yaml', 'r') as stream:
            curr_config = yaml.safe_load(stream)

        job_numbers = list(curr_config.keys())

        for i in range(len(job_numbers)):
            job_numbers[i] = int(job_numbers[i].split('job')[-1])

        last_job = max(job_numbers)
        print('last job number detected =', last_job)

    for conf in new_configs:
        last_job += 1
        curr_config['job'+str(last_job)] = conf

    with open('config.yaml', 'w+') as stream:
        yaml.safe_dump(curr_config, stream, sort_keys=False)


if __name__ == '__main__':
    overwrite = True

    models = [
        'resnet18',
    ]
    datasets = [
        'cifar10',
        'cifar100',
    ]
    lrs = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]

    new_configs = []

    # Your code for creating new configs
    # Append to new_configs
    # and pass the array to update_configs()
    for dataset in datasets:
        for model in models:
            for mode in [0, 1, 2]:
                for lr in lrs:

                    if mode == 0:
                        pretrained = False
                        freeze_conv = False
                    if mode == 1:
                        pretrained = True
                        freeze_conv = True
                    if mode == 2:
                        pretrained = True
                        freeze_conv = False

                    conf = {
                        'model': model,
                        'dataset': dataset,
                        'lr': lr,
                        'pretrained': pretrained,
                        'freeze_conv': freeze_conv,
                        'launched': False,
                    }

                    new_configs.append(conf)

    update_config(new_configs, overwrite)
    print('created', len(new_configs), 'job configurations')
