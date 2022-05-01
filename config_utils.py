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
        curr_config['job' + str(last_job)] = conf

    with open('config.yaml', 'w+') as stream:
        yaml.safe_dump(curr_config, stream, sort_keys=False)