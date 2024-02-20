import os
import re
import subprocess
import pathlib
import multiprocessing
import tqdm

if __name__ == '__main__':
    lst = {
        'SplitResult-gender': {
            'SL-1': (1, '090055', 50),
            'SL-2': (2, '090056', 46),
            'SL-3': (3, '090058', 42),
            'LL-1': (1, '030549', 27),
            'LL-2': (2, '025912', 22),
            'LL-3': (3, '025916', 21),
        }
    }


    aws_base_dir = '/home/ubuntu/SwarmSense/NIHCHEST'
    saved_base_dir = './saved'
    if os.path.exists(saved_base_dir):
        os.system(f'rm -rf {saved_base_dir}')

    cmd_to_run = []
    for subdir, v in lst.items():
        for name, (num, timestamp, epoch) in v.items():
            cmd = f'fd -IH ".*{timestamp}.*" {subdir}/'
            out = subprocess.check_output(cmd, shell=True)
            out = out.decode('utf-8').strip().split('\n')
            out = [x for x in out if not x.endswith('.pkl')]
            assert len(out) == 1
            out = out[0].split('/')[-1]
            dest_dir = os.path.join(saved_base_dir, subdir, name) + '/'
            src_dir = f'"aws:{aws_base_dir}/{subdir}/{num}/saved_models/{out}-{epoch}"'
            pathlib.Path(dest_dir).mkdir(parents=True, exist_ok=True)
            cmd = ['/usr/bin/scp', '-r', src_dir, dest_dir]
            cmd = ' '.join(cmd)
            cmd_to_run.append(cmd)

    with multiprocessing.Pool(processes=8) as pool:
        pool.map(os.system, cmd_to_run)
