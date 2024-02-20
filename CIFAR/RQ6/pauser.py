import os
import pickle
import random
import time
import datetime
import multiprocessing as mp
import docker


loss_probability=10
# loss_percentage=100
interval_length=120
sl_num=8
sn_addr='172.21.0.4'

RULE = f'OUTPUT -p tcp -d {sn_addr} -j DROP'
CMD_MAP = {
    True: f'iptables -D {RULE}',
    False: f'iptables -A {RULE}',
}


def work(container_name:str):
    cryptogen = random.SystemRandom()
    client = docker.from_env()
    container = client.containers.get(container_name)

    last_connectivity = True
    cmd = CMD_MAP[last_connectivity]
    print(container_name, cmd)
    container.exec_run(cmd)
    # os.system(cmd)

    result = []

    disconnect_count = 0
    
    while container.status == 'running':
        connectivity = False if cryptogen.random() * 100 < loss_probability else True
        if not connectivity:
            disconnect_count += 1
        if disconnect_count > 3:
            disconnect_count = 0
            connectivity = True
            print(f"{container_name} force connect!")
        result.append((time.time(), connectivity))
        
        if connectivity != last_connectivity:
            last_connectivity = connectivity
            cmd = CMD_MAP[connectivity]
            print(container_name, cmd)
            container.exec_run(cmd)
            # os.system(cmd)
        else:
            print(f'{container_name} remains the same: {connectivity}')
        
        time.sleep(interval_length)
        container = client.containers.get(container_name)
    
    print(f'{container_name} is down')
    return container_name,result

if __name__ == '__main__':
    current_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    sl_containers = ['sl-{0:.2f}-0-{1}'.format(loss_probability/100.0, i) for i in range(sl_num)]

    print('CPU count:', mp.cpu_count())

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map_async(work, sl_containers).get()
    
    with open(f'results-{current_time}.pickle', 'wb') as f:
        pickle.dump(results, f)

