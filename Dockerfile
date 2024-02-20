FROM hub.myenterpriselicense.hpe.com/hpe_eval/swarm-learning/sl-tf:0.3.0

RUN python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -U tensorflow numpy pandas
