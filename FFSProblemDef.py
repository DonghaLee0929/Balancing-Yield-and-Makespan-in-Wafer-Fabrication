
"""
The MIT License

Copyright (c) 2021 MatNet

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.



THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import torch


def get_random_problems(batch_size, stage_cnt, machine_cnt_list, job_cnt, process_time_params,
                        seed=None, device=None):
    # seed 가 주어지면 로컬 torch.Generator 로 결정론적 샘플링 (전역 RNG 비오염).
    # seed=None 이면 전역 RNG 사용 (FFSPEnv.load_problems 처럼 매번 다른 문제를 원할 때).
    # device 가 주어지면 처음부터 해당 device 에 생성 (CPU→GPU 복사 제거).
    time_low = process_time_params['time_low']
    time_high = process_time_params['time_high']

    randint_kwargs = {}
    if device is not None:
        randint_kwargs['device'] = device
    if seed is not None:
        gen = torch.Generator(device=device if device is not None else 'cpu')
        gen.manual_seed(int(seed))
        randint_kwargs['generator'] = gen

    problems_INT_list = []
    for stage_num in range(stage_cnt):
        machine_cnt = machine_cnt_list[stage_num]
        stage_problems_INT = torch.randint(
            low=time_low, high=time_high,
            size=(batch_size, job_cnt, machine_cnt),
            **randint_kwargs,
        )
        problems_INT_list.append(stage_problems_INT)

    return problems_INT_list
    # type(problems_list) = list
    # len(problems_list) = stage_cnt
    # problems_list[stage_num].shape: (batch, job, machine_cnt_list[stage_num])

def get_random_problems_identical_jobs(batch_size, stage_cnt, machine_cnt_list,
                                       job_cnt, process_time_params, seed=None, device=None):
    """(stage, machine) 마다 1개 random proc_time, 모든 job 이 공유 (job-independent).

    get_random_problems 와 동일한 shape 의 list 를 반환하지만, job 차원에 대해
    같은 값이 broadcast 되어 있다.

    Returns:
        list of stage_cnt tensors, 각 (batch_size, job_cnt, machine_cnt_list[s]) int64.
    """
    time_low = process_time_params['time_low']
    time_high = process_time_params['time_high']

    randint_kwargs = {}
    if device is not None:
        randint_kwargs['device'] = device
    if seed is not None:
        gen = torch.Generator(device=device if device is not None else 'cpu')
        gen.manual_seed(int(seed))
        randint_kwargs['generator'] = gen

    problems_INT_list = []
    for stage_num in range(stage_cnt):
        machine_cnt = machine_cnt_list[stage_num]
        # (batch, 1, machine_cnt) → expand over job 차원
        per_machine = torch.randint(
            low=time_low, high=time_high,
            size=(batch_size, 1, machine_cnt),
            **randint_kwargs,
        )
        stage_problems_INT = per_machine.expand(
            batch_size, job_cnt, machine_cnt
        ).contiguous()
        problems_INT_list.append(stage_problems_INT)
    return problems_INT_list


def load_problems_from__quality_file(filename, job_cnt, machine_cnt_list,
                                     device=torch.device('cpu')):
    """quality_data/P_*.csv 형식 (Step, Machine, ProcessingTime) → problems_INT_list.

    모든 job 이 같은 (stage, machine) proc_time 매트릭스를 공유 (job-independent).
    CSV 의 Step / Machine 은 1-indexed.

    Args:
        filename:           CSV 경로 (예: 'quality_data/P_1.csv').
        job_cnt:            job 차원으로 broadcast 할 작업 수.
        machine_cnt_list:   stage 별 머신 수 — CSV 의 row 분포를 검증하는 용도.

    Returns:
        list of len(machine_cnt_list) tensors, 각 shape (1, job_cnt, machine_cnt_list[s]).
        (batch=1; 여러 시나리오를 stack 하려면 호출자가 torch.cat dim=0.)
    """
    import pandas as pd
    df = pd.read_csv(filename)
    problems_INT_list = []
    for stage_num, machine_cnt in enumerate(machine_cnt_list):
        rows = df[df['Step'] == stage_num + 1].sort_values('Machine')
        if len(rows) != machine_cnt:
            raise ValueError(
                f"Stage {stage_num + 1}: CSV 에 {len(rows)} 행, "
                f"machine_cnt={machine_cnt} 와 불일치")
        times = torch.tensor(
            rows['ProcessingTime'].to_numpy(), dtype=torch.long, device=device
        )                                                          # (machine_cnt,)
        # (1, 1, machine_cnt) → expand over (batch=1, job)
        stage_problems_INT = (
            times.view(1, 1, machine_cnt)
                 .expand(1, job_cnt, machine_cnt)
                 .contiguous()
        )
        problems_INT_list.append(stage_problems_INT)
    return problems_INT_list


def load_problems_from_file(filename, device=torch.device('cpu')):
    data = torch.load(filename)

    problems_INT_list = data['problems_INT_list']

    for stage_idx in range(data['stage_cnt']):
        problems_INT_list[stage_idx] = problems_INT_list[stage_idx].to(device)

    return problems_INT_list


def load_ONE_problem_from_file(filename, device=torch.device('cpu'), index=0):
    data = torch.load(filename)

    problems_INT_list = data['problems_INT_list']
    problems_list = data['problems_list']

    for stage_idx in range(data['stage_cnt']):
        problems_INT_list[stage_idx] = problems_INT_list[stage_idx][[index], :, :]
        problems_INT_list[stage_idx] = problems_INT_list[stage_idx].to(device)

        problems_list[stage_idx] = problems_list[stage_idx][[index], :, :]
        problems_list[stage_idx] = problems_list[stage_idx].to(device)

    return problems_INT_list, problems_list
  
