"""
Ground-truth quality augmentation — 단일 모듈
=============================================

historical_paths CSV 의 **실제 (stage, machine) 품질 인자**(deterministic) 를 그대로 읽어,
yield 예측 모델(best_hb_model.zip) 없이 ground-truth quality 를 직접 제공한다.

핵심 사실 (검증: diff=0.0):
    Yield(path) = wafer_quality · Π_s step_q[s, m_s]
    → 완전 곱셈 모델. CSV 의 Yield 컬럼은 step quality 들의 곱 × Wafer quality 와 정확히 일치.

따라서 학습된 회귀모델 없이, 한 시나리오의 compact 한 (S, machine) 품질 인자 테이블만으로
ground-truth quality(피처·리워드·평가)를 전부 제공할 수 있다. 예측 모델 개념이 사라진다.

머신 수 증강 (stage 수는 CSV 의 6 으로 고정, 머신 수만 가변):
    target 머신 m(0-based) 은 base 머신 (m % base_cnt_s) 을 **그대로 복제**한다.
        target == base  → 동일 (변화 없음)
        target == 2·base → 정확히 동일한 품질 인자가 2 벌 (exact 복제)
        target  < base  → 앞쪽 k 개 subset
    proc_time(P CSV) 도 같은 규칙으로 복제 → 복제 머신은 품질·처리시간 모두 동일.

이 모듈이 제공하는 quality 는 세 곳에서 동일 ground truth 로 쓰인다:
    ① 상태 엣지 피처  : compute_machine_quality  → stage-내 z-score (곱셈 모델이라 prefix·wafer 무관)
    ② 시뮬레이션 리워드: compute_yield           → Π step_q × wafer_quality
    ③ 평가 채점        : path_product / score     → raw 곱 / 전체 분포 대비 백분위

인터페이스 호환:
    - QualityHelper (HFSPWrapper.py) 호환: is_active / sample_wafer_quality /
      compute_yield / compute_machine_quality / wafer_quality_min,max
    - PathPercentileScorer (test.py, nsga2.py) 호환: step_q / machine_cnt_list /
      num_stages / path_product / score / all_products_sorted / n_paths
    → 한 객체를 quality_helper 와 scorer 양쪽에 그대로 넘길 수 있다 (test.py·nsga2.py 무수정).
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import torch


def _read_step_q_base(df: pd.DataFrame, num_stages: int) -> tuple[np.ndarray, list[int]]:
    """CSV 에서 base (stage, machine) 품질 인자 + stage 별 base 머신 수 추출.

    각 (stage, machine) 의 'Step s quality' 가 CSV 내에서 단일값이어야 함 (deterministic).
    Returns: (step_q_base (S, max_base), base_cnt_list (S,)).
    """
    S = num_stages
    base_cnt = [int(df[f'Step {s + 1}'].max()) for s in range(S)]
    max_base = max(base_cnt)
    step_q_base = np.zeros((S, max_base), dtype=np.float64)
    for s in range(S):
        for m in range(1, base_cnt[s] + 1):
            vals = df.loc[df[f'Step {s + 1}'] == m, f'Step {s + 1} quality'].unique()
            if len(vals) != 1:
                raise ValueError(
                    f"(stage {s + 1}, machine {m}) quality 가 {len(vals)} 종류 — "
                    "CSV 가 deterministic 하지 않음 (per-(stage,machine) 단일값 필요)")
            step_q_base[s, m - 1] = vals[0]
    return step_q_base, base_cnt


def load_proc_time_augmented(p_csv: str, job_cnt: int, machine_cnt_list,
                             device=torch.device('cpu')):
    """P CSV (Step, Machine, ProcessingTime) → target machine_cnt_list 로 복제 증강한 problems_INT_list.

    target 머신 m 은 base 머신 (m % base_cnt_s) 의 처리시간을 그대로 복제 (머신 증가 = exact 복제).
    모든 job 이 같은 (stage, machine) proc_time 을 공유 (job-independent).

    Returns: list of len(machine_cnt_list) tensors, 각 (1, job_cnt, machine_cnt_list[s]) long.
    (load_problems_from__quality_file 와 동일 출력 규격 — drop-in)
    """
    df = pd.read_csv(p_csv)
    S = len(machine_cnt_list)
    out = []
    for s in range(S):
        rows = df[df['Step'] == s + 1].sort_values('Machine')
        base_cnt = len(rows)
        if base_cnt == 0:
            raise ValueError(f"P CSV stage {s + 1} 행 없음 — {p_csv}")
        base_times = torch.tensor(rows['ProcessingTime'].to_numpy(),
                                  dtype=torch.long, device=device)        # (base_cnt,)
        T = int(machine_cnt_list[s])
        idx = torch.arange(T, device=device) % base_cnt                   # m % base
        times = base_times[idx]                                           # (T,) 복제
        stage = times.view(1, 1, T).expand(1, job_cnt, T).contiguous()
        out.append(stage)
    return out


class GroundTruthQuality:
    """historical_paths CSV 의 exact 품질 인자로 동작하는 ground-truth quality provider.

    quality_helper(QualityHelper) + scorer(PathPercentileScorer) 인터페이스를 동시에 구현 →
    한 객체로 ① 엣지 피처 ② 리워드 ③ 평가 채점을 모두 ground truth 로 제공.

    Args:
        num_stages:        공정 stage 수 (S). CSV stage 수와 같아야 함 (stage 증감 불가).
        device:            텐서 device.
        q_idx, paths_idx:  csv_path 미지정 시 quality_data/Q_{q}/historical_paths_{p}.csv 로 구성.
        csv_path:          품질 인자 원본 CSV 직접 지정 (우선).
        machine_cnt_list:  TARGET stage 별 머신 수. None → CSV 의 base 그대로.
                           base 초과 시 (m % base) 복제 증강, 미만 시 앞쪽 subset.
        wafer_quality_min/max: 초기 웨이퍼 품질 U[min,max] 샘플 범위.
        z_scale:           엣지 피처 z-score 스케일 (원본 compute_machine_quality Z_SCALE=1.0 호환).
    """

    def __init__(self, num_stages: int, device,
                 q_idx: int = 3, paths_idx: int = 1,
                 csv_path: str | None = None,
                 machine_cnt_list=None,
                 wafer_quality_min: float = 0.99,
                 wafer_quality_max: float = 1.00,
                 z_scale: float = 1.0):
        self.num_stages = int(num_stages)
        self.device = device
        self.wafer_quality_min = float(wafer_quality_min)
        self.wafer_quality_max = float(wafer_quality_max)
        self.z_scale = float(z_scale)

        if csv_path is None:
            csv_path = f"quality_data/Q_{q_idx}/historical_paths_{paths_idx}.csv"
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"[quality_augment] 품질 인자 원본 CSV 없음 → {csv_path}. "
                "데이터가 없어도 되는 모드가 아니라, 이 데이터를 증강하는 모드입니다.")
        self.csv_path = csv_path

        S = self.num_stages
        df = pd.read_csv(csv_path)

        # ── base 품질 인자 + base 머신 수 (CSV) ──
        step_q_base, base_cnt = _read_step_q_base(df, S)
        self.base_machine_cnt_list = base_cnt

        # ── target 머신 수 (기본 = base). stage 수는 고정. ──
        target = list(base_cnt) if machine_cnt_list is None else [int(x) for x in machine_cnt_list]
        if len(target) != S:
            raise ValueError(
                f"machine_cnt_list 길이 {len(target)} != num_stages {S} — "
                "stage 수는 CSV 에 고정 (머신 수만 증강 가능)")
        if any(t < 1 for t in target):
            raise ValueError(f"각 stage 머신 수는 >=1 이어야 함, got {target}")
        self.machine_cnt_list = target

        # ── (m % base) 복제로 augmented step_q (S, max_target) ──
        max_t = max(target)
        step_q = np.zeros((S, max_t), dtype=np.float64)
        for s in range(S):
            for m in range(target[s]):
                step_q[s, m] = step_q_base[s, m % base_cnt[s]]
        self.step_q = step_q                                              # (S, max_target) np
        self.step_q_t = torch.tensor(step_q, dtype=torch.float32, device=device)

        # ── stage-내 z-score (prefix·wafer 무관, min→0). 복제 머신은 동일 z-score. ──
        zq_local = np.zeros((S, max_t), dtype=np.float64)
        for s in range(S):
            cnt = target[s]
            qs = step_q[s, :cnt]
            if cnt > 1 and np.ptp(qs) > 0:
                mu, std = qs.mean(), qs.std(ddof=1)                       # torch .std() 와 동일(unbiased)
                z = (qs - mu) / max(std, 1e-8) * self.z_scale
                z = z - z.min()
            else:
                z = np.zeros(cnt, dtype=np.float64)                       # 단일/동일값 stage → 0
            zq_local[s, :cnt] = z
        self.zq_local_t = torch.tensor(zq_local, dtype=torch.float32, device=device)
        self._zq_global = None                                           # env 기반 lazy (M,)
        self._all_products_sorted = None                                 # percentile lazy

        aug = "" if target == base_cnt else f" → augmented {target} (m%base 복제)"
        print(f"[quality_augment] {csv_path}: stages={S} base_machine_cnt={base_cnt}{aug}  "
              f"step_q range=[{step_q[step_q > 0].min():.4f}, {step_q.max():.4f}]  "
              f"wafer_q=[{self.wafer_quality_min:.4f}, {self.wafer_quality_max:.4f}]")

    # ============================================================ QualityHelper 인터페이스
    @property
    def is_active(self) -> bool:
        return True

    def sample_wafer_quality(self, B: int, num_jobs: int, seed: int, device) -> torch.Tensor:
        """U(wafer_quality_min, wafer_quality_max) 에서 (B, num_jobs) 샘플."""
        g = torch.Generator(device=device).manual_seed(int(seed))
        return torch.empty((B, num_jobs), dtype=torch.float32, device=device).uniform_(
            self.wafer_quality_min, self.wafer_quality_max, generator=g)

    def compute_yield(self, env, wafer_quality: torch.Tensor | None,
                      aggregate: str = "mean") -> torch.Tensor:
        """② 리워드 ground truth — env 완성 path 따라 Π step_q × wafer_quality.

        QualityHelper.compute_yield 와 동일 시그니처/반환 규칙.
        """
        B, J, S = env.batch_size, env.num_jobs, env.num_stages
        zeros_bj = torch.zeros((B, J), dtype=torch.float32, device=env.device)
        zeros_b = torch.zeros(B, dtype=torch.float32, device=env.device)

        if wafer_quality is None:
            return zeros_bj if aggregate == "none" else zeros_b
        completed = env.op_assigned.all(dim=1)
        if not completed.any():
            return zeros_bj if aggregate == "none" else zeros_b

        op_m_done = env.op_machine[completed].view(-1, J, S)             # (Nd, J, S) global m
        local_m = env.machine_local[op_m_done.clamp(min=0)]             # (Nd, J, S) 0-based local
        wq_done = wafer_quality[completed]                              # (Nd, J)

        q_path = torch.ones_like(wq_done)                              # (Nd, J)
        for s in range(S):
            q_path = q_path * self.step_q_t[s, local_m[:, :, s]]
        y_done = q_path * wq_done                                      # ground-truth yield

        out_bj = zeros_bj.clone()
        out_bj[completed] = y_done
        if aggregate == "none":
            return out_bj
        if aggregate == "sum":
            return out_bj.sum(dim=1)
        return out_bj.mean(dim=1)

    def compute_machine_quality(self, env) -> torch.Tensor:
        """① 엣지 피처 ground truth — (BP, J, M) stage-내 z-score prior (broadcast)."""
        if self._zq_global is None:
            M = env.total_machines
            zq_g = torch.zeros(M, dtype=torch.float32, device=env.device)
            for s in range(self.num_stages):
                g_idx = (env.machine_stage == s).nonzero(as_tuple=False).view(-1)
                zq_g[g_idx] = self.zq_local_t[s, env.machine_local[g_idx]]
            self._zq_global = zq_g                                      # (M,)
        BP, J = env.batch_size, env.num_jobs
        return self._zq_global.view(1, 1, -1).expand(BP, J, -1)        # (BP, J, M)

    # ============================================================ PathPercentileScorer 인터페이스
    @property
    def all_products_sorted(self) -> np.ndarray:
        """모든 target path 의 (정렬된) product 분포 — percentile 채점용 (lazy)."""
        if self._all_products_sorted is None:
            S = self.num_stages
            n_total = int(np.prod(self.machine_cnt_list))
            if n_total > 5_000_000:
                raise MemoryError(
                    f"percentile 채점용 전체 path 수 {n_total:,} 가 너무 큼 "
                    f"(machines={self.machine_cnt_list}). raw yield_mode 를 쓰거나 머신 수를 줄이세요.")
            grids = np.meshgrid(*[self.step_q[s, :self.machine_cnt_list[s]] for s in range(S)],
                                indexing='ij')
            all_paths = np.stack(grids, axis=-1).reshape(-1, S)
            self._all_products_sorted = np.sort(all_paths.prod(axis=1))
        return self._all_products_sorted

    @property
    def n_paths(self) -> int:
        return int(self.all_products_sorted.size)

    def path_product(self, paths_1indexed: np.ndarray) -> np.ndarray:
        """paths: (..., S) 1-indexed → (stage,machine) quality factor 의 path 곱 (...,)."""
        S = self.num_stages
        if paths_1indexed.shape[-1] != S:
            raise ValueError(
                f"path 마지막 차원 {paths_1indexed.shape[-1]} != num_stages {S}")
        flat = paths_1indexed.reshape(-1, S).astype(np.int64)
        q = np.empty_like(flat, dtype=np.float64)
        for s in range(S):
            q[:, s] = self.step_q[s, flat[:, s] - 1]
        return q.prod(axis=1).reshape(paths_1indexed.shape[:-1])

    def score(self, paths_1indexed: np.ndarray) -> np.ndarray:
        """paths: (..., S) 1-indexed → percentile (..., ) in [0, 100]."""
        prod = self.path_product(paths_1indexed)
        rank = np.searchsorted(self.all_products_sorted, prod.reshape(-1), side='right')
        return (rank.astype(np.float64) / self.n_paths * 100.0).reshape(prod.shape)

    # ============================================================ 데이터 증강 유틸
    def replicate_dataframe(self, replicate: int = 1) -> pd.DataFrame:
        """원본 CSV 를 exact 복제 — N 배 = 정확히 동일한 행 N 벌 (가장 단순한 증강)."""
        if replicate < 1:
            raise ValueError(f"replicate must be >= 1, got {replicate}")
        return pd.concat([pd.read_csv(self.csv_path)] * replicate, ignore_index=True)

    def enumerate_all_paths(self, wafer_quality: float = 1.0) -> pd.DataFrame:
        """compact 품질 인자 → 모든 target path 의 exact ground-truth yield 데이터셋."""
        S = self.num_stages
        grids = np.meshgrid(*[np.arange(1, self.machine_cnt_list[s] + 1) for s in range(S)],
                            indexing='ij')
        all_paths = np.stack([g.reshape(-1) for g in grids], axis=-1)
        q = np.stack([self.step_q[s, all_paths[:, s] - 1] for s in range(S)], axis=-1)
        yld = q.prod(axis=1) * wafer_quality
        cols = {f'Step {s + 1}': all_paths[:, s] for s in range(S)}
        cols.update({f'Step {s + 1} quality': q[:, s] for s in range(S)})
        cols['Wafer quality'] = wafer_quality
        cols['Yield'] = yld
        return pd.DataFrame(cols)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--q_idx', type=int, default=3)
    p.add_argument('--paths_idx', type=int, default=1)
    p.add_argument('--machines', type=str, default='',
                   help="target machine_cnt_list (예: '10,6,14,6,10,14' = 2x). 빈값=base.")
    args = p.parse_args()
    dev = torch.device('cpu')
    target = [int(x) for x in args.machines.split(',')] if args.machines else None

    gq = GroundTruthQuality(num_stages=6, device=dev, q_idx=args.q_idx,
                            paths_idx=args.paths_idx, machine_cnt_list=target)

    # 1) base 행 yield == CSV Yield (곱셈 모델 일치)
    df = pd.read_csv(gq.csv_path)
    S = 6
    paths = df[[f'Step {s + 1}' for s in range(S)]].to_numpy()
    q = np.stack([gq.step_q[s, paths[:, s] - 1] for s in range(S)], axis=-1)
    yld_pred = q.prod(axis=1) * df['Wafer quality'].to_numpy()
    err = np.abs(yld_pred - df['Yield'].to_numpy()).max()
    print(f"[selftest] base rows max|Yield_pred - Yield_csv| = {err:.3e}")
    assert err < 1e-9

    # 2) 머신 2배 증강 → 복제 머신이 정확히 동일한 품질 인자인지
    base = gq.base_machine_cnt_list
    dbl = [2 * c for c in base]
    gq2 = GroundTruthQuality(num_stages=6, device=dev, q_idx=args.q_idx,
                             paths_idx=args.paths_idx, machine_cnt_list=dbl)
    ok = all(np.allclose(gq2.step_q[s, :base[s]], gq2.step_q[s, base[s]:2 * base[s]])
             for s in range(S))
    print(f"[selftest] 2x augment: replicated machines identical = {ok}  (base={base} -> {dbl})")
    assert ok, "복제 머신 품질 인자가 동일하지 않음"
    print("[selftest] OK - ground-truth match + exact machine replication")
