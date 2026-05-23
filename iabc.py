"""
IABC (Improved Artificial Bee Colony) baseline for the bi-objective HFSP
(min makespan, max yield).

nsga2.py 와 *완전히 동일한 고정 인스턴스* 를 풀어 Hypervolume(HV) 로 비교하기 위한
또 하나의 메타휴리스틱 비교군. 알고리즘 코어(탐색 전략)만 다르고,
인스턴스 정의 / 인코딩 / 디코더(env) / 평가 / HV 지표는 nsga2.py 와 1:1 동일하다.

  → 공정 비교를 위해 평가·지표·인코딩 인프라를 nsga2.py 에서 그대로 import 한다.
    (HFSPDecoder, PathPercentileScorer, hypervolume, compute_pareto_front,
     plot_results, fast_nondominated_sort, crowding_distance, assign_rank_crowd,
     pox_crossover, ma_crossover, mutate_os, mutate_ma, init_population, dominates)

Encoding (nsga2.py 와 동일):
  OS (operation sequence): 길이 J·S 의 job-id 반복열 (dispatch 우선순위 + precedence 보장).
  MA (machine assignment) : 각 op 의 stage 내 local machine idx ∈ [0, cnt_s).
Decode: OS 순서대로 env.step → insertion decoder (모델 평가와 의미 동일).

IABC (Improved Artificial Bee Colony) — 이산 스케줄링용 적응 + 개선점:
  food source  : 하나의 해 (OS, MA). 개수 SN = pop.
  fitness      : 다목적 → 비지배 rank 기반 (rank 작을수록 우수).
  employed bee : 각 food source 마다 이웃해 1개 생성 후 greedy(지배) 선택.
  onlooker bee : rank 기반 확률(roulette)로 food source 선택 → 이웃해 생성 후 greedy 선택.
  scout bee    : trial(개선 실패 누적) ≥ limit 인 food source 를 무작위 재초기화.
  이웃해 생성  : 표준 ABC 의 연속식 대신 *이산 연산* 사용 —
                 OS = POX(source, guide) + swap mutation,
                 MA = uniform-crossover(source, guide) + per-gene mutation.
  [Improved]   : guide 를 확률 p_global 로 *외부 Pareto archive* 의 elite 에서 선택
                 (gbest-guided ABC, GABC). 나머지는 임의 food source.
  archive      : 외부 비지배 archive (crowding 으로 cap 유지). 최종 HV/front 는 archive 로 평가.

비교 지표: nsga2.py 와 동일한 hypervolume() / anchor box 규약.
"""

from __future__ import annotations
import argparse
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')

from HFSPGraphEnv import HFSPGraphEnv
from FFSProblemDef import load_problems_from__quality_file
from HFSPWrapper import QualityHelper

# ── 평가/지표/인코딩 인프라는 nsga2.py 와 *완전히 동일* 하게 공유 (공정 비교) ──
from nsga2 import (
    HFSPDecoder, PathPercentileScorer, hypervolume, compute_pareto_front,
    plot_results, fast_nondominated_sort, crowding_distance, assign_rank_crowd,
    pox_crossover, ma_crossover, mutate_os, mutate_ma, init_population, dominates,
)


# =====================================================
# 평가 helper: population → (ms, gt, obj, objs)
# =====================================================
def eval_pop(decoder: HFSPDecoder, OS: np.ndarray, MA: np.ndarray):
    """(OS, MA) 배치를 한 번에 평가. objs = [makespan, -obj_yield] (둘 다 minimize)."""
    ms, gt, obj = decoder.evaluate(OS, MA)
    objs = np.stack([ms, -obj], axis=1)
    return ms, gt, obj, objs


# =====================================================
# 이웃해 생성 (이산 ABC + gbest guidance)
# =====================================================
def make_neighbors(OS, MA, archive_OS, archive_MA, J, cnt_per_op, rng, *,
                   p_global, p_cross_os, p_cross_ma, p_mut_os, p_mut_ma):
    """각 source 마다 이웃해 1개 생성.
    guide = 확률 p_global 로 archive elite, 아니면 임의의 다른 food source.
    OS = POX(source, guide) + swap mut,  MA = uniform-cross(source, guide) + gene mut.
    """
    SN = OS.shape[0]
    n_arch = archive_OS.shape[0]
    nOS = np.empty_like(OS)
    nMA = np.empty_like(MA)
    for i in range(SN):
        if n_arch > 0 and rng.random() < p_global:           # [Improved] 전역 elite 유도
            g = int(rng.integers(0, n_arch))
            gOS, gMA = archive_OS[g], archive_MA[g]
        else:                                                # 임의 동료 food source
            k = i
            while k == i and SN > 1:
                k = int(rng.integers(0, SN))
            gOS, gMA = OS[k], MA[k]
        os_c = (pox_crossover(OS[i], gOS, J, rng)
                if rng.random() < p_cross_os else OS[i].copy())
        ma_c = (ma_crossover(MA[i], gMA, rng)
                if rng.random() < p_cross_ma else MA[i].copy())
        nOS[i] = mutate_os(os_c, rng, p_mut_os)
        nMA[i] = mutate_ma(ma_c, cnt_per_op, rng, p_mut_ma)
    return nOS, nMA


# =====================================================
# greedy(지배) 선택 — candidate 가 source 를 개선하면 교체
# =====================================================
def greedy_select(OS, MA, ms, gt, obj, objs, trial,
                  cOS, cMA, c_ms, c_gt, c_obj, c_objs, src_idx, rng):
    """candidate n 을 food source src_idx[n] 와 비교 (in-place 갱신).

    규칙:
      cand 가 source 를 지배        → 교체, trial 리셋(개선).
      source 가 cand 를 지배        → 유지, trial += 1.
      상호 비지배                   → 확률 0.5 로 교체(다양성), trial += 1.
    """
    for n in range(len(src_idx)):
        i = int(src_idx[n])
        c, s = c_objs[n], objs[i]
        if dominates(c, s):
            accept, improved = True, True
        elif dominates(s, c):
            accept, improved = False, False
        else:
            accept, improved = (rng.random() < 0.5), False
        if accept:
            OS[i], MA[i] = cOS[n], cMA[n]
            ms[i], gt[i], obj[i], objs[i] = c_ms[n], c_gt[n], c_obj[n], c_objs[n]
        trial[i] = 0 if improved else trial[i] + 1


# =====================================================
# 외부 Pareto archive 갱신 (비지배 + crowding 으로 cap 유지)
# =====================================================
def update_archive(arch, OS, MA, ms, gt, obj, cap):
    """기존 archive 에 새 점들을 합쳐 비지배 front0 만 남기고, cap 초과 시 crowding 으로 절단.
    비지배 판정은 objs=[ms, -obj] (선택 목적함수) 기준 — nsga2.py 선택 규약과 동일.
    """
    if arch is None:
        aOS, aMA = OS.copy(), MA.copy()
        ams, agt, aobj = ms.copy(), gt.copy(), obj.copy()
    else:
        aOS = np.concatenate([arch['OS'], OS], axis=0)
        aMA = np.concatenate([arch['MA'], MA], axis=0)
        ams = np.concatenate([arch['ms'], ms])
        agt = np.concatenate([arch['gt'], gt])
        aobj = np.concatenate([arch['obj'], obj])
    aobjs = np.stack([ams, -aobj], axis=1)

    # (ms, obj) 중복 제거 (동일 schedule 누적 방지)
    _, uniq = np.unique(aobjs.round(6), axis=0, return_index=True)
    uniq = np.sort(uniq)
    aOS, aMA, ams, agt, aobj, aobjs = (
        aOS[uniq], aMA[uniq], ams[uniq], agt[uniq], aobj[uniq], aobjs[uniq])

    # 비지배 front0 만 유지
    f0 = fast_nondominated_sort(aobjs)[0]
    aOS, aMA, ams, agt, aobj, aobjs = (
        aOS[f0], aMA[f0], ams[f0], agt[f0], aobj[f0], aobjs[f0])

    if aobjs.shape[0] > cap:                                  # crowding 절단
        cd = crowding_distance(aobjs)
        keep = np.argsort(-cd)[:cap]
        aOS, aMA, ams, agt, aobj, aobjs = (
            aOS[keep], aMA[keep], ams[keep], agt[keep], aobj[keep], aobjs[keep])

    return {'OS': aOS, 'MA': aMA, 'ms': ams, 'gt': agt, 'obj': aobj, 'objs': aobjs}


# =====================================================
# Main IABC loop
# =====================================================
def run_iabc(decoder: HFSPDecoder, J, S, machine_cnt_list, *,
             pop_size=100, n_gen=100, limit=20, p_global=0.5,
             p_cross_os=1.0, p_cross_ma=1.0, p_mut_os=0.2, p_mut_ma_gene=0.05,
             archive_cap=200, seed=0, hv_anchors=(100.0, 500.0, 0.85, 0.50)):
    rng = np.random.default_rng(seed)
    m_best, m_ref, q_best, q_ref = hv_anchors

    # ── 초기 food sources ──
    OS, MA, cnt_per_op = init_population(pop_size, J, S, machine_cnt_list, rng)
    ms, gt, obj, objs = eval_pop(decoder, OS, MA)
    trial = np.zeros(pop_size, dtype=np.int64)
    archive = update_archive(None, OS, MA, ms, gt, obj, archive_cap)

    nb_kw = dict(p_cross_os=p_cross_os, p_cross_ma=p_cross_ma,
                 p_mut_os=p_mut_os, p_mut_ma=p_mut_ma_gene)

    hv_hist = []
    t_prev = time.time()
    for gen in range(n_gen):
        # ── (1) Employed bee phase ──
        nOS, nMA = make_neighbors(OS, MA, archive['OS'], archive['MA'],
                                  J, cnt_per_op, rng, p_global=p_global, **nb_kw)
        n_ms, n_gt, n_obj, n_objs = eval_pop(decoder, nOS, nMA)
        greedy_select(OS, MA, ms, gt, obj, objs, trial,
                      nOS, nMA, n_ms, n_gt, n_obj, n_objs, np.arange(pop_size), rng)
        archive = update_archive(archive, nOS, nMA, n_ms, n_gt, n_obj, archive_cap)

        # ── (2) Onlooker bee phase (rank 기반 roulette) ──
        rank, _ = assign_rank_crowd(objs)
        fit = 1.0 / (rank + 1.0)
        prob = fit / fit.sum()
        sel_idx = rng.choice(pop_size, size=pop_size, p=prob)
        oOS, oMA = make_neighbors(OS[sel_idx], MA[sel_idx], archive['OS'], archive['MA'],
                                  J, cnt_per_op, rng, p_global=p_global, **nb_kw)
        o_ms, o_gt, o_obj, o_objs = eval_pop(decoder, oOS, oMA)
        greedy_select(OS, MA, ms, gt, obj, objs, trial,
                      oOS, oMA, o_ms, o_gt, o_obj, o_objs, sel_idx, rng)
        archive = update_archive(archive, oOS, oMA, o_ms, o_gt, o_obj, archive_cap)

        # ── (3) Scout bee phase ──
        scouts = np.where(trial >= limit)[0]
        if scouts.size:
            sOS, sMA, _ = init_population(scouts.size, J, S, machine_cnt_list, rng)
            s_ms, s_gt, s_obj, s_objs = eval_pop(decoder, sOS, sMA)
            OS[scouts], MA[scouts] = sOS, sMA
            ms[scouts], gt[scouts], obj[scouts], objs[scouts] = s_ms, s_gt, s_obj, s_objs
            trial[scouts] = 0
            archive = update_archive(archive, sOS, sMA, s_ms, s_gt, s_obj, archive_cap)

        # ── HV 는 항상 archive 의 ground-truth yield 로 평가 ──
        hv, _ = hypervolume(archive['ms'], archive['gt'], m_ref, q_ref)
        hv_hist.append(hv)
        now = time.time()
        gen_dt, t_prev = now - t_prev, now
        if gen == 0 or (gen + 1) % 10 == 0 or gen == n_gen - 1:
            print(f"[cycle {gen+1:4d}/{n_gen}] HV_raw={hv:10.4f}  "
                  f"|archive|={archive['ms'].size:3d}  ms[min={archive['ms'].min():.1f}]  "
                  f"GTyld[max={archive['gt'].max():.4f}]  scouts={scouts.size:2d}  "
                  f"({gen_dt:.2f}s/cycle)")

    return archive, np.asarray(hv_hist)


# =====================================================
# Entry  (인스턴스 셋업은 nsga2.py main() 과 1:1 동일)
# =====================================================
def main():
    p = argparse.ArgumentParser(description="IABC baseline for HFSP (test.py / nsga2.py 비교군)")
    p.add_argument('--num_jobs',  type=int, default=25)
    p.add_argument('--machines',  type=str, default='5,3,7,3,5,7')
    p.add_argument('--p_idx',     type=int, choices=[1, 2, 3], default=3)
    p.add_argument('--q_idx',     type=int, choices=[1, 2, 3], default=3)
    p.add_argument('--paths_idx', type=int, default=1)
    p.add_argument('--yield_mode', type=str, default='raw', choices=['raw', 'percentile'],
                   help="HV/리포트용 ground-truth yield 계산 방식 (scorer 기반).")
    p.add_argument('--yield_objective', type=str, default='predicted',
                   choices=['ground_truth', 'predicted'],
                   help="IABC 선택(domination)에 쓰는 yield. "
                        "'ground_truth' = scorer GT (oracle 상한), "
                        "'predicted' = QualityHelper.compute_yield (모델 학습 신호와 동일 → 공정 비교). "
                        "어느 쪽이든 최종 HV/plot 은 ground-truth 로 평가.")
    p.add_argument('--wafer_quality', type=str, default='0.99,1.00',
                   help="초기 웨이퍼 품질 U[lo,hi]. test.py/nsga2.py 와 동일하게 0.99,1.00.")
    p.add_argument('--seed',      type=int, default=0,
                   help="wafer_quality / proc_time 인스턴스용 seed (test.py 와 동일하게 0).")
    p.add_argument('--ga_seed',   type=int, default=0, help="IABC RNG seed.")
    # IABC hyperparams
    p.add_argument('--pop',       type=int, default=100, help="food source 수 SN.")
    p.add_argument('--gen',       type=int, default=50, help="cycle 수 MCN.")
    p.add_argument('--limit',     type=int, default=20,
                   help="scout 발동 임계 (개선 실패 누적 trial ≥ limit → 재초기화).")
    p.add_argument('--p_global',  type=float, default=0.5,
                   help="이웃해 생성 시 archive elite 를 guide 로 쓸 확률 (gbest-guided).")
    p.add_argument('--archive_cap', type=int, default=200, help="외부 Pareto archive 용량.")
    p.add_argument('--p_mut_os',  type=float, default=0.2)
    p.add_argument('--p_mut_ma',  type=float, default=0.05, help="MA per-gene mutation prob.")
    # 축 범위 + HV anchor (nsga2.py 와 동일 규약: 'lo,hi,step', tick 은 hi 제외).
    p.add_argument('--xlim', type=str, default='100,600,100',
                   help="x축 (makespan) 'lo,hi' 또는 'lo,hi,step'. HV anchor m=[best=lo, ref=hi].")
    p.add_argument('--ylim', type=str, default='0.0,1.0,0.1',
                   help="y축 (yield) 'lo,hi' 또는 'lo,hi,step'. HV anchor q=[ref=lo, best=hi].")
    args = p.parse_args()

    machine_cnt_list = [int(x) for x in args.machines.split(',')]
    J, S = args.num_jobs, len(machine_cnt_list)
    wq_lo, wq_hi = [float(x) for x in args.wafer_quality.split(',')]

    def _parse_axis(s):
        parts = [float(x) for x in s.split(',')]
        if len(parts) == 2:
            return (parts[0], parts[1]), None
        if len(parts) == 3:
            lo, hi, step = parts
            ticks = np.arange(lo, hi, step)
            if ticks.size == 0 or abs(ticks[-1] - hi) > 1e-9:
                ticks = np.append(ticks, hi)
            return (lo, hi), ticks
        raise ValueError(f"axis must be 'lo,hi' or 'lo,hi,step', got: {s!r}")

    xlim, xticks = _parse_axis(args.xlim)
    ylim, yticks = _parse_axis(args.ylim)
    hv_m_best, hv_m_ref = xlim
    hv_q_ref,  hv_q_best = ylim
    if args.yield_mode == 'percentile':
        hv_q_best, hv_q_ref = 100.0, 0.0
        ylim, yticks = (0.0, 100.0), None
    hv_anchors = (hv_m_best, hv_m_ref, hv_q_best, hv_q_ref)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)

    # ── env / 인스턴스 (nsga2.py 와 1:1 동일) ──
    env = HFSPGraphEnv(num_jobs=J, machine_cnt_list=machine_cnt_list, device=device)
    problems_INT_list = load_problems_from__quality_file(
        f'quality_data/P_{args.p_idx}.csv', J, machine_cnt_list)
    env_edge_lookup_np = env.env_edge_lookup.cpu().numpy()
    machine_offset_np = np.cumsum([0] + machine_cnt_list[:-1]).astype(np.int64)

    # ── wafer_quality: test.py(QualityHelper.sample_wafer_quality) 와 동일 RNG ──
    wq_csv = f'quality_data/Q_{args.q_idx}/historical_paths_{args.paths_idx}.csv'
    model_path = f'quality_results/Q_{args.q_idx}/paths_{args.paths_idx}/best_hb_model.zip'
    wafer_path = f'quality_results/Q_{args.q_idx}/paths_{args.paths_idx}/wafer_quality.json'
    qh = QualityHelper(num_stages=S, device=device,
                       model_path=model_path, wafer_path=wafer_path)
    if not qh.is_active:
        raise RuntimeError(f"quality pipeline not available — {model_path}, {wafer_path}")
    qh.wafer_quality_min, qh.wafer_quality_max = wq_lo, wq_hi
    wq_1 = qh.sample_wafer_quality(B=1, num_jobs=J, seed=args.seed, device=device)
    wq_np = wq_1[0].cpu().numpy().astype(np.float64)

    scorer = PathPercentileScorer(wq_csv, machine_cnt_list)

    print(f"[iabc] device={device}  J={J}  machines={machine_cnt_list}  NE={env.num_edges}  "
          f"yield_mode={args.yield_mode}  yield_objective={args.yield_objective}")
    print(f"[iabc] instance: P_{args.p_idx}, Q_{args.q_idx}, paths_{args.paths_idx}  "
          f"wafer_q=U[{wq_lo:.4f},{wq_hi:.4f}] mean={wq_np.mean():.4f}")
    print(f"[iabc] HV anchors: m=[best={hv_m_best}, ref={hv_m_ref}]  "
          f"q=[ref={hv_q_ref}, best={hv_q_best}]")
    print(f"[iabc] SN(pop)={args.pop}  cycles(gen)={args.gen}  limit={args.limit}  "
          f"p_global={args.p_global}  archive_cap={args.archive_cap}  "
          f"p_mut_os={args.p_mut_os}  p_mut_ma={args.p_mut_ma}  ga_seed={args.ga_seed}")
    print(f"[iabc] evals/cycle ≈ 2·SN (+scouts) = {2 * args.pop}+  "
          f"(employed {args.pop} + onlooker {args.pop})")

    decoder = HFSPDecoder(env, problems_INT_list, env_edge_lookup_np,
                          machine_offset_np, scorer, wq_np, yield_mode=args.yield_mode,
                          yield_objective=args.yield_objective, quality_helper=qh)

    archive, hv_hist = run_iabc(
        decoder, J, S, machine_cnt_list,
        pop_size=args.pop, n_gen=args.gen, limit=args.limit, p_global=args.p_global,
        p_mut_os=args.p_mut_os, p_mut_ma_gene=args.p_mut_ma, archive_cap=args.archive_cap,
        seed=args.ga_seed, hv_anchors=hv_anchors)

    # ── 최종 HV / front : 항상 archive 의 ground-truth yield(gt) 로 평가 ──
    ms_a, gt_a, obj_a = archive['ms'], archive['gt'], archive['obj']
    yld = gt_a
    hv_raw, _ = hypervolume(ms_a, yld, hv_m_ref, hv_q_ref)
    box = (hv_m_ref - hv_m_best) * (hv_q_best - hv_q_ref)
    hv_norm = hv_raw / box if box > 0 else float('nan')
    front_idx = compute_pareto_front(ms_a, yld)

    print("\n========== IABC result ==========")
    print(f"yield_objective = {args.yield_objective}"
          + ("  (IABC 가 예측 yield 로 선택 → HV 는 GT 로 평가)"
             if args.yield_objective == 'predicted' else "  (GT oracle)"))
    if args.yield_objective == 'predicted':
        print(f"objective(pred) yield: min={obj_a.min():.4f}  max={obj_a.max():.4f}  "
              f"(GT 와의 평균 괴리={np.abs(obj_a - gt_a).mean():.4f})")
    print(f"archive size = {ms_a.size}   Pareto front size = {len(front_idx)}")
    print(f"makespan: min={ms_a.min():.2f}  max={ms_a.max():.2f}")
    print(f"yield(GT): min={yld.min():.4f}  max={yld.max():.4f}")
    print(f"HV_raw  = {hv_raw:.4f}   (ref=({hv_m_ref:.0f}, {hv_q_ref:.2f}))")
    print(f"HV_norm = {hv_norm:.4f}  (anchor box [{hv_m_best:.0f},{hv_m_ref:.0f}]"
          f"×[{hv_q_ref:.2f},{hv_q_best:.2f}])")

    # ── 저장 ──
    tag = f'iabc_J{J}_Q{args.q_idx}_P{args.p_idx}_p{args.paths_idx}'
    png_path = f'test_results/{tag}.png'
    title = (f'IABC  W={J}, (Q{args.q_idx},P{args.p_idx}), paths_{args.paths_idx}')
    plot_results(ms_a, yld, png_path, title, hv_norm,
                 xlim=xlim, ylim=ylim, xticks=xticks, yticks=yticks)
    print(f"\nsaved -> {png_path}")


if __name__ == "__main__":
    main()
