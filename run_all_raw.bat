@echo off
REM 6 predefined (num_jobs, q_idx, p_idx, xlim, ylim) combinations.
REM Change SAMPLES to control --samples for all 6 runs.

set SAMPLES=1

echo === [1/6] W=15, (Q1,P1) ===
python test.py --num_jobs 15 --samples %SAMPLES% --q_idx 1 --p_idx 1 --xlim "30,111,20"  --ylim "0,1"

echo === [2/6] W=15, (Q2,P2) ===
python test.py --num_jobs 15 --samples %SAMPLES% --q_idx 2 --p_idx 2 --xlim "50,231,20"  --ylim "0,1"

echo === [3/6] W=15, (Q3,P3) ===
python test.py --num_jobs 15 --samples %SAMPLES% --q_idx 3 --p_idx 3 --xlim "70,331,30"  --ylim "0,1"

echo === [4/6] W=25, (Q1,P1) ===
python test.py --num_jobs 25 --samples %SAMPLES% --q_idx 1 --p_idx 1 --xlim "40,181,30"  --ylim "0,1"

echo === [5/6] W=25, (Q2,P2) ===
python test.py --num_jobs 25 --samples %SAMPLES% --q_idx 2 --p_idx 2 --xlim "40,361,80"  --ylim "0,1"

echo === [6/6] W=25, (Q3,P3) ===
python test.py --num_jobs 25 --samples %SAMPLES% --q_idx 3 --p_idx 3 --xlim "100,501,100" --ylim "0,1"

echo === done ===
