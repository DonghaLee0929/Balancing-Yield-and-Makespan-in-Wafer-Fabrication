@echo off
REM experiment_table.py 를 3가지 (num_jobs, machines) 세팅으로 순차 실행한 뒤,
REM 마지막에 experiment_pareto.py 실행. 각 단계는 성공/실패와 무관하게 계속 진행.

echo ===== [1/3] table  J=25  M=5,3,7,3,5,7 =====
python experiment_table.py --num_jobs 25 --machines 5,3,7,3,5,7
echo ----- done (exit %errorlevel%) -----
echo.

echo ===== [2/3] table  J=50  M=10,6,14,6,10,14 =====
python experiment_table.py --num_jobs 50 --machines 10,6,14,6,10,14
echo ----- done (exit %errorlevel%) -----
echo.

REM echo ===== [3/3] table  J=100  M=20,12,28,12,20,28 =====
REM python experiment_table.py --num_jobs 100 --machines 20,12,28,12,20,28
REM echo ----- done (exit %errorlevel%) -----
REM echo.

echo ===== pareto =====
python experiment_pareto.py
echo ----- done (exit %errorlevel%) -----
