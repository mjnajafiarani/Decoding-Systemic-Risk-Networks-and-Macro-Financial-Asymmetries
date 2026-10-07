from pathlib import Path
import subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
PYTHON=sys.executable
TARGETS=['CDS','EXC','INF','TPD','INT','STC','CAB','EID','GRI','INV']
for t in TARGETS:
    subprocess.run([PYTHON, str(ROOT/'scripts'/'run_primary_target.py'), t], check=True)
subprocess.run([PYTHON, str(ROOT/'scripts'/'aggregate_primary.py')], check=True)
subprocess.run([PYTHON, str(ROOT/'scripts'/'run_joint_ablation.py')], check=True)
for t in TARGETS:
    subprocess.run([PYTHON, str(ROOT/'scripts'/'run_difference_target.py'), t], check=True)
subprocess.run([PYTHON, str(ROOT/'scripts'/'aggregate_difference.py')], check=True)
print('All analyses completed.')
