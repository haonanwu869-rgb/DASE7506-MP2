"""Official evaluation in two fresh processes. Defaults to validation, never test."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'
import argparse, json, subprocess, sys, time
from pathlib import Path

def main():
    code = Path(__file__).resolve().parent; root = code.parent
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--split', choices=['validation','test','audit'], default='validation')
    p.add_argument('--checkpoint', type=Path, default=root/'model')
    p.add_argument('--output-dir', type=Path, default=root/'evaluation')
    args = p.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    pred = args.output_dir/f'{args.split}-predictions.json'
    score = args.output_dir/f'{args.split}-score.json'
    start = time.perf_counter()
    subprocess.run([sys.executable,str(code/'predict.py'),'--method',str(code/'submission.py'),
        '--checkpoint',str(args.checkpoint.resolve()),'--data-dir',str(args.data_dir.resolve()),
        '--split',args.split,'--output',str(pred.resolve())], check=True)
    prediction_wall = time.perf_counter()-start
    tick = time.perf_counter()
    subprocess.run([sys.executable,str(code/'score_predictions.py'),'--data-dir',str(args.data_dir.resolve()),
        '--split',args.split,'--predictions',str(pred.resolve()),'--output',str(score.resolve())], check=True)
    scoring_wall = time.perf_counter()-tick
    timing = {'split':args.split,'threads':4,'prediction_process_seconds':prediction_wall,
        'scoring_process_seconds':scoring_wall,'total_seconds':prediction_wall+scoring_wall}
    (args.output_dir/f'{args.split}-timing.json').write_text(json.dumps(timing, indent=2)+'\n')
    print(json.dumps(timing, indent=2))

if __name__ == '__main__': main()
