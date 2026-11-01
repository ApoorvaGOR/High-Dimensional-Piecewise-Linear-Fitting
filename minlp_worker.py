"""Isolated MINLP execution so solver-process failures can be recorded per case."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from minlp_benchmark import solve_minlp


def run_isolated(input_path,bounds_path,result_path,solution_path,log_path,k,
                 time_limit=2000.0,threads=1,seed=0,console_log=False):
    result_path=Path(result_path)
    result_path.parent.mkdir(parents=True,exist_ok=True)
    result_path.unlink(missing_ok=True)
    output_path=result_path.with_suffix('.process.log')
    command=[sys.executable,str(Path(__file__).resolve()),'--input',str(input_path),
             '--bounds',str(bounds_path),'--result',str(result_path),'--solution',str(solution_path),
             '--log',str(log_path),'--k',str(k),'--time-limit',str(time_limit),
             '--threads',str(threads),'--seed',str(seed)]
    if console_log:
        command.append('--console-log')
    started=time.perf_counter()
    env=os.environ.copy()
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
        env[name]=str(threads)
    with output_path.open('w') as output:
        completed=subprocess.run(command,stdout=None if console_log else output,
                                 stderr=subprocess.STDOUT,env=env,check=False)
    if completed.returncode or not result_path.exists():
        raise RuntimeError(f'MINLP subprocess exit={completed.returncode}; see {output_path}')
    result=json.loads(result_path.read_text())
    if 'error' in result:
        raise RuntimeError(result['error'])
    result['process_wall_seconds']=time.perf_counter()-started
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['input','bounds','result','solution','log']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--k',type=int,required=True)
    p.add_argument('--time-limit',type=float,default=2000)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--console-log',action='store_true')
    args=p.parse_args()
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=args.threads)
        with np.load(args.input,allow_pickle=False) as source:
            x,Y,names=source['x'],source['Y'],source['names']
        with np.load(args.bounds,allow_pickle=False) as source:
            bounds={key:source[key] for key in source.files}
        result=solve_minlp(x,Y,k=args.k,bounds=bounds,time_limit=args.time_limit,
                           threads=args.threads,random_seed=args.seed,output_flag=int(args.console_log),
                           solution_path=args.solution,log_path=args.log,names=names)
    except Exception as exc:
        result={'error':f'{type(exc).__name__}: {exc}'}
    temp=args.result.with_suffix('.tmp.json')
    temp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    os.replace(temp,args.result)


if __name__=='__main__':
    main()
