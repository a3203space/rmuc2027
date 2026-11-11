"""Serial fixed-policy validation matrix; no PPO training is performed."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[7,42,1001])
    parser.add_argument('--configs',nargs='+',default=[
        'configs/rl/sentry-mjwarp-dual-tunnel.json','configs/rl/sentry-mjwarp-mixed.json',
        'configs/rl/sentry-mjwarp-all-omni.json'])
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    results=[]
    for config in args.configs:
        for seed in args.seeds:
            case=args.output/f'{Path(config).stem}-{seed}'
            command=[sys.executable,'-m','rmuc2027_research.rl.warp_kinematic.benchmark',
                     '--config',config,'--seed',str(seed),'--fixed-policy-hz','10','--output',str(case)]
            with case.with_suffix('.log').open('w') as stream:
                process=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
            row=dict(config=config,seed=seed,exit_code=process.returncode,output=str(case))
            if process.returncode==0:
                summary=json.loads((case/'summary.json').read_text())
                audit=json.loads((case/'motion-audit.json').read_text())
                row.update(realtime_factor=summary['realtime_factor'],sim_seconds=summary['sim_seconds'],
                           policy_errors=summary['result']['policy_errors'],motion_findings=audit['findings'])
                row['basic_gates_pass']=(row['sim_seconds']==420 and row['realtime_factor']>=5
                                        and row['policy_errors']==0 and not row['motion_findings'])
            else:row['basic_gates_pass']=False
            results.append(row)
            report=dict(completed=len(results),total=len(args.configs)*len(args.seeds),results=results,
                        limitation='Fixed-policy speed and telemetry only; does not prove transfer, all navigation or PPO throughput')
            temporary=args.output/'progress.tmp'
            temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2))
            temporary.replace(args.output/'progress.json')
            print(json.dumps(row,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
