"""Offline policy development, matches, and RL; never imported by the LAN host."""
import argparse, json, sys
from pathlib import Path

def main():
    if len(sys.argv)>1 and sys.argv[1]=='rl':
        del sys.argv[1]
        from .rl.__main__ import main as rl_main
        return rl_main()
    parser=argparse.ArgumentParser(prog='rmuc2027-research')
    sub=parser.add_subparsers(dest='command',required=True)
    battle=sub.add_parser('battle')
    battle.add_argument('--red',type=Path,required=True);battle.add_argument('--blue',type=Path,required=True)
    battle.add_argument('--seed',type=int,default=42);battle.add_argument('--duration',type=float,default=420)
    battle.add_argument('--sandbox',choices=['trusted-local','docker'],default='trusted-local')
    battle.add_argument('--output',type=Path,required=True)
    new=sub.add_parser('new-strategy');new.add_argument('output',type=Path)
    save=sub.add_parser('bundle');save.add_argument('source',type=Path);save.add_argument('output',type=Path)
    migrate=sub.add_parser('migrate-model');migrate.add_argument('source',type=Path);migrate.add_argument('output',type=Path)
    args=parser.parse_args()
    if args.command=='battle':
        from .strategies.battle import Battle
        if args.output.exists():parser.error('Output exists; choose a new replay path')
        with Battle(args.red,args.blue,seed=args.seed,duration=args.duration,map_name='rmuc2027',sandbox=args.sandbox) as match:
            result=match.run();match.save(args.output)
        print(json.dumps(result,indent=2))
    elif args.command=='migrate-model':
        from .migrate import migrate_model
        print(json.dumps(migrate_model(args.source,args.output),indent=2))
    elif args.command=='new-strategy':
        args.output.mkdir(parents=True,exist_ok=False)
        (args.output/'strategy.py').write_text('from sdk import action\ndef act(observation, state):\n    return {str(slot): action(move=[0,0], fire=False) for slot in observation["controlled_ids"]}, state\n')
    else:
        from rmuc2027_core.submissions import save_submission
        print(json.dumps(save_submission(args.output,'team',{'team':args.source},season='2027'),indent=2))
