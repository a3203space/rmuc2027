"""Public LAN hosting and local replay entry points."""
import argparse, json
from pathlib import Path
from rmuc2027_core.paths import ROOT

def main():
    parser = argparse.ArgumentParser(prog='rmuc2027', description='RMUC 2027 LAN play')
    sub = parser.add_subparsers(dest='command', required=True)
    host = sub.add_parser('host')
    host.add_argument('--host', default='0.0.0.0')
    host.add_argument('--port', type=int, default=8766)
    host.add_argument('--parent-stdin', action='store_true')
    host.add_argument('--replay-dir', type=Path)
    verify = sub.add_parser('verify-replay'); verify.add_argument('replay', type=Path)
    replay = sub.add_parser('replay'); replay.add_argument('replay', type=Path); replay.add_argument('--port', type=int, default=8767)
    args = parser.parse_args()
    if args.command == 'host':
        from .server import serve
        serve(args.host, args.port, parent_stdin=args.parent_stdin, directory=args.replay_dir)
    elif args.command == 'verify-replay':
        from rmuc2027_core.replay import verify_replay
        print(json.dumps(verify_replay(args.replay), indent=2))
    else:
        from .viewer import create_replay_app
        import uvicorn
        app = create_replay_app(args.replay)
        print(f'Replay viewer: http://127.0.0.1:{args.port}', flush=True)
        uvicorn.run(app, host='127.0.0.1', port=args.port)
