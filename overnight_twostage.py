"""Run the four-model suite, then stop THIS Workbench via its managed API.

Run inside tmux. No IAM changes, keys, or cloud SDK installation are needed.
Permission/metadata checks happen BEFORE training. After training starts, both
success and failure trigger managed stop. A hung process is not a completed run.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
VARIANTS = ['resnet50_preact', 'botnet_twostage', 'coatnet_twostage', 'conformer_tiny']


def metadata(path):
    req = Request('http://metadata.google.internal/computeMetadata/v1/' + path,
                  headers={'Metadata-Flavor': 'Google'})
    with urlopen(req, timeout=10) as response:
        return response.read().decode()


def api(url, body=None):
    token = json.loads(metadata('instance/service-accounts/default/token'))['access_token']
    req = Request(url, data=None if body is None else json.dumps(body).encode(),
                  headers={'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'})
    with urlopen(req, timeout=30) as response:
        return json.load(response)


def preflight():
    project = metadata('project/project-id')
    zone = metadata('instance/zone').split('/')[-1]
    name = metadata('instance/name')
    url = f'https://notebooks.googleapis.com/v2/projects/{project}/locations/{zone}/instances/{name}'
    instance = api(url)  # Verify this is a Workbench resource and is readable.
    granted = api(f'https://cloudresourcemanager.googleapis.com/v1/projects/{project}:testIamPermissions',
                  {'permissions': ['notebooks.instances.stop']}).get('permissions', [])
    if 'notebooks.instances.stop' not in granted:
        raise RuntimeError('Attached VM service account lacks project-level notebooks.instances.stop. '
                           'No training started. Ask your administrator for a narrowly scoped stop permission; '
                           'this script does not change IAM. Instance-only grants are not accepted by this preflight.')
    print(f'Auto-stop target: {instance["name"]}; stop permission verified.', flush=True)
    return url


def stop_workbench(url, fallback=False):
    for attempt in range(3):
        try:
            state = api(url).get('state')
            if state in ('STOPPING', 'STOPPED'):
                print(f'Workbench state: {state}', flush=True)
                return
            operation = api(url + ':stop', {})
            if operation.get('error'):
                raise RuntimeError(str(operation['error']))
            print(f'Managed stop accepted: {operation.get("name", "operation")}', flush=True)
            # Usually the process disappears during this poll as the VM stops.
            for _ in range(18):
                time.sleep(10)
                operation = api('https://notebooks.googleapis.com/v2/' + operation['name'])
                if operation.get('error'):
                    raise RuntimeError(str(operation['error']))
                if api(url).get('state') == 'STOPPED':
                    return
            raise RuntimeError('Stop still pending after 180 seconds')
        except Exception as exc:
            print(f'Managed stop attempt {attempt+1} failed: {exc}', flush=True)
            time.sleep(5)
    if fallback:
        print('Falling back to OS shutdown to limit cost; Workbench UI may temporarily remain Active.', flush=True)
        subprocess.run(['sudo', '-n', '/usr/sbin/shutdown', '-h', 'now'], check=True)
    else:
        raise RuntimeError('Managed stop failed: stop the instance manually in Console.')


def training_command(args):
    command = [sys.executable, '-u', str(ROOT/'literature_experiment.py'), 'suite']
    if args.resume:
        return command + ['--resume', args.resume]
    return command + ['--variants', *VARIANTS, '--config', str(ROOT/'configs/tiny_imagenet_fusion.yaml'),
                      '--seeds', '42', '--epochs', str(args.epochs)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--resume', help='Printed suite directory; keeps its original schedule')
    parser.add_argument('--check-only', action='store_true', help='Check GPU and permissions; NO training or shutdown')
    parser.add_argument('--fallback-os-shutdown', action='store_true', help='Fallback if managed stop fails')
    parser.add_argument('--log', default='twostage_100ep_overnight.log')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('epochs must be positive')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing overnight CPU training')
    if args.fallback_os_shutdown:
        subprocess.run(['sudo', '-n', '-l', '/usr/sbin/shutdown', '-h', 'now'], check=True)
    url = preflight()
    if args.check_only:
        print('Preflight OK; no training or shutdown performed.', flush=True)
        return
    # Append rather than discard previous logs. Check log access before shutdown is armed.
    with open(args.log, 'a', encoding='utf-8', buffering=1) as log:
        code = 1
        try:
            command = training_command(args)
            print('Running: ' + ' '.join(command), flush=True)
            with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                for line in process.stdout:
                    log.write(line)
                    print(line, end='', flush=True)
                code = process.wait()
        finally:
            log.write(f'\nTraining exit code: {code}\n')
            log.flush()
            os.fsync(log.fileno())
            if hasattr(os, 'sync'):
                os.sync()
            stop_workbench(url, args.fallback_os_shutdown)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
