"""Finite sequential local jobs; no deployment, git writes or notifications."""
import argparse,datetime,hashlib,json,os,subprocess,time
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
    except FileNotFoundError:return False

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);a=p.parse_args();plan=json.loads(a.plan.read_text());root=Path(plan['root']);state=root/'queue_status.json'
    def save(**fields):state.write_text(json.dumps(dict(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**fields),indent=2))
    completed=[];save(status='waiting_for_hold_training',completed=completed,hold_pid=plan['hold_pid']);deadline=time.monotonic()+3600
    try:
        while alive(plan['hold_pid']):
            if time.monotonic()>deadline:raise RuntimeError('First training exceeded wait budget; not starting another GPU job')
            time.sleep(2)
        hold_complete=root/'train_hold/complete.json'
        if not hold_complete.exists():raise RuntimeError('Hold training exited without complete.json')
        for job in plan['jobs']:
            for path,digest in plan['source_hashes'].items():
                if sha(Path(path))!=digest:raise RuntimeError('Queued source changed '+path)
            command=[]
            for token in job['command']:
                if token=='@hold_checkpoint':token=json.loads((root/'train_hold/complete.json').read_text())['final_checkpoint']
                elif token=='@march_checkpoint':token=json.loads((root/'train_march/complete.json').read_text())['final_checkpoint']
                command.append(token)
            log=root/'queue_logs'/f"{job['name']}.log";log.parent.mkdir(exist_ok=True);save(status='running',current_job=job['name'],completed=completed,log=str(log))
            with log.open('w') as stream:
                proc=subprocess.Popen(command,cwd=plan['cwd'],stdout=stream,stderr=subprocess.STDOUT,env={**os.environ,'OPENBLAS_NUM_THREADS':'1'})
                save(status='running',current_job=job['name'],pid=proc.pid,completed=completed,log=str(log))
                try:code=proc.wait(timeout=job['timeout_s'])
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    try:proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
                    raise RuntimeError('Job timed out '+job['name'])
            artifact=Path(job['required_artifact'])
            if code!=0 or not artifact.exists():raise RuntimeError(f"Job failed {job['name']} code={code}, artifact={artifact.exists()}, log={log}")
            completed.append(dict(name=job['name'],log=str(log),artifact=str(artifact),artifact_sha256=sha(artifact)))
        save(status='completed',completed=completed,note='Training/export/evaluation pipeline completed; inspect evaluation pass counts before any skill promotion. No watchdog registration or git commit/push.')
    except Exception as exc:
        save(status='failed',completed=completed,error=str(exc));raise

if __name__=='__main__':main()
