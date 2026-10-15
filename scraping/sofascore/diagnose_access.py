"""One bounded, accounted comparison of an inaccessible resource and controls."""
import json
from pathlib import Path
import time

from .client import BrowserClient
from .config import RatePolicy
from .models import Job
from .rate_limit import RateLimiter, retry_after_seconds
from .store import Store, atomic_write, json_bytes


def main():
    root=Path(__file__).resolve().parents[2]
    policy=RatePolicy(min_interval_seconds=None,per_minute=None,per_hour=None,max_attempts_per_run=3)
    driver=root/'data/collection_state/browser_cache/chromedriver/win64/152.0.7977.82/chromedriver.exe'
    report=[]
    with Store(root/'data/collection_state') as store,store.run_lock():
        original=store.setting('access_block')
        if not original or original.get('status')!=403:
            raise RuntimeError('Expected the inspected provider403 latch')
        run_id=store.start_run()
        limiter=RateLimiter(store,policy,run_id)
        client=BrowserClient(policy,driver_path=driver)
        try:
            # Collector was stopped before this bounded diagnostic. Restore the
            # original latch in finally; this command cannot resume collection.
            store.set_setting('access_block',None)
            for eid,component in ((12499520,'shotmap'),(12499520,'statistics'),(12499548,'shotmap')):
                job=Job(f'/event/{eid}/{component}',component,eid)
                attempt=limiter.reserve(job)
                response=client.fetch(job)
                limiter.complete(attempt,response.status,'access_diagnostic')
                record={'path':job.path,'status':response.status,'error':response.error,
                        'headers':response.headers,'extra_api_requests':response.additional_api_requests}
                if response.status!=200:
                    record['visible_body']=client.driver.find_element('tag name','body').text[:4000]
                report.append(record)
                delay=retry_after_seconds(response.headers.get('retry-after'),time.time())
                if delay is not None:
                    limiter.cooldown(delay)
                if response.additional_api_requests:
                    raise RuntimeError('Unexpected diagnostic browser traffic')
                if response.status==429 or (delay is not None and delay>0):
                    break
        finally:
            client.close()
            store.set_setting('access_block',original)
            store.finish_run(run_id,'diagnostic',{'responses':report})
            atomic_write(root/'data/xDiyo_data/_collection/access_diagnostic.json',json_bytes(report))
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
