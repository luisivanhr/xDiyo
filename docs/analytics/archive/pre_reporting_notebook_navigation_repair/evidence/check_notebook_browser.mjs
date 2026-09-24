import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createRequire} from 'node:module';
const require = createRequire('C:/Users/luisi/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/review.cjs');
const {chromium} = require('playwright');
const scratch = path.join(process.cwd(), '.pytest_tmp/reporting_verification/jupyter_frontend');
const base = process.env.REPORTING_JUPYTER_URL;
const token = process.env.REPORTING_JUPYTER_TOKEN;
const exported = process.argv.includes('--export');
const checks = [], errors = [], remote = [];
const check = (name, passed, detail=null) => checks.push({name, passed, detail});
const browser = await chromium.launch({headless:true, executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe'});
const page = await browser.newPage({viewport:{width:1550,height:1150}});
page.on('pageerror', error=>errors.push(error.message));
await page.route(/https?:\/\//, route=>{
  if (!exported && new URL(route.request().url()).origin === base) return route.continue();
  remote.push(route.request().url());
  return route.abort();
});
try {
  await page.goto(exported ? pathToFileURL(path.join(scratch,'saved_notebook.html')).href : base + '/lab/tree/notebooks/08_reporting_quickstart.ipynb?token=' + encodeURIComponent(token));
  await page.locator('.jp-Notebook').waitFor({timeout:40000});
  check(exported ? 'jupyter_lab_template_export' : 'actual_jupyter_notebook_frontend', await page.locator(exported ? '.jp-Notebook' : '.jp-NotebookPanel').count() === 1);
  for (let attempt=0;attempt<5 && !(await page.locator('iframe[srcdoc]').count());attempt++) {
    await page.locator('.jp-Notebook').evaluate(el=>{el.scrollTop=el.scrollHeight;});
    await page.waitForTimeout(600);
  }
  const iframe = page.locator('iframe[srcdoc]');
  await iframe.waitFor({timeout:25000});
  await iframe.scrollIntoViewIfNeeded();
  const report = page.frameLocator('iframe[srcdoc]');
  await report.locator('details.study').first().waitFor();
  check('saved_native_srcdoc_iframe', (await iframe.getAttribute('src')) === 'about:blank');
  check('three_named_studies', await report.locator('details.study').count() === 3);
  const firstScope = await report.locator('#study-0 .variant:not([hidden]) .scope').innerText();
  await report.locator('#study-0 .fold-selector').selectOption('1');
  const secondScope = await report.locator('#study-0 .variant:not([hidden]) .scope').innerText();
  check('saved_fold_switch', firstScope !== secondScope && secondScope.includes('train'), {firstScope, secondScope});
  check('saved_association_table_and_bars', await report.locator('#study-0 .variant:not([hidden]) table').count() > 0 && await report.locator('#study-0 .variant:not([hidden]) .bar').count() > 0);
  await page.screenshot({path:path.join(scratch, 'jupyter_associations.png')});
  await report.locator('nav a[href="#study-2"]').click();
  const plot = report.locator('#study-2 .variant:not([hidden]) .js-plotly-plot').first();
  await plot.waitFor();
  check('saved_histogram_and_kde_run', await plot.evaluate(el=>el.data.some(t=>t.type==='bar') && el.data.some(t=>t.type==='scatter')));
  await report.locator('#study-2 .fold-selector').selectOption('1');
  await report.locator('#study-2 .variant[data-fold="1"] .js-plotly-plot').first().waitFor();
  check('second_test_fold_plot', (await report.locator('#study-2 .variant:not([hidden]) .scope').innerText()).includes('test'));
  await page.screenshot({path:path.join(scratch, 'jupyter_distribution.png')});
  check('no_notebook_execution_errors', await page.locator('.jp-OutputArea-output[data-mime-type="application/vnd.jupyter.stderr"]').count() === 0);
} catch(error) {
  await page.screenshot({path:path.join(scratch, 'jupyter_failure.png')});
  fs.writeFileSync(path.join(scratch, 'failure_dom.txt'), (await page.locator('body').innerText()).replaceAll(token ?? 'TOKEN_NOT_PRESENT','[redacted]'));
  throw error;
} finally {
  const result={checked_at_utc:new Date().toISOString(),mode:exported?'saved_nbconvert_lab_export':'live_jupyterlab',checks,page_errors:errors,remote_requests:remote};
  fs.writeFileSync(path.join(scratch,'browser.json'),JSON.stringify(result,null,2)+'\n');
  await browser.close();
}
if (checks.some(c=>!c.passed) || errors.length || remote.length) throw Error('Jupyter frontend checks failed; inspect browser.json.');
console.log(JSON.stringify({checks:checks.length,passed:checks.filter(c=>c.passed).length,errors,remote_requests:remote.length}));
