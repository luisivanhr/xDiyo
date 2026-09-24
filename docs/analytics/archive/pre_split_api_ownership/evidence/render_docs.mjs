import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {createRequire} from 'node:module';

const root = process.cwd();
const scratch = path.join(root, '.pytest_tmp/splits_verification');
const archive = path.join(root, 'docs/analytics/archive/pre_equation_review');
const assets = 'C:/Users/luisi/.vscode/extensions/openai.chatgpt-26.908.40401-win32-x64/webview/assets';
const packages = 'C:/Users/luisi/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const require = createRequire(path.join(packages, 'review.cjs'));
const {marked} = require('marked');
const sha = data => crypto.createHash('sha256').update(data).digest('hex');
const read = rel => fs.readFileSync(path.join(root, rel), 'utf8');
const hash = rel => sha(fs.readFileSync(path.join(root, rel)));
const guides = ['docs/analytics/splits.md', 'docs/analytics/splits_reference.md', 'docs/analytics/splits_documentation_checklist.md'];
const targets = [...guides, 'docs/analytics/league_warmup_documentation_checklist.md', 'IMPLEMENTATION_PROGRESS.md'];
const blocks = text => [...text.matchAll(/^```python\r?\n([\s\S]*?)^```/gm)].map(m => sha(m[1].replace(/\r\n/g, '\n')));
const walk = rel => fs.readdirSync(path.join(root, rel), {withFileTypes:true}).flatMap(e => {
  const name = rel + '/' + e.name;
  return e.isDirectory() ? (e.name === '__pycache__' ? [] : walk(name)) : [name];
});
fs.mkdirSync(scratch, {recursive:true});
const katexSource = fs.readFileSync(path.join(assets, 'katex-ce1e5fbd580e.js'),'utf8');
const {default:katex} = await import('data:text/javascript;base64,'+Buffer.from(katexSource).toString('base64'));
const report = {checked_at_utc:new Date().toISOString(), katex_version:katex.version, documents:[], errors:[]};
const escapeHtml = s=>s.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
let page = '';
for (const rel of guides) {
  const text = read(rel);
  const equations = [];
  const protectedCode = [];
  let body = text.replace(/^(`{3,})[^\n]*\n[\s\S]*?^\1\s*$/gm,m=>{protectedCode.push(m);return `CODEBLOCKTOKEN${protectedCode.length-1}END`;});
  body = body.replace(/\\\[([\s\S]*?)\\\]|\\\(([\s\S]*?)\\\)/g,(whole,display,inline,offset)=>{
    const latex = display ?? inline;
    const id = 'eq-'+report.documents.length+'-'+equations.length;
    const eq={id,latex,display:display!==undefined,line:text.slice(0,text.indexOf(whole)).split('\n').length};
    try {eq.html=katex.renderToString(latex,{displayMode:eq.display,throwOnError:true,strict:'error',output:'htmlAndMathml'});}
    catch(error){eq.error=error.message;report.errors.push({file:rel,line:eq.line,latex,message:error.message});eq.html='<strong class="math-error">'+escapeHtml(error.message)+'</strong>';}
    equations.push(eq);
    return `MATHTOKEN${equations.length-1}END`;
  });
  body = body.replace(/CODEBLOCKTOKEN(\d+)END/g,(_,i)=>protectedCode[+i]);
  let rendered = marked.parse(body,{gfm:true});
  rendered = rendered.replace(/MATHTOKEN(\d+)END/g,(_,i)=>`<span class="equation${equations[+i].display?' display':''}" id="${equations[+i].id}">${equations[+i].html}</span>`);
  page += `<article data-guide="${rel}"><p class="file-name">${rel}</p>${rendered}</article>`;
  report.documents.push({file:rel,sha256:hash(rel),equation_count:equations.length,display_count:equations.filter(e=>e.display).length,equations:equations.map(({html,...eq})=>eq)});
}
const cssFiles = fs.readdirSync(assets).filter(x=>x.endsWith('.css')&&fs.readFileSync(path.join(assets,x),'utf8').includes('.katex{'));
if (!cssFiles.length) throw Error('Local KaTeX CSS not found.');
const css = cssFiles.map(name=>{
  const source = fs.readFileSync(path.join(assets,name),'utf8');
  // Isolate KaTeX stylesheet rules from application CSS, retaining font URLs.
  const start=source.indexOf('@font-face{font-family:KaTeX_');
  const end=source.indexOf('.katex .fallback');
  return {name,start,end,source};
});
// Full styles remain local; review-specific layout rules override application defaults.
const html='<!doctype html><html><head><meta charset="utf-8">'+cssFiles.map(name=>`<link rel="stylesheet" href="${pathToFileURL(path.join(assets,name)).href}">`).join('')+'<style>html,body{margin:0;background:white;color:#15212b;font:16px/1.55 Segoe UI,sans-serif}body{padding:32px}article{max-width:1000px;margin:0 auto 80px}h1{font-size:28px}h2{font-size:24px;margin-top:32px}h3{font-size:20px;margin-top:24px}p{margin:12px 0}pre{background:#f4f6f8;padding:12px;overflow:auto}code{font:14px Consolas,monospace}table{border-collapse:collapse;width:100%;margin:16px 0}th,td{border:1px solid #bcc5cf;padding:8px;text-align:left}li{margin:5px 0}ul,ol{padding-left:24px}ul{list-style:disc}ol{list-style:decimal}a{color:#1653a3}.display{display:block;margin:18px 0}.katex{font-size:1.05em}.katex-display{overflow:visible}.math-error{color:#b00020}.file-name{color:#5c6570}</style></head><body>'+page+'</body></html>';
const suffix=process.argv.includes('--after')?'after':'before';
fs.writeFileSync(path.join(scratch,`render_${suffix}.html`),html);
fs.writeFileSync(path.join(scratch,`render_${suffix}.json`),JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify({phase:suffix,katex_version:report.katex_version,equations:report.documents.map(d=>({file:d.file,count:d.equation_count,display:d.display_count})),errors:report.errors,cssFiles},null,2));
if (process.argv.includes('--screenshot')) {
  const {chromium}=require('playwright');
  const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe'});
  try {
    const browserPage=await browser.newPage({viewport:{width:1200,height:950},deviceScaleFactor:1});
    await browserPage.route(/https?:\/\//,route=>route.abort());
    await browserPage.goto(pathToFileURL(path.join(scratch,`render_${suffix}.html`)).href);
    await browserPage.evaluate(()=>document.fonts.ready);
    report.layout=await browserPage.locator('.equation').evaluateAll(nodes=>nodes.filter(e=>e.classList.contains('display')).map(e=>({id:e.id,width:e.getBoundingClientRect().width,content_width:e.scrollWidth,overflow:e.scrollWidth>e.clientWidth+2})));
    report.rendered_math_count=await browserPage.locator('.katex').count();
    report.rendered_error_count=await browserPage.locator('.math-error,.katex-error').count();
    report.table_rows=await browserPage.locator('table').evaluateAll(tables=>tables.map(t=>({headers:[...t.querySelectorAll('th')].map(x=>x.innerText),rows:[...t.querySelectorAll('tbody tr')].map(tr=>tr.children.length)})));
    const visualSections=['Separate held-out membership from scoring','State prediction and availability times','Choose calendars deliberately','Retrospective match and group K-fold','Combinatorial purging and embargo','Reconstruct held-out values without fitting','Connect a future fitting layer','Fold and SplitPlan records','TemporalSplit settings and eligibility','Temporal fold metadata','CPCV configuration, intervals and paths','Prediction path contract','Complete public API coverage'];
    for (const [index,heading] of visualSections.entries()) {
      await browserPage.getByRole('heading',{name:heading,exact:true}).scrollIntoViewIfNeeded();
      await browserPage.getByRole('heading',{name:heading,exact:true}).evaluate(e=>window.scrollTo(0,e.getBoundingClientRect().top+window.scrollY-24));
      await browserPage.evaluate(async()=>{await document.fonts.ready;await new Promise(requestAnimationFrame);});
      await browserPage.screenshot({path:path.join(scratch,`render_${suffix}_${index+1}.png`)});
    }
    fs.writeFileSync(path.join(scratch,`render_${suffix}.json`),JSON.stringify(report,null,2)+'\n');
    console.log(JSON.stringify({rendered_math:report.rendered_math_count,rendered_errors:report.rendered_error_count,overflow:report.layout.filter(e=>e.overflow),screenshots:visualSections.length},null,2));
  } finally {await browser.close();}
}
