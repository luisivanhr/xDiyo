// Bundle creation produces ordinary recipe expressions, with no runtime bundle type.
export function statSources(node) {
  if (!node || typeof node !== 'object') return [];
  if (node.component === 'features.Stat') return [node];
  return Object.values(node).flatMap(statSources);
}

export function bundleStatistics(template, statistics) {
  const sources = statSources(template);
  if (sources.length !== 1) return [];
  const period = sources[0].params.period;
  const matching = statistics.filter(s => period == null || s.period === period);
  return [...new Map(matching.map(s => [JSON.stringify([s.group_name, s.key]), s])).values()];
}

const slug = value => String(value).replace(/([a-z0-9])([A-Z])/g, '$1_$2')
  .replace(/[^a-zA-Z0-9]+/g, '_').replace(/^_|_$/g, '').toLowerCase();

export function expandFeatureBundle(template, statistics, existingNames = [], components = []) {
  if (statSources(template).length !== 1) throw Error('Choose a computation with exactly one Stat source.');
  const used = new Set(existingNames);
  const selected = new Set();
  return statistics.map(stat => {
    const identity = JSON.stringify([stat.group_name, stat.key]);
    if (selected.has(identity)) throw Error('Select each statistic once.');
    selected.add(identity);
    const expression = structuredClone(template);
    const source = statSources(expression)[0];
    Object.assign(source.params, {group: stat.group_name, key: stat.key});
    const operators = [], windows = [];
    function describe(node) {
      if (!node || typeof node !== 'object') return;
      if (node.component && node.component !== 'features.Stat') {
        for (const field of components.find(c => c.id === node.component)?.fields || []) {
          if (['window', 'periods', 'span', 'alpha', 'halflife'].includes(field.name)
              && !Object.hasOwn(node.params, field.name) && field.default != null)
            node.params[field.name] = structuredClone(field.default);
        }
        operators.push(slug(node.component.split('.').pop()));
        for (const [key, value] of Object.entries(node.params || {})) {
          if (['window', 'periods', 'span', 'alpha', 'halflife'].includes(key) && value != null)
            windows.push(slug(key) + '_' + slug(value));
        }
      }
      Object.values(node).forEach(describe);
    }
    describe(expression);
    const period = source.params.period;
    const base = [...(operators.length ? operators : ['raw_stat']), slug(stat.key), ...windows,
      ...(period === 'ALL' ? [] : [period == null ? 'all_periods' : slug(period)])].join('_');
    let name = base, suffix = 2;
    while (used.has(name)) name = base + '_' + suffix++;
    used.add(name);
    return {name, expression};
  });
}
