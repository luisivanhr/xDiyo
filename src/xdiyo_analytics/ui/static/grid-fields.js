// Tunable scalar controls from the same inventory that renders each component.
export function tunableFields(recipe, spec) {
  const fields=[];
  function visit(node,path,label) {
    if(!node || !node.component) return;
    for(const field of spec(node.component)?.fields || []) {
      const key=path?`${path}.params.${field.name}`:field.name;
      if(!field.hidden && ['number','boolean','text','select'].includes(field.kind)
          && !['type','partition'].includes(field.name)) {
        fields.push({...field,name:key,title:`${label} · ${field.title}`});
      }
      const child=node.params?.[field.name];
      if(child?.component) visit(child,`${path || 'model'}.params.${field.name}`,`${label} · ${field.title}`);
    }
  }
  visit(recipe.model,'','Model');
  for(const [name,node] of Object.entries(recipe.fitted_reporters || {})) visit(node,`fitted_reporters.${name}`,name);
  for(const [i,node] of (recipe.preprocessors || []).entries()) visit(node,`preprocessors.${i}`,`Preprocessor ${i+1}`);
  for(const [name,node] of Object.entries(recipe.candidate || {})) if(node?.component) visit(node,`candidate.${name}`,name);
  if(recipe.target_transformer) visit(recipe.target_transformer,'target_transformer','Target transform');
  return fields;
}
