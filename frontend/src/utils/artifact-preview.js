export function artifactRoute(artifact) {
  const uuid = '[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}'
  if (!new RegExp(`^/artifacts/${uuid}/${uuid}$`).test(artifact?.download_url || '')) throw new Error('Invalid artifact route')
  return artifact.download_url
}

// Preview documents live in an opaque, scriptless iframe, never in the app DOM.
export function previewDocument(content) {
  const doc = new DOMParser().parseFromString(content, 'text/html')
  doc.querySelectorAll('script,iframe,object,embed,base,meta,link,form,input,button,textarea,select').forEach(el => el.remove())
  doc.querySelectorAll('*').forEach(el => {
    for (const attr of [...el.attributes]) {
      if (attr.name.startsWith('on') || ['srcdoc', 'action', 'formaction', 'target', 'srcset', 'ping'].includes(attr.name)
        || (attr.name === 'href' && !attr.value.startsWith('#'))
        || (attr.name === 'src' && !/^data:image\/(png|jpeg|gif|webp);base64,/i.test(attr.value))) el.removeAttribute(attr.name)
    }
  })
  const policy = "default-src 'none'; style-src 'unsafe-inline'; img-src data:; font-src 'none'; form-action 'none'; base-uri 'none'"
  return `<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="${policy}"><style>
    body{margin:32px;color:#262626;font:15px/1.75 system-ui,sans-serif;overflow-wrap:anywhere}
    table{border-collapse:collapse;max-width:100%;margin:16px 0}td,th{border:1px solid #ddd;padding:8px 12px;white-space:pre-wrap}
    th{background:#f5f6f7}pre{white-space:pre-wrap;padding:16px;background:#f6f6f6;border-radius:8px}img{max-width:100%}
    h1,h2,h3{line-height:1.4}blockquote{border-left:3px solid #ddd;margin-left:0;padding-left:16px;color:#666}
    @media(max-width:600px){body{margin:16px}}
  </style>${[...doc.head.querySelectorAll('style')].map(el => el.outerHTML).join('')}</head><body>${doc.body.innerHTML}</body></html>`
}
