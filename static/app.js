import { $, button, chart, element, field, format, keyValues, link, table, whole } from './widgets.js'

const VIEW = {
  overview: ['Station overview', 'Corrected reflectance aligned to encoder station', 'station', 'grid'],
  sensor: ['Raw scan', 'Acquisition order with encoder timing and transport observations', 'sensor'],
  flat: ['Flat field', 'Reference-corrected reflectance in acquisition order', 'flat', 'calibration'],
  coverage: ['Coverage', 'Native samples, interpolated rows and separate evidence gaps', 'coverage', 'grid'],
  edges: ['Strip geometry', 'Edge traces and measured material width', 'edges', 'edges'],
  spots: ['Surface spots', 'Candidate footprints against the local material background', 'spots', 'spots'],
  streaks: ['Streaks', 'Longitudinal tracks with sustained local contrast', 'streaks', 'streaks'],
  marks: ['Repeat pitch', 'Dark repeat marks and measured station spacing', 'marks', 'marks'],
  splices: ['Transverse joins', 'Same-sign contrast across the material width', 'splices', 'splices'],
  texture: ['Texture', 'Stripe direction estimated inside supported material tiles', 'texture', 'texture'],
  illumination: ['Reflectance trend', 'Across-strip and along-roll material reflectance', 'illumination'],
  windows: ['Review windows', 'Saved bounds in physical station coordinates', 'station'],
  transverse: ['Transverse calibration', 'Cross-web ruler mapping with evidence grades and a reliable span', 'sensor', 'transverse'],
  ruler: ['Ruler', 'Measure across sheet, bad pixels and acquisition gaps', 'station'],
  snapshots: ['Saved reports', 'Previously recorded measurements and settings', 'station'],
}

const SETTINGS = {
  calibration: [
    ['minimum_response', 'Minimum response (ADU)', '1'],
    ['maximum_reference_noise', 'Maximum reference variation (ADU)'],
    ['saturation_code', 'Saturation code (ADU)', '1'],
  ],
  transverse: [
    ['minimum_prominence', 'Minimum ruler line prominence'],
  ],
  grid: [
    ['pitch_mm', 'Station pitch (mm)'],
    ['gap_limit_mm', 'Interpolation gap limit (mm)'],
  ],
  edges: [
    ['threshold', 'Edge reflectance threshold'],
    ['minimum_width_mm', 'Minimum material width (mm)'],
    ['bridge_mm', 'Bridge over blind columns (mm)'],
    ['continuity_mm', 'Continuity tolerance (mm)'],
  ],
  spots: [
    ['contrast', 'Minimum spot contrast'],
    ['background_mm', 'Background scale (mm)'],
    ['minimum_area_mm2', 'Minimum spot area (mm²)'],
    ['edge_guard_mm', 'Edge guard (mm)'],
  ],
  streaks: [
    ['contrast', 'Minimum streak contrast'],
    ['background_mm', 'Background scale (mm)'],
    ['minimum_length_mm', 'Minimum streak length (mm)'],
    ['minimum_coverage', 'Minimum evidence fraction'],
  ],
  marks: [
    ['expected_spacing_mm', 'Expected mark spacing (mm)'],
    ['minimum_prominence', 'Minimum prominence'],
    ['band_start_fraction', 'Band start (strip width fraction)'],
    ['band_end_fraction', 'Band end (strip width fraction)'],
  ],
  texture: [
    ['tile_size', 'Tile width and height (samples)', '1'],
    ['overlap_fraction', 'Tile overlap fraction'],
    ['minimum_wave_mm', 'Shortest wavelength (mm)'],
    ['maximum_wave_mm', 'Longest wavelength (mm)'],
  ],
  splices: [
    ['contrast', 'Minimum join contrast'],
    ['minimum_fraction', 'Minimum width fraction'],
    ['local_scale_mm', 'Local scale (mm)'],
    ['minimum_spacing_mm', 'Minimum spacing (mm)'],
  ],
}

const LEGEND = {
  overview: [['reflectance', 'Reflectance'], ['gap', 'Encoder gap'], ['bad', 'Invalid sample']],
  sensor: [['reflectance', 'Raw sensor codes']],
  flat: [['reflectance', 'Reflectance'], ['bad', 'Rejected column or pixel']],
  coverage: [['native', 'Native'], ['blend', 'Interpolated'], ['gap', 'Encoder gap'], ['bad', 'Invalid sample']],
  edges: [['edge', 'Tracked edge'], ['gap', 'Encoder gap'], ['bad', 'Invalid sample']],
  spots: [['dark', 'Darker contrast'], ['bright', 'Brighter contrast'], ['gap', 'Encoder gap'], ['bad', 'Excluded area']],
  streaks: [['dark', 'Dark streak'], ['bright', 'Bright streak'], ['gap', 'Encoder gap'], ['bad', 'Excluded area']],
  marks: [['dark', 'Detected mark'], ['gap', 'Encoder gap'], ['bad', 'Invalid sample']],
  splices: [['splice', 'Join candidate'], ['gap', 'Encoder gap'], ['bad', 'Invalid sample']],
  texture: [['edge', 'Stripe direction'], ['gap', 'Encoder gap'], ['bad', 'Invalid sample']],
  illumination: [['reflectance', 'Reflectance'], ['gap', 'Encoder gap'], ['bad', 'Invalid sample']],
  transverse: [['reflectance', 'Raw sensor codes'], ['mark', 'Ruler line'], ['dark', 'Stain or weak evidence']],
}

const REGION = [
  ['station_start_mm', 'Start station (mm)'],
  ['station_end_mm', 'End station (mm)'],
  ['transverse_start_mm', 'Start across (mm)'],
  ['transverse_end_mm', 'End across (mm)'],
]

const chosen = {
  capture: null,
  section: 'overview',
  tab: 'analysis',
  analysis: null,
  extent: null,
  window: '',
  reviewed: '',
  windowReport: null,
  snapshot: null,
  compare: null,
  probe: null,
  profile: null,
  rulerPoints: [],
  ruler: null,
  rulerLines: null,
  transverse: null,
  request: 0,
  load: 0,
  probeRequest: 0,
}

let toastTimer
const regularView = () => !['windows', 'ruler', 'snapshots', 'transverse'].includes(chosen.section)
const layer = () => VIEW[chosen.section][2]
const revision = () => chosen.capture.state.revision
const base = () => '/api/captures/' + chosen.capture.id

async function api(path, options = {}) {
  const response = await fetch(path, options)
  if (!response.ok) {
    let content
    try { content = await response.json() } catch { content = {} }
    const error = new Error(content.error || ('Request failed (' + response.status + ')'))
    error.status = response.status
    throw error
  }
  return response
}

async function json(path, options) {
  return (await api(path, options)).json()
}

function send(path, data, method = 'POST') {
  return json(path, {
    method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data),
  })
}

function notify(message, error = false) {
  const status = $('#status')
  status.textContent = message
  status.classList.toggle('error', error)
  status.hidden = false
  clearTimeout(toastTimer)
  toastTimer = setTimeout(() => { status.hidden = true }, error ? 6500 : 3500)
}

function guarded(run) {
  return async (event) => {
    try {
      await run(event)
    } catch (error) {
      notify(error.message, true)
      if (error.status === 409 && chosen.capture) {
        await openCapture(chosen.capture.id, false).catch(() => {})
      }
    }
  }
}

function anchorURL(path, params = {}) {
  return path + '?' + new URLSearchParams(params).toString()
}

function viewParams(extra = {}) {
  const parameters = { layer: layer(), revision: revision(), ...extra }
  if (chosen.window) parameters.window = chosen.window
  return parameters
}

function replaceOptions(control, choices, current) {
  control.replaceChildren(...choices.map(([value, title]) => {
    const node = element('option', title)
    node.value = value
    return node
  }))
  control.value = current || choices[0]?.[0] || ''
}

async function loadLibrary(selectID) {
  const library = await json('/api/captures')
  const selector = $('#capture-picker')
  if (!library.captures.length) throw new Error('No captures are available.')
  const available = library.captures.map((item) => [item.id, item.name])
  const previous = selectID || localStorage.getItem('rollscope:last-capture')
  const ref = available.some(([id]) => id === previous) ? previous : available[0][0]
  replaceOptions(selector, available, ref)
  $('#demo-list').replaceChildren(...library.demos.map((demo) => {
    const action = button(demo.name, guarded(async () => {
      const item = await send('/api/captures/demo', { profile: demo.id })
      $('#library-dialog').close()
      await loadLibrary(item.id)
      notify('Synthetic capture added.')
    }))
    action.append(element('small', demo.material))
    return action
  }))
  await openCapture(ref)
}

async function openCapture(id, reset = true) {
  const token = ++chosen.load
  const detail = await json('/api/captures/' + id)
  if (token !== chosen.load) return
  localStorage.setItem('rollscope:last-capture', id)
  chosen.capture = detail
  if (reset) {
    Object.assign(chosen, {
      section: 'overview', tab: 'analysis', window: '', reviewed: '', snapshot: null,
      compare: null, probe: null, profile: null, ruler: null, rulerPoints: [],
      rulerLines: null, transverse: null,
    })
  }
  $('#capture-picker').value = detail.id
  $('#capture-name').textContent = detail.name
  $('#capture-badge').hidden = !detail.metadata.synthetic
  const dimensions = `${whole.format(detail.overview.rows)} lines × ${whole.format(detail.overview.columns)} columns`
  $('#capture-meta').textContent = `${detail.metadata.material} · ${dimensions} · ${format(detail.overview.station_end_mm - detail.overview.station_start_mm)} mm captured`
  $('#source-download').href = base() + '/source'
  $('#project-download').href = base() + '/project'
  $('#source-sha').textContent = `Source ${detail.sha256.slice(0, 14)}…`
  updateCaptureChrome()
  await renderView()
}

function updateCaptureChrome() {
  const sample = chosen.capture.quick_readings
  const entries = [
    ['Station samples', whole.format(sample.stations), '', '▥'],
    ['Station pitch', format(sample.pitch_mm), 'mm', '↕'],
    ['Usable grid', (sample.known_fraction * 100).toFixed(1), '%', '⊡'],
    ['Encoder gaps', whole.format(sample.gap_count), '', '⋯', sample.gap_count > 0],
    [sample.width_calibrated ? 'Calibrated strip width' : 'Median strip width',
      format(sample.width_mm), 'mm', '↔', sample.uncalibrated_columns > 0],
  ]
  $('#readings').replaceChildren(...entries.map(([title, value, unit, signal, caution]) => {
    const card = element('div', null, 'reading' + (caution ? ' attention' : ''))
    const reading = element('div', value, 'reading-value')
    if (unit) reading.append(element('small', unit))
    const mark = element('span', signal, 'reading-signal')
    mark.setAttribute('aria-hidden', 'true')
    card.append(element('div', title, 'reading-label'), reading, mark)
    return card
  }))
  const windows = chosen.capture.state.windows.map((item) => [item.id, item.label])
  if (!windows.some(([id]) => id === chosen.window)) chosen.window = ''
  replaceOptions($('#window-picker'), [['', 'Full capture'], ...windows], chosen.window)
}

function showErrorImage(error) {
  $('#raster').removeAttribute('src')
  $('#empty-image').textContent = error.message
  $('#empty-image').hidden = false
}

async function renderView() {
  if (!chosen.capture) return
  const token = ++chosen.request
  const [title, description] = VIEW[chosen.section]
  const raw = layer() === 'sensor' || layer() === 'flat'
  $('#marker').hidden = true
  $('#window-picker').disabled = raw || chosen.section === 'ruler'
  $('#view-title').textContent = title
  $('#view-subtitle').textContent = description
  $('#axis-y-caption').textContent = raw ? 'ACQUISITION LINE' : 'STATION · MM'
  $('#snapshot-button').disabled = !regularView()
  for (const nav of $('#navigation').querySelectorAll('button')) {
    nav.setAttribute('aria-current', nav.dataset.section === chosen.section)
  }
  const categories = LEGEND[chosen.section] || LEGEND.overview
  $('#legend').replaceChildren(...categories.map(([color, name]) => {
    const item = element('span', null, 'legend-item')
    item.append(element('i', null, 'swatch ' + color), document.createTextNode(name))
    return item
  }))
  $('#view-footnote').textContent = chosen.section === 'ruler'
    ? 'Select two points on the station image.'
    : chosen.section === 'transverse'
      ? 'Ruler lines are detected from this raw picture. Grade their evidence in the inspector.'
      : 'Select a pixel to inspect its source lines.'
  const sectionLink = $('#section-download')
  sectionLink.hidden = !regularView()
  if (regularView()) {
    sectionLink.href = anchorURL(base() + `/analysis/${chosen.section}/export`, { revision: revision() })
  }
  const parameters = viewParams({ overlay: $('#overlay-toggle').checked ? 1 : 0 })
  const viewURL = anchorURL(base() + '/view', parameters)
  const imageURL = anchorURL(base() + '/preview.png', parameters)
  const analysisURL = regularView()
    ? anchorURL(base() + '/analysis/' + chosen.section, { revision: revision() })
    : null
  chosen.extent = null
  chosen.analysis = null
  chosen.probe = null
  chosen.profile = null
  renderInspector()
  renderTables()
  try {
    const requests = [
      json(viewURL),
      analysisURL ? json(analysisURL) : Promise.resolve(null),
    ]
    if (chosen.section === 'transverse') requests.push(loadTransverseData())
    const result = await Promise.all(requests)
    if (token !== chosen.request) return
    chosen.extent = result[0]
    chosen.analysis = result[1]
  } catch (error) {
    if (token === chosen.request) showErrorImage(error)
    throw error
  }
  const image = $('#raster')
  image.src = imageURL
  image.alt = title + ' image'
  $('#empty-image').hidden = true
  $('#grid-size').textContent = `${whole.format(chosen.extent.source_width)} × ${whole.format(chosen.extent.source_height)} cells`
  paintTicks()
  renderInspector()
  renderTables()
  redrawRuler()
}

function spacedTicks(first, last, count) {
  return Array.from({ length: count }, (_, index) => {
    const value = first + (last - first) * index / (count - 1)
    return element('span', whole.format(value))
  })
}

function paintTicks() {
  const extent = chosen.extent
  $('#axis-y-ticks').replaceChildren(...spacedTicks(extent.y_from, extent.y_to, 8))
  $('#axis-x-ticks').replaceChildren(...spacedTicks(extent.x_from_mm, extent.x_to_mm, 6))
}

function activeSection(section) {
  chosen.section = section
  chosen.tab = 'analysis'
  if (section === 'ruler') chosen.window = ''
  if (layer() === 'sensor' || layer() === 'flat') chosen.window = ''
  updateCaptureChrome()
  return renderView()
}

function selectTab(tab) {
  chosen.tab = tab
  renderInspector()
}

function title(name) {
  return element('h3', name, 'detail-title')
}

function copy(message) {
  return element('p', message, 'detail-copy')
}

function renderInspector() {
  for (const control of $('#detail-tabs').querySelectorAll('button')) {
    control.setAttribute('aria-current', control.dataset.tab === chosen.tab)
    control.disabled = control.dataset.tab === 'settings' && !VIEW[chosen.section][3]
  }
  const body = $('#detail-body')
  body.replaceChildren()
  if (!chosen.capture) return
  if (chosen.tab === 'settings') return renderSettings(body)
  if (chosen.tab === 'probe') return renderProbe(body)
  if (chosen.tab === 'notes') return renderNotes(body)
  if (chosen.section === 'transverse') return renderTransverse(body)
  if (chosen.section === 'windows') return renderWindows(body)
  if (chosen.section === 'ruler') return renderRuler(body)
  if (chosen.section === 'snapshots') return renderSnapshots(body)
  if (!chosen.analysis) {
    body.append(copy('Loading measurements…'))
    return
  }
  body.append(title(chosen.analysis.name), keyValues(chosen.analysis.summary))
  for (const plot of chosen.analysis.charts) body.append(chart(plot))
}

function tableBlocks() {
  if (chosen.section === 'transverse') return []
  if (chosen.section === 'windows' && chosen.windowReport) {
    return [
      { title: 'Unsampled intervals', rows: chosen.windowReport.gaps },
      { title: 'Detected spot footprints in window', rows: chosen.windowReport.spots },
      { title: 'Station band measurements', rows: chosen.windowReport.profiles.station_bands },
    ]
  }
  if (chosen.section === 'ruler' && chosen.ruler) {
    return [
      { title: 'Sampling intervals', rows: chosen.ruler.intervals },
      { title: 'Spot intersections', rows: chosen.ruler.spots },
      { title: 'Path samples', rows: chosen.ruler.profile },
    ]
  }
  if (chosen.section === 'snapshots' && chosen.compare) {
    return [
      { title: 'Measurement changes', rows: chosen.compare.changes },
      { title: 'Setting changes', rows: chosen.compare.setting_changes },
      { title: 'Table size changes', rows: chosen.compare.table_sizes },
    ]
  }
  if (chosen.section === 'snapshots' && chosen.snapshot) return chosen.snapshot.result.tables
  return chosen.analysis?.tables || []
}

function renderTables() {
  const blocks = tableBlocks()
  const panel = $('#data-panel')
  panel.hidden = !blocks.length
  if (!blocks.length) return
  $('#tables-heading').textContent = chosen.section === 'snapshots' && chosen.compare
    ? 'Report comparison'
    : 'Measurements'
  $('#tables-subhead').textContent = VIEW[chosen.section][0]
  $('#tables').replaceChildren(...blocks.map((report) => {
    const group = element('section')
    group.append(element('h3', report.title, 'table-title'), table(report.rows, guarded(locate)))
    return group
  }))
}

function renderSettings(body) {
  const section = VIEW[chosen.section][3]
  if (!section) {
    body.append(copy('This view has no adjustable settings.'))
    return
  }
  const settings = chosen.capture.state.settings[section]
  const form = element('form')
  if (section === 'grid') form.append(copy('Leave a field empty to use the measured encoder pitch or its automatic gap limit.'))
  for (const [name, label, step] of SETTINGS[section]) {
    form.append(field(label, name, settings[name], { type: 'number', step, required: section !== 'grid' }))
  }
  const submit = element('button', 'Apply settings', 'button primary small')
  submit.type = 'submit'
  form.append(submit)
  form.addEventListener('submit', guarded(async (event) => {
    event.preventDefault()
    const values = Object.fromEntries(Array.from(new FormData(form), ([name, value]) => [
      name, value === '' ? null : Number(value),
    ]))
    const response = await send(base() + '/settings', { section, values, revision: revision() })
    Object.assign(chosen.capture, response)
    if (section === 'grid') chosen.window = ''
    updateCaptureChrome()
    await renderView()
    notify('Settings applied.')
  }))
  body.append(form)
}

function renderNotes(body) {
  const form = element('form')
  form.append(title('Capture notes'))
  form.append(field('Notes', 'notes', chosen.capture.state.notes, { long: true, max: 4000 }))
  const submit = element('button', 'Save notes', 'button primary small')
  submit.type = 'submit'
  form.append(submit)
  form.addEventListener('submit', guarded(async (event) => {
    event.preventDefault()
    const notes = new FormData(form).get('notes')
    const result = await send(base() + '/notes', { notes, revision: revision() }, 'PUT')
    chosen.capture.state = result.state
    await renderView()
    notify('Notes saved.')
  }))
  body.append(form)
}

function rasterPoint(event) {
  const area = $('#raster').getBoundingClientRect()
  return {
    x: Math.max(0, Math.min(1, (event.clientX - area.left) / area.width)),
    y: Math.max(0, Math.min(1, (event.clientY - area.top) / area.height)),
  }
}

function markPoint(coords) {
  const marker = $('#marker')
  marker.style.setProperty('--point-x', coords.x * 100 + '%')
  marker.style.setProperty('--point-y', coords.y * 100 + '%')
  marker.hidden = false
}

async function locate(data) {
  const extent = chosen.extent
  if (!extent) return
  const row = data.next_line ?? data.source_line ?? data.row
  const station = extent.y_unit === 'line'
    ? row
    : data.station_mm ?? data.station_after_mm ?? data.from_mm
  const x = data.transverse_mm ?? data.center_mm
  const yCoord = (station - extent.y_from) / (extent.y_to - extent.y_from || 1)
  const xCoord = Number.isFinite(x)
    ? (x - extent.x_from_mm) / (extent.x_to_mm - extent.x_from_mm || 1)
    : 0.5
  if (!Number.isFinite(yCoord)) return
  if (yCoord < 0 || yCoord > 1 || xCoord < 0 || xCoord > 1) {
    notify('Switch to Full capture to inspect this position.')
    return
  }
  $('#viewport').scrollIntoView({ block: 'nearest' })
  await pick({ x: xCoord, y: yCoord })
}

async function pick(coords) {
  if (!chosen.extent) return
  const token = ++chosen.probeRequest
  const viewToken = chosen.request
  markPoint(coords)
  const outcome = await json(anchorURL(base() + '/probe', viewParams(coords)))
  if (token !== chosen.probeRequest || viewToken !== chosen.request) return
  if (chosen.section === 'ruler') {
    if (chosen.rulerPoints.length >= 2) chosen.rulerPoints = []
    chosen.rulerPoints.push({ station_mm: outcome.station_mm, transverse_mm: outcome.transverse_mm })
    chosen.ruler = null
    if (chosen.rulerPoints.length === 2) await evaluateRuler()
    else {
      redrawRuler()
      renderInspector()
      renderTables()
    }
    return
  }
  chosen.probe = outcome
  chosen.profile = null
  chosen.probeY = coords.y
  if (chosen.section === 'windows' && chosen.tab === 'analysis') return
  selectTab('probe')
}

function renderProbe(body) {
  if (!chosen.probe) {
    body.append(copy('Select a point on the image to see its samples and physical coordinates.'))
    return
  }
  const result = chosen.probe
  const { lineage } = result
  const values = { ...result }
  for (const key of ["coordinate", "lineage", "sources", "layer", "revision"]) delete values[key]
  body.append(title('Selected cell'), keyValues(values))
  if (typeof lineage.sum_of_weights === 'number') {
    body.append(keyValues({
      composed_reflectance: lineage.composed_reflectance,
      source_weight: lineage.sum_of_weights,
    }))
  }
  for (const group of lineage.segments) {
    const items = table(group.rows, null, 'table-scroll mini-table')
    body.append(title(group.kind.replaceAll('_', ' ')), items)
  }
  body.append(button('Show transverse profile', guarded(async () => {
    chosen.profile = await json(anchorURL(base() + '/profile', viewParams({ y: chosen.probeY })))
    renderInspector()
  }), 'small'))
  if (chosen.profile) {
    const plot = { title: 'Transverse profile', x: 'transverse_mm', y: ['value'],
      unit: chosen.profile.value_unit, records: chosen.profile.measurements,
    }
    body.append(chart(plot))
  }
}

function reviewChoice(item, active, onClick, sublabel) {
  const control = element('button', item.label, 'review-item' + (active ? ' active' : ''))
  control.type = 'button'
  control.append(element('span', sublabel))
  control.addEventListener('click', guarded(onClick))
  return control
}

async function inspectWindow(id, zoom = false) {
  chosen.reviewed = id
  const url = anchorURL(base() + '/windows/' + id, { revision: revision() })
  const response = await json(url)
  if (chosen.reviewed !== id || chosen.section !== 'windows') return
  chosen.windowReport = response
  if (zoom) {
    chosen.window = id
    updateCaptureChrome()
    await renderView()
  } else {
    renderInspector()
    renderTables()
  }
}

function defaultBounds() {
  const info = chosen.capture.overview
  const y0 = info.station_start_mm
  const y1 = info.station_end_mm
  const x0 = info.transverse_start_mm
  const x1 = info.transverse_end_mm
  return {
    station_start_mm: y0 + 0.3 * (y1 - y0),
    station_end_mm: y0 + 0.43 * (y1 - y0),
    transverse_start_mm: x0 + 0.12 * (x1 - x0),
    transverse_end_mm: x0 + 0.89 * (x1 - x0),
  }
}

function renderWindows(body) {
  const items = chosen.capture.state.windows
  body.append(title('Saved regions'))
  if (!items.length) body.append(copy('Save a physical interval to revisit its image and evidence.'))
  if (items.length) {
    const stack = element('div', null, 'review-list')
    for (const item of items) {
      const interval = `${format(item.region.station_start_mm)}–${format(item.region.station_end_mm)} mm · ${item.status}`
      stack.append(reviewChoice(item, item.id === chosen.reviewed, () => inspectWindow(item.id), interval))
    }
    body.append(stack)
  }
  const selected = items.find((item) => item.id === chosen.reviewed)
  if (selected) {
    const actions = element('div', null, 'field-row')
    const exportUrl = anchorURL(base() + '/windows/' + selected.id + '/export', { revision: revision() })
    actions.append(
      button('View window', guarded(() => inspectWindow(selected.id, true)), 'small'),
      link('Export evidence', exportUrl),
      button('New window', () => {
        chosen.reviewed = ''
        chosen.windowReport = null
        renderInspector()
        renderTables()
      }, 'small')
    )
    body.append(actions)
  }
  if (selected && chosen.windowReport?.window.id === selected.id) {
    body.append(title('Coverage'), keyValues(chosen.windowReport.summary))
    const plot = { title: 'Window reflectance by station', x: 'center', y: ['median'],
      unit: 'reflectance', records: chosen.windowReport.profiles.station_bands,
    }
    body.append(chart(plot))
  }
  body.append(title(selected ? 'Edit window' : 'New window'), windowForm(selected))
}

function windowForm(selected) {
  const form = element('form')
  const coordinates = element('div', null, 'field-grid')
  const bounds = selected?.region || defaultBounds()
  form.append(field('Window name', 'label', selected?.label || '', { required: true, max: 75 }))
  for (const [name, label] of REGION) {
    coordinates.append(field(label, name, bounds[name], { type: 'number', required: true }))
  }
  form.append(coordinates)
  form.append(field('Status', 'status', selected?.status || 'open', {
    choices: [['open', 'Open'], ['watch', 'Watch'], ['resolved', 'Resolved']],
  }))
  form.append(field('Review note', 'note', selected?.note || '', { long: true, max: 1000 }))
  const actions = element('div', null, 'field-row')
  const submit = element('button', 'Save window', 'button primary small')
  submit.type = 'submit'
  actions.append(submit)
  if (selected) {
    const remove = button('Remove window', guarded(async () => {
      if (!window.confirm(`Remove “${selected.label}” from this capture?`)) return
      const url = base() + '/windows/' + selected.id
      const response = await send(url, { revision: revision() }, 'DELETE')
      chosen.capture.state = response.state
      if (chosen.window === selected.id) chosen.window = ''
      chosen.reviewed = ''
      chosen.windowReport = null
      updateCaptureChrome()
      await renderView()
      notify('Window removed.')
    }), 'small danger')
    actions.append(remove)
  }
  form.append(actions)
  form.addEventListener('submit', guarded(async (event) => {
    event.preventDefault()
    const values = new FormData(form)
    const region = Object.fromEntries(REGION.map(([name]) => [name, Number(values.get(name))]))
    const payload = {
      label: values.get('label'), status: values.get('status'), note: values.get('note'),
      revision: revision(), region,
    }
    const url = base() + '/windows' + (selected ? '/' + selected.id : '')
    const outcome = await send(url, payload, selected ? 'PUT' : 'POST')
    chosen.capture.state = outcome.state
    await inspectWindow(outcome.window.id, true)
    notify('Review window saved.')
  }))
  return form
}

async function loadTransverseData() {
  const [lines, calibration] = await Promise.all([
    json(base() + '/transverse/lines'),
    json(base() + '/transverse'),
  ])
  chosen.rulerLines = lines
  chosen.transverse = calibration
}

function renderTransverse(body) {
  if (!chosen.rulerLines) {
    body.append(copy('Loading ruler lines…'))
    return
  }
  const data = chosen.rulerLines
  const existing = chosen.capture.state.transverse_calibration
  body.append(title('Ruler picture'))
  body.append(copy(
    `Detected ${data.line_count} dark ruler line${data.line_count === 1 ? '' : 's'} across ${data.capture.columns} detector columns. `
    + 'Enter the known transverse position (mm) printed beside each line, then build the calibration.'
  ))
  if (existing) body.append(calibrationCallout(existing))
  body.append(transverseForm(data))
  if (chosen.transverse?.calibration) body.append(savedCalibration())
}

function calibrationCallout(calibration) {
  const note = element('p', null, 'result-callout')
  note.textContent = `Active calibration: ${calibration.anchor_count} anchors, RMS ${format(calibration.rms_error_mm)} mm, `
    + `reliable span ${format(calibration.first_mm)}–${format(calibration.last_mm)} mm. Columns outside this span stay uncalibrated.`
  return note
}

function transverseForm(data) {
  const form = element('form')
  const existing = chosen.capture.state.transverse_calibration
  const previous = new Map((existing?.anchors || []).map((anchor) => [anchor.line_id, anchor.known_mm]))
  const rows = data.lines.map((line) => ({
    line_id: line.line_id,
    column: line.column,
    slope: line.slope,
    prominence: line.prominence,
    coverage: line.coverage,
    straightness: line.straightness,
    score: line.score,
    stained: line.stained,
    stain_fraction: line.stain_fraction,
    status: line.rejected_reason || 'ok',
    known_mm: previous.has(line.line_id) ? previous.get(line.line_id) : '',
    included: !line.rejected_reason,
  }))
  const grid = element('div', null, 'ruler-grid')
  const head = element('div', null, 'ruler-row ruler-head')
  ['Line', 'Column', 'Slope', 'Prominence', 'Coverage', 'Bend', 'Evidence', 'Known mm', 'Use'].forEach((name) => {
    head.append(element('span', name))
  })
  grid.append(head)
  for (const row of rows) {
    const node = element('div', null, 'ruler-row' + (row.stained ? ' flagged' : '') + (row.status !== 'ok' ? ' weak' : ''))
    node.append(element('span', String(row.line_id), 'mono'))
    node.append(element('span', format(row.column), 'mono'))
    node.append(element('span', format(row.slope), 'mono'))
    node.append(element('span', format(row.prominence), 'mono'))
    node.append(element('span', (row.coverage * 100).toFixed(0) + '%', 'mono'))
    node.append(element('span', format(row.straightness), 'mono'))
    const evidence = element('span', null, 'evidence-cell')
    evidence.append(evidenceBadge(row))
    node.append(evidence)
    const position = element('input')
    position.type = 'number'
    position.step = 'any'
    position.name = `position_${row.line_id}`
    position.value = row.known_mm
    position.required = true
    node.append(position)
    const include = element('input')
    include.type = 'checkbox'
    include.name = `include_${row.line_id}`
    include.checked = row.included
    node.append(include)
    grid.append(node)
  }
  form.append(grid)
  const weakest = [...rows].sort((a, b) => a.score - b.score)[0]
  if (weakest) {
    body_weakness(form, weakest, rows)
  }
  const actions = element('div', null, 'field-row')
  const submit = element('button', 'Build calibration', 'button primary small')
  submit.type = 'submit'
  actions.append(submit)
  actions.append(button('Export calibration', guarded(exportTransverse), 'small'))
  const importButton = button('Import calibration…', () => importTransverse(form), 'small')
  actions.append(importButton)
  if (chosen.capture.state.transverse_calibration) {
    actions.append(button('Remove calibration', guarded(removeTransverse), 'small danger'))
  }
  form.append(actions)
  form.addEventListener('submit', guarded(async (event) => {
    event.preventDefault()
    const entries = new FormData(form)
    const positions = data.lines.map((line) => Number(entries.get(`position_${line.line_id}`)))
    if (positions.some((value) => !Number.isFinite(value))) {
      throw new Error('Give every ruler line a known transverse position in millimetres.')
    }
    const accepted = data.lines
      .filter((line) => entries.get(`include_${line.line_id}`) === 'on')
      .map((line) => line.line_id)
    const response = await send(base() + '/transverse', {
      revision: revision(), positions_mm: positions, accepted_lines: accepted,
    })
    chosen.capture.state = response.state
    await refreshTransverse()
    notify('Transverse calibration built. Unreliable lines are listed below.')
  }))
  return form
}

function body_weakness(form, weakest, rows) {
  const reasons = {
    faint_line: 'faint contrast',
    partial_line: 'partial row coverage',
    bent_line: 'excessive bend',
    ok: weakest.stained ? 'a nearby stain' : 'the lowest evidence score',
  }
  const text = `Weakest evidence: line ${weakest.line_id} (${reasons[weakest.status] || weakest.status}, score ${format(weakest.score)}). `
    + 'Untick "Use" to exclude it before building so a local deformation cannot pull neighbouring widths.'
  form.append(element('p', text, 'detail-copy warning-copy'))
}

function evidenceBadge(row) {
  const badge = element('span', null, 'evidence-badge')
  if (row.status !== 'ok') {
    badge.classList.add('badge-weak')
    badge.textContent = row.status.replaceAll('_', ' ')
    return badge
  }
  if (row.stained) {
    badge.classList.add('badge-stain')
    badge.textContent = `stain ${(row.stain_fraction * 100).toFixed(0)}%`
    return badge
  }
  badge.classList.add('badge-ok')
  badge.textContent = 'reliable'
  return badge
}

function savedCalibration() {
  const block = element('div')
  const calibration = chosen.transverse.calibration
  block.append(title('Current calibration'), keyValues(calibration))
  const anchorRows = chosen.transverse.anchors.map((anchor) => ({
    line_id: anchor.line_id,
    column: anchor.column,
    known_mm: anchor.known_mm,
    calibrated_mm: anchor.calibrated_mm,
    fit_residual_mm: anchor.fit_residual_mm,
    robust_outlier: anchor.robust_outlier,
    score: anchor.score,
    stained: anchor.stained,
  }))
  block.append(table(anchorRows, null, 'table-scroll mini-table'))
  block.append(chart({
    title: 'Anchor fit residual (mm)', x: 'line_id', y: ['fit_residual_mm'], unit: 'mm',
    records: anchorRows,
  }))
  if (chosen.transverse.rejected_lines.length) {
    block.append(title('Excluded ruler lines'))
    block.append(table(chosen.transverse.rejected_lines, null, 'table-scroll mini-table'))
  }
  return block
}

async function refreshTransverse() {
  await loadTransverseData()
  renderInspector()
  renderTables()
  updateCaptureChrome()
}

async function exportTransverse() {
  const response = await api(base() + '/transverse/export')
  const payload = await response.json()
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const anchor = link('Download calibration JSON', url)
  anchor.download = `rollscope-${chosen.capture.id}-transverse.json`
  document.body.append(anchor)
  anchor.click()
  anchor.remove()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

function importTransverse(form) {
  const picker = element('input')
  picker.type = 'file'
  picker.accept = '.json,application/json'
  picker.addEventListener('change', guarded(async () => {
    const [file] = picker.files
    if (!file) return
    const calibration = JSON.parse(await file.text())
    const response = await send(base() + '/transverse', { revision: revision(), calibration }, 'PUT')
    chosen.capture.state = response.state
    await refreshTransverse()
    notify('Transverse calibration imported.')
  }))
  picker.click()
}

async function removeTransverse() {
  if (!window.confirm('Remove the transverse calibration from this capture? Measurements revert to the nominal column pitch.')) return
  const response = await send(base() + '/transverse', { revision: revision() }, 'DELETE')
  chosen.capture.state = response.state
  await refreshTransverse()
  notify('Transverse calibration removed.')
}

function rulerDefaults() {
  const dimensions = chosen.extent
  if (!dimensions) return [{}, {}]
  const px = dimensions.x_to_mm - dimensions.x_from_mm
  const py = dimensions.y_to - dimensions.y_from
  return [
    { station_mm: dimensions.y_from + py * 0.18, transverse_mm: dimensions.x_from_mm + px * 0.26 },
    { station_mm: dimensions.y_from + py * 0.61, transverse_mm: dimensions.x_from_mm + px * 0.69 },
  ]
}

function parseRuler(form) {
  const entries = new FormData(form)
  return ['start', 'end'].map((id) => ({
    station_mm: Number(entries.get(id + '_station')),
    transverse_mm: Number(entries.get(id + '_cross')),
  }))
}

function rulerInput(form, id, label, point) {
  form.append(title(label))
  const values = element('div', null, 'field-grid')
  values.append(field('Station (mm)', id + '_station', point.station_mm, { type: 'number', required: true }))
  values.append(field('Across (mm)', id + '_cross', point.transverse_mm, { type: 'number', required: true }))
  form.append(values)
}

function renderRuler(body) {
  if (!chosen.extent) {
    body.append(copy('Loading station coordinates…'))
    return
  }
  body.append(copy(chosen.rulerPoints.length === 1
    ? 'Select the second point in the image.'
    : 'Pick two image cells or enter exact physical coordinates.'))
  const form = element('form')
  const starting = rulerDefaults()
  rulerInput(form, 'start', 'Start', chosen.rulerPoints[0] || starting[0])
  rulerInput(form, 'end', 'End', chosen.rulerPoints[1] || starting[1])
  const submit = element('button', 'Measure path', 'button primary small')
  submit.type = 'submit'
  const tools = element('div', null, 'field-row')
  tools.append(submit)
  if (chosen.ruler || chosen.rulerPoints.length) {
    tools.append(button('Reset', () => {
      chosen.rulerPoints = []
      chosen.ruler = null
      $('#marker').hidden = true
      redrawRuler()
      renderInspector()
      renderTables()
    }, 'small'))
  }
  form.append(tools)
  form.addEventListener('submit', guarded(async (event) => {
    event.preventDefault()
    chosen.rulerPoints = parseRuler(form)
    await evaluateRuler()
  }))
  body.append(form)
  if (!chosen.ruler) return
  body.append(title('Path measurement'), keyValues(chosen.ruler.summary))
  body.append(button('Export path ZIP', guarded(exportRuler), 'small'))
  body.append(chart({
    title: 'Line profile', x: 'distance_mm', y: ['reflectance'], unit: 'reflectance',
    records: chosen.ruler.profile,
  }))
}

function rulerPayload() {
  return { start: chosen.rulerPoints[0], end: chosen.rulerPoints[1], revision: revision() }
}

async function evaluateRuler() {
  const token = chosen.request
  const result = await send(base() + '/ruler', rulerPayload())
  if (token !== chosen.request) return
  chosen.ruler = result
  redrawRuler()
  renderInspector()
  renderTables()
}

function physicalProjection(point, width, height) {
  const extent = chosen.extent
  const x = (point.transverse_mm - extent.x_from_mm) / (extent.x_to_mm - extent.x_from_mm || 1)
  const y = (point.station_mm - extent.y_from) / (extent.y_to - extent.y_from || 1)
  return [x * width, y * height]
}

function redrawRuler() {
  const canvas = $('#ruler-layer')
  const width = canvas.clientWidth
  const height = canvas.clientHeight
  const ratio = window.devicePixelRatio || 1
  canvas.width = Math.round(width * ratio)
  canvas.height = Math.round(height * ratio)
  if (chosen.section !== 'ruler' || !chosen.extent || !chosen.rulerPoints.length) return
  const brush = canvas.getContext('2d')
  brush.scale(ratio, ratio)
  const xy = (point) => physicalProjection(point, width, height)
  const [start, end] = chosen.rulerPoints
  brush.lineWidth = 3
  brush.strokeStyle = '#f9d870'
  brush.fillStyle = '#f9d870'
  if (end) {
    const colors = { gap: '#e87977', bad_pixel: '#d992d4', background: '#78b9f4', material: '#f9d870' }
    const length = chosen.ruler?.summary.length_mm || 1
    const sections = chosen.ruler?.intervals || [{ from_mm: 0, to_mm: length, support: 'material' }]
    const at = (fraction) => ({
      station_mm: start.station_mm + fraction * (end.station_mm - start.station_mm),
      transverse_mm: start.transverse_mm + fraction * (end.transverse_mm - start.transverse_mm),
    })
    for (const interval of sections) {
      brush.strokeStyle = colors[interval.support]
      brush.beginPath()
      brush.moveTo(...xy(at(interval.from_mm / length)))
      brush.lineTo(...xy(at(interval.to_mm / length)))
      brush.stroke()
    }
  }
  for (const point of chosen.rulerPoints) {
    brush.beginPath()
    brush.arc(...xy(point), 5, 0, 2 * Math.PI)
    brush.fill()
  }
}

async function exportRuler() {
  const response = await api(base() + '/ruler/export', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(rulerPayload()),
  })
  const url = URL.createObjectURL(await response.blob())
  const anchor = link('Download path', url)
  anchor.download = 'rollscope-ruler.zip'
  document.body.append(anchor)
  anchor.click()
  anchor.remove()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

async function inspectSnapshot(id) {
  const result = await json(base() + '/snapshots/' + id)
  if (chosen.section !== 'snapshots') return
  chosen.snapshot = result
  chosen.compare = null
  renderInspector()
  renderTables()
}

function renderSnapshots(body) {
  const reports = chosen.capture.snapshots
  body.append(title('Inspection reports'))
  if (!reports.length) {
    body.append(copy('Open an analysis view and save its report to compare it later.'))
    return
  }
  const list = element('div', null, 'review-list')
  for (const item of reports) {
    const details = `${VIEW[item.section][0]} · saved at revision ${item.source_revision}`
    list.append(reviewChoice(item, item.id === chosen.snapshot?.id, () => inspectSnapshot(item.id), details))
  }
  body.append(list)
  const saved = chosen.snapshot
  if (!saved) return
  const reportPath = base() + '/snapshots/' + saved.id
  const actions = element('div', null, 'field-row')
  actions.append(link('Export saved report', reportPath + '/export'))
  actions.append(button('Compare with current', guarded(async () => {
    chosen.compare = await json(anchorURL(reportPath + '/compare', { revision: revision() }))
    renderInspector()
    renderTables()
  }), 'small'))
  body.append(actions)
  if (chosen.compare) {
    const summary = chosen.compare.changes.length
      ? `${chosen.compare.changes.length} measurements changed. See the comparison table below.`
      : 'The recorded summary matches the current analysis.'
    body.append(element('p', summary, 'result-callout'))
  }
  body.append(title(saved.label), keyValues(saved.result.summary))
  for (const plot of saved.result.charts) body.append(chart(plot))
}

function wireEvents() {
  $('#capture-picker').addEventListener('change', guarded((event) => openCapture(event.target.value)))
  $('#navigation').addEventListener('click', guarded((event) => {
    const control = event.target.closest('[data-section]')
    if (control && chosen.capture) return activeSection(control.dataset.section)
  }))
  $('#window-picker').addEventListener('change', guarded((event) => {
    chosen.window = event.target.value
    return renderView()
  }))
  $('#overlay-toggle').addEventListener('change', guarded(renderView))
  $('#detail-tabs').addEventListener('click', (event) => {
    const control = event.target.closest('[data-tab]')
    if (control && !control.disabled) selectTab(control.dataset.tab)
  })
  $('#image-frame').addEventListener('click', guarded((event) => pick(rasterPoint(event))))
  $('#raster').addEventListener('error', () => {
    if ($('#raster').getAttribute('src')) showErrorImage(new Error('Could not load this view.'))
  })
  window.addEventListener('resize', redrawRuler)
  $('#library-button').addEventListener('click', () => $('#library-dialog').showModal())
  document.querySelectorAll('[data-close]').forEach((control) => {
    control.addEventListener('click', () => document.getElementById(control.dataset.close).close())
  })
  for (const [input, route, fieldName] of [
    ['#capture-file', 'import', 'capture'], ['#project-file', 'restore', 'project'],
  ]) {
    $(input).addEventListener('change', guarded(async (event) => {
      const [file] = event.target.files
      if (!file) return
      const payload = new FormData()
      payload.append(fieldName, file)
      event.target.value = ''
      const result = await json('/api/captures/' + route, { method: 'POST', body: payload })
      $('#library-dialog').close()
      await loadLibrary(result.id)
      notify(route === 'restore' ? 'Project restored as a new capture.' : 'Capture imported.')
    }))
  }
  $('#snapshot-button').addEventListener('click', () => {
    if (!chosen.capture) return
    $('#snapshot-label').value = VIEW[chosen.section][0]
    $('#snapshot-section-label').textContent = 'Save the current measurements and analysis settings.'
    $('#snapshot-dialog').showModal()
  })
  $('#snapshot-form').addEventListener('submit', guarded(async (event) => {
    event.preventDefault()
    const label = $('#snapshot-label').value
    const result = await send(base() + '/snapshots', { label, section: chosen.section, revision: revision() })
    chosen.capture.state = result.state
    chosen.capture.snapshots.unshift(result.snapshot)
    $('#snapshot-dialog').close()
    await renderView()
    notify('Report saved.')
  }))
}

wireEvents()
loadLibrary().catch((error) => {
  showErrorImage(error)
  notify(error.message, true)
})
