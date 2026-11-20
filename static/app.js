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
  trim: ['Trim planning', 'Isolate up to three spots with two cuts and plan the remaining roll', 'trim'],
  ruler: ['Ruler', 'Measure across sheet, bad pixels and acquisition gaps', 'station'],
  snapshots: ['Saved reports', 'Previously recorded measurements and settings', 'station'],
}

const SETTINGS = {
  calibration: [
    ['minimum_response', 'Minimum response (ADU)', '1'],
    ['maximum_reference_noise', 'Maximum reference variation (ADU)'],
    ['saturation_code', 'Saturation code (ADU)', '1'],
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
  trim: [['trimcut', 'Cut line (sampled station)'], ['trimout', 'Cut-away isolation'], ['trimretain', 'Retained sample'], ['gap', 'Encoder gap'], ['bad', 'Invalid sample']],
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
  trim: null,
  trimBasis: 'longest_segment',
  trimPlanId: '',
  request: 0,
  load: 0,
  probeRequest: 0,
}

let toastTimer
const regularView = () => !['windows', 'ruler', 'snapshots', 'trim'].includes(chosen.section)
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
  if (chosen.section === 'trim' && parameters.layer === 'trim') parameters.layer = 'spots'
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
      trim: null, trimBasis: 'longest_segment', trimPlanId: '',
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
    ['Median strip width', format(sample.width_mm), 'mm', '↔'],
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
  $('#window-picker').disabled = raw || ['ruler', 'trim'].includes(chosen.section)
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
    : chosen.section === 'trim'
      ? 'Cut lines snap to sampled stations. Confirm the plan to export it.'
      : 'Select a pixel to inspect its source lines.'
  const sectionLink = $('#section-download')
  sectionLink.hidden = !regularView()
  if (regularView()) {
    sectionLink.href = anchorURL(base() + `/analysis/${chosen.section}/export`, { revision: revision() })
  }
  if (chosen.section === 'trim') return renderTrimView(token)
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
    const result = await Promise.all([
      json(viewURL),
      analysisURL ? json(analysisURL) : Promise.resolve(null),
    ])
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
  if (section === 'trim') {
    chosen.window = ''
    chosen.trimPlanId = ''
    if (!chosen.trim) chosen.trim = { options: null, selected: [], opened: null, requestId: 0 }
  }
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
    control.disabled =
      (control.dataset.tab === 'settings' && !VIEW[chosen.section][3]) ||
      (control.dataset.tab === 'probe' && chosen.section === 'trim')
  }
  const body = $('#detail-body')
  body.replaceChildren()
  if (!chosen.capture) return
  if (chosen.tab === 'settings') return renderSettings(body)
  if (chosen.tab === 'probe') return renderProbe(body)
  if (chosen.tab === 'notes') return renderNotes(body)
  if (chosen.section === 'windows') return renderWindows(body)
  if (chosen.section === 'trim') return renderTrim(body)
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
  if (chosen.section === 'trim' && chosen.trim) {
    const plan = activeTrimPlan()
    if (!plan) return []
    const spotRows = plan.selected_spots.map((spot) => ({
      ...spot,
      from_mm: spot.footprint_start_mm,
      station_mm: spot.footprint_start_mm,
    }))
    return [
      { title: 'Remnant segments', rows: plan.segments },
      { title: 'Cut positions', rows: trimCutRows(plan) },
      { title: 'Selected spot footprints', rows: spotRows },
    ]
  }
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
  if (chosen.section === 'trim') {
    if (typeof data.spot_id !== 'number') return
    return locateTrimSpot(data.spot_id)
  }
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
  if (chosen.section === 'trim') {
    if (typeof outcome.spot_id === 'number') {
      await locateTrimSpot(outcome.spot_id)
    } else {
      notify('That cell is not part of a detected spot. Click inside a spot footprint to select it for removal.')
    }
    return
  }
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

// ---------- Trim planning ----------

const TRIM_DEFAULTS = { margin_mm: 3, minimum_length_mm: 50, retain_start_mm: null, retain_end_mm: null }

function trimFormState() {
  const form = $('#trim-form')
  if (!form) return { ...TRIM_DEFAULTS, spot_ids: [] }
  const values = new FormData(form)
  const number = (name) => {
    const raw = values.get(name)
    return raw === '' || raw === null ? null : Number(raw)
  }
  return {
    margin_mm: Number(values.get('margin_mm')),
    minimum_length_mm: Number(values.get('minimum_length_mm')),
    retain_start_mm: number('retain_start_mm'),
    retain_end_mm: number('retain_end_mm'),
    spot_ids: chosen.trim?.selected || [],
  }
}

function activeTrimPlan() {
  if (!chosen.trim?.options) return null
  if (chosen.trimPlanId) {
    return chosen.trim.opened ? chosen.trim.opened.current : null
  }
  if (chosen.trim.selected.length && chosen.trim.options.requested) {
    return chosen.trim.options.requested
  }
  return chosen.trim.options[chosen.trimBasis] || null
}

function trimCutRows(plan) {
  const rows = []
  for (const region of plan.cut_regions) {
    rows.push({ role: 'start', ...region.start_cut, region_start_mm: region.start_mm, spot_ids: region.spot_ids })
    rows.push({ role: 'end', ...region.end_cut, region_end_mm: region.end_mm, spot_ids: region.spot_ids })
  }
  return rows
}

async function renderTrimView(token) {
  chosen.extent = null
  chosen.analysis = null
  chosen.probe = null
  chosen.profile = null
  if (!chosen.trim) {
    chosen.trim = { options: null, selected: [], opened: null, requestId: 0 }
  }
  renderInspector()
  renderTables()
  const viewURL = anchorURL(base() + '/view', { layer: 'trim', revision: revision() })
  const extent = await json(viewURL)
  if (token !== chosen.request) return
  chosen.extent = extent
  $('#grid-size').textContent = `${whole.format(extent.source_width)} × ${whole.format(extent.source_height)} cells`
  paintTicks()
  if (chosen.trimPlanId) {
    renderInspector()
    renderTables()
    await paintTrimImage(token)
    return
  }
  await refreshTrim(token)
}

async function refreshTrim(token = chosen.request) {
  if (!chosen.trim) return
  const state = trimFormState()
  const requestToken = ++chosen.trim.requestId
  try {
    const options = await send(base() + '/trim/options', { ...state, revision: revision() })
    if (token !== chosen.request || requestToken !== chosen.trim.requestId) return
    chosen.trim.options = options
    if (!options.objectives_differ) chosen.trimBasis = 'longest_segment'
    renderInspector()
    renderTables()
    await paintTrimImage(token)
  } catch (error) {
    if (requestToken === chosen.trim.requestId) throw error
  }
}

async function paintTrimImage(token) {
  const plan = activeTrimPlan()
  const image = $('#raster')
  $('#empty-image').hidden = true
  if (chosen.trimPlanId && chosen.trim?.opened) {
    const basis = chosen.trim.savedBasis ? 'saved' : 'current'
    const url = anchorURL(base() + `/trim/plans/${chosen.trimPlanId}/preview.png`, { basis, revision: revision() })
    if (image.dataset.trimBlob) {
      URL.revokeObjectURL(image.dataset.trimBlob)
      delete image.dataset.trimBlob
    }
    image.src = url
    redrawRuler()
    return
  }
  if (!plan) return
  const state = trimFormState()
  const response = await api(base() + '/trim/preview.png', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...state, objective: chosen.trimBasis, revision: revision() }),
  })
  if (token !== chosen.request) return
  const previous = image.dataset.trimBlob
  image.src = URL.createObjectURL(await response.blob())
  if (previous) URL.revokeObjectURL(previous)
  image.dataset.trimBlob = image.src
  redrawRuler()
}

function renderTrim(body) {
  if (!chosen.trim) chosen.trim = { options: null, selected: [], opened: null, requestId: 0 }
  const savedPlans = chosen.capture.trim_plans || []
  body.append(title('Confirmed trim plans'))
  if (savedPlans.length) {
    const list = element('div', null, 'review-list')
    for (const item of savedPlans) {
      const sub = `${item.objective === 'longest_segment' ? 'Longest segment' : 'Max retained'} · ${item.spot_ids.length} spot(s) · rev ${item.source_revision}`
      list.append(reviewChoice(item, item.id === chosen.trimPlanId, () => openTrimPlan(item.id), sub))
    }
    body.append(list)
  } else {
    body.append(copy('No confirmed plan yet. Adjust the inputs below and confirm one of the two trade-off plans.'))
  }
  const planActions = element('div', null, 'field-row')
  if (chosen.trimPlanId) {
    planActions.append(button('Back to planning', () => {
      chosen.trimPlanId = ''
      chosen.trim.opened = null
      return renderView()
    }, 'small'))
  }
  body.append(planActions)

  if (chosen.trimPlanId && chosen.trim.opened) return renderOpenedTrim(body)
  body.append(title('Trim inputs'), trimInputsForm())
  const options = chosen.trim.options
  if (!options) {
    body.append(copy('Loading trim candidates…'))
    return
  }
  body.append(copy(`Evaluated ${options.candidate_count} feasible cut layout(s) from ${options.available_spots} detected spots. Two cuts, at most three removed spots.`))
  body.append(objectiveCards(options))
  const plan = activeTrimPlan()
  if (plan) body.append(planSummary(plan))
}

function trimInputsForm() {
  const form = element('form')
  form.id = 'trim-form'
  const values = trimFormState()
  const grid = element('div', null, 'field-grid')
  grid.append(field('Spot margin (mm)', 'margin_mm', values.margin_mm, { type: 'number', required: true }))
  grid.append(field('Minimum shippable length (mm)', 'minimum_length_mm', values.minimum_length_mm, { type: 'number', required: true }))
  form.append(grid)
  const retain = element('div', null, 'field-grid')
  retain.append(field('Retained sample start (mm)', 'retain_start_mm', values.retain_start_mm ?? '', { type: 'number' }))
  retain.append(field('Retained sample end (mm)', 'retain_end_mm', values.retain_end_mm ?? '', { type: 'number' }))
  form.append(retain)
  const spotLine = element('p', null, 'detail-copy')
  spotLine.append(
    Object.assign(document.createElement('strong'), { textContent: 'Removed spots: ' }),
    document.createTextNode(values.spot_ids.length ? values.spot_ids.join(', ') : 'auto-selected by the chosen trade-off')
  )
  form.append(spotLine)
  form.append(copy('Pick spots from the Surface spots table below (rows are clickable), or leave the automatic recommendation. Selections are limited to three spots that merge into one cut-out interval.'))
  const actions = element('div', null, 'field-row')
  actions.append(button('Recompute', guarded(async (event) => {
    event.preventDefault()
    await refreshTrim()
  }), 'small primary'))
  if (values.spot_ids.length) {
    actions.append(button('Clear spot selection', () => {
      chosen.trim.selected = []
      renderInspector()
      return refreshTrim()
    }, 'small'))
  }
  form.append(actions)
  form.addEventListener('submit', (event) => event.preventDefault())
  let debounce
  form.addEventListener('input', () => {
    clearTimeout(debounce)
    debounce = setTimeout(guarded(() => {
      if (!form.reportValidity()) return
      return refreshTrim()
    }), 350)
  })
  return form
}

function objectiveCards(options) {
  const wrap = element('div', null, 'objective-list')
  const manual = chosen.trim.selected.length > 0
  if (manual) {
    wrap.append(element('p', `Manual selection of spot(s) ${chosen.trim.selected.join(', ')} is active. The layout below is the exact evaluation of that selection; click a trade-off card to return to automatic recommendations.`, 'result-callout'))
  }
  for (const [key, label] of [['longest_segment', 'Longest continuous good material'], ['max_retained', 'Maximum retained material']]) {
    const plan = options[key]
    const active = !manual && chosen.trimBasis === key
    const card = element('button', null, 'objective-card' + (active ? ' active' : ''))
    card.type = 'button'
    card.append(element('span', label, 'objective-name'))
    const metrics = element('span', null, 'objective-metrics')
    metrics.append(
      element('em', `Longest ${format(plan.longest_segment_mm)} mm`),
      element('em', `Retained ${format(plan.retained_length_mm)} mm`),
      element('em', `${plan.shippable_segment_count} shippable`),
    )
    card.append(metrics)
    card.append(element('span', `Spots ${plan.selected_spots.map((s) => s.spot_id).join(', ') || '—'}`, 'objective-spots'))
    card.addEventListener('click', guarded(async () => {
      chosen.trimBasis = key
      if (manual) chosen.trim.selected = []
      renderInspector()
      await refreshTrim()
    }))
    wrap.append(card)
  }
  if (!options.objectives_differ) {
    wrap.append(element('p', 'Both trade-offs produce the same cut layout for these constraints.', 'result-callout'))
  }
  return wrap
}

function planSummary(plan) {
  const wrap = element('div')
  if (plan.conflicts.length) {
    for (const item of plan.conflicts) {
      wrap.append(element('p', '⚠ ' + item.message, 'conflict-callout'))
    }
  }
  for (const item of plan.warnings) {
    wrap.append(element('p', item, 'warning-callout'))
  }
  wrap.append(keyValues({
    feasible: plan.feasible,
    cut_count: plan.cut_count,
    removed_length_mm: plan.removed_length_mm,
    retained_length_mm: plan.retained_length_mm,
    longest_segment_mm: plan.longest_segment_mm,
    shippable_length_mm: plan.shippable_length_mm,
    total_length_mm: plan.total_length_mm,
  }))
  if (plan.feasible && !chosen.trimPlanId) {
    wrap.append(button('Confirm this plan', guarded(confirmTrimPlan), 'small primary'))
  }
  return wrap
}

async function confirmTrimPlan() {
  const state = trimFormState()
  const label = window.prompt('Name this confirmed trim plan', `Trim ${new Date().toLocaleDateString()}`)
  if (!label) return
  const result = await send(base() + '/trim/plans', {
    label,
    objective: chosen.trimBasis,
    ...state,
    revision: revision(),
  })
  chosen.capture.state = result.state
  chosen.capture.trim_plans = [...(chosen.capture.trim_plans || []), result.trim_plan]
  notify('Trim plan confirmed and stored.')
  await openTrimPlan(result.trim_plan.id)
}

async function openTrimPlan(id) {
  chosen.trimPlanId = id
  chosen.section = 'trim'
  chosen.tab = 'analysis'
  chosen.trim = chosen.trim || { options: null, selected: [], opened: null, requestId: 0 }
  chosen.trim.savedBasis = true
  const response = await json(base() + '/trim/plans/' + id)
  chosen.trim.opened = response
  await renderView()
}

function renderOpenedTrim(body) {
  const opened = chosen.trim.opened
  const saved = opened.saved
  body.append(title(saved.label))
  if (!opened.settings_match) {
    body.append(element('p', 'Surface-spot detection settings changed after this plan was confirmed. The saved cut layout remains available; the current recomputation may differ.', 'conflict-callout'))
  }
  const basisActions = element('div', null, 'field-row')
  basisActions.append(button('Saved confirmation', () => {
    chosen.trim.savedBasis = true
    renderInspector()
    return paintTrimImage()
  }, 'small' + (chosen.trim.savedBasis ? ' primary' : '')))
  basisActions.append(button('Recompute on current scan', () => {
    chosen.trim.savedBasis = false
    renderInspector()
    return paintTrimImage()
  }, 'small' + (!chosen.trim.savedBasis ? ' primary' : '')))
  body.append(basisActions)
  const shown = chosen.trim.savedBasis ? saved.plan : opened.current
  body.append(keyValues({
    objective: saved.objective,
    margin_mm: saved.margin_mm,
    minimum_length_mm: saved.minimum_length_mm,
    spot_ids: saved.spot_ids,
    source_revision: saved.source_revision,
    created_at: saved.created_at,
  }))
  body.append(planSummary(shown))
  const actions = element('div', null, 'field-row')
  actions.append(link('Export confirmed plan ZIP', base() + '/trim/plans/' + saved.id + '/export'))
  actions.append(button('Delete plan', guarded(async () => {
    if (!window.confirm(`Delete trim plan “${saved.label}”?`)) return
    const response = await send(base() + '/trim/plans/' + saved.id, { revision: revision() }, 'DELETE')
    chosen.capture.state = response.state
    chosen.capture.trim_plans = (chosen.capture.trim_plans || []).filter((item) => item.id !== saved.id)
    chosen.trimPlanId = ''
    chosen.trim.opened = null
    notify('Trim plan deleted.')
    await renderView()
  }), 'small danger'))
  body.append(actions)
}

function locateTrimSpot(spotId) {
  if (chosen.section !== 'trim' || chosen.trimPlanId) return
  if (!chosen.trim) chosen.trim = { options: null, selected: [], opened: null, requestId: 0 }
  if (chosen.trim.selected.includes(spotId)) {
    chosen.trim.selected = chosen.trim.selected.filter((id) => id !== spotId)
  } else if (chosen.trim.selected.length >= 3) {
    notify('At most three spots can be removed in this downtime window.')
    return
  } else {
    chosen.trim.selected = [...chosen.trim.selected, spotId]
  }
  renderInspector()
  return refreshTrim()
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
