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
  ruler: ['Ruler', 'Measure across sheet, bad pixels and acquisition gaps', 'station'],
  alignment: ['Event alignment', 'Operation log mapped onto material stations', 'station'],
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
  alignment: [['anchor', 'Anchor range'], ['confident', 'Confident band'], ['ambiguous', 'Ambiguous band'], ['gap', 'Acquisition gap']],
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
  alignment: null,
  alignmentHit: null,
  draft: null,
  parsed: null,
  pickTarget: null,
  request: 0,
  load: 0,
  probeRequest: 0,
}

let toastTimer
const regularView = () => !['windows', 'ruler', 'snapshots', 'alignment'].includes(chosen.section)
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
      alignment: null, alignmentHit: null, draft: null, parsed: null, pickTarget: null,
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
    : chosen.section === 'alignment'
      ? 'Click the image to list the log records covering that station.'
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
  const alignmentURL = chosen.section === 'alignment'
    ? anchorURL(base() + '/alignment', { revision: revision() })
    : null
  chosen.extent = null
  chosen.analysis = null
  chosen.probe = null
  chosen.profile = null
  chosen.alignmentHit = null
  renderInspector()
  renderTables()
  try {
    const result = await Promise.all([
      json(viewURL),
      analysisURL ? json(analysisURL) : Promise.resolve(null),
      alignmentURL ? json(alignmentURL) : Promise.resolve(null),
    ])
    if (token !== chosen.request) return
    chosen.extent = result[0]
    chosen.analysis = result[1]
    chosen.alignment = result[2]
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
  redrawAlignment()
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
  if (chosen.section === 'windows') return renderWindows(body)
  if (chosen.section === 'ruler') return renderRuler(body)
  if (chosen.section === 'alignment') return renderAlignment(body)
  if (chosen.section === 'snapshots') return renderSnapshots(body)
  if (!chosen.analysis) {
    body.append(copy('Loading measurements…'))
    return
  }
  body.append(title(chosen.analysis.name), keyValues(chosen.analysis.summary))
  for (const plot of chosen.analysis.charts) body.append(chart(plot))
}

function tableBlocks() {
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
  if (chosen.section === 'alignment' && chosen.alignment?.report) {
    const report = chosen.alignment.report
    const blocks = [
      { title: 'Clock legs', rows: report.legs.map(legRow) },
      { title: 'Event placements', rows: report.placements.map(placementRow) },
    ]
    const anchors = chosen.alignment.alignment?.anchors || []
    if (anchors.length) blocks.push({ title: 'Anchors', rows: anchors.map(anchorRow) })
    return blocks
  }
  return chosen.analysis?.tables || []
}

const rounded = (value, digits = 1) => (value == null ? null : Number(value.toFixed(digits)))

function legRow(leg) {
  return {
    leg: leg.title,
    events: leg.event_count,
    anchors: leg.anchor_count,
    status: leg.status.replaceAll('_', ' '),
    implied_speed_mm_s: leg.speed_range_mm_s?.map((v) => rounded(v, 2)) || null,
    measured_speed_mm_s: rounded(leg.measured_speed_mm_s, 2),
  }
}

function placementRow(p) {
  return {
    event: p.event_id,
    kind: p.kind,
    start: p.start_text,
    end: p.end_text,
    leg: p.leg + 1,
    status: p.status,
    station_mm: rounded(p.station_mm),
    from_mm: rounded(p.station_start_mm),
    to_mm: rounded(p.station_end_mm),
    width_mm: rounded(p.width_mm),
    crosses_gap: p.crosses_gap,
  }
}

function anchorRow(anchor) {
  return {
    event: anchor.event_id,
    from_mm: rounded(anchor.station_start_mm),
    to_mm: rounded(anchor.station_end_mm),
  }
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
  if (chosen.section === 'alignment') return alignmentPick(coords)
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

function alignmentDraft() {
  if (!chosen.draft) {
    const saved = chosen.alignment?.alignment
    chosen.draft = {
      log_text: saved?.log_text || '',
      certainty_mm: saved?.certainty_mm ?? 25,
      anchors: (saved?.anchors || []).map((anchor) => ({ ...anchor })),
    }
  }
  return chosen.draft
}

function renderAlignment(body) {
  const saved = chosen.alignment?.alignment || null
  const report = chosen.alignment?.report || null
  if (report) {
    const counts = report.status_counts
    body.append(title('Alignment summary'), keyValues({
      records: report.event_count,
      clock_legs: report.leg_count,
      rollbacks: report.rollback_count,
      anchors: report.anchor_count,
      certainty_band_mm: report.certainty_mm,
      confident: counts.confident || 0,
      ambiguous: counts.ambiguous || 0,
      unplaced: counts.unplaced || 0,
      outside_capture: counts.outside || 0,
    }))
    for (const boundary of report.boundaries) {
      const tone = boundary.status === 'inconsistent' ? 'result-callout problem' : 'result-callout'
      body.append(element('p', boundary.detail, tone))
    }
    if (report.gaps.length) {
      body.append(copy(`${report.gaps.length} real acquisition gap(s) stay visible on the image. Bands crossing a gap are flagged in the placements table.`))
    }
    if (chosen.alignmentHit) renderAlignmentHit(body)
  }
  renderAlignmentEditor(body, saved)
}

function renderAlignmentHit(body) {
  const hit = chosen.alignmentHit
  const events = chosen.alignment.alignment?.events || []
  const placements = chosen.alignment.report.placements
  const byId = new Map(events.map((event) => [event.id, event]))
  body.append(title(`Log records at ${format(hit.station_mm)} mm`))
  if (!hit.event_ids.length) {
    body.append(copy('No log record covers this station.'))
    return
  }
  for (const id of hit.event_ids) {
    const event = byId.get(id)
    const placement = placements.find((p) => p.event_id === id)
    if (!event) continue
    const card = element('div', null, 'hit-card')
    card.append(element('strong', `${event.id} · ${event.kind}`))
    card.append(keyValues({
      start: event.start_text,
      end: event.end_text,
      note: event.note || '—',
      status: placement?.status,
      band_mm: placement?.station_start_mm != null
        ? `${format(placement.station_start_mm)} – ${format(placement.station_end_mm)}`
        : '—',
      crosses_gap: placement?.crosses_gap,
    }))
    body.append(card)
  }
}

function renderAlignmentEditor(body, saved) {
  if (!chosen.alignment) {
    body.append(copy('Loading alignment…'))
    return
  }
  const draft = alignmentDraft()
  body.append(title('Operation log'))
  body.append(copy('One record per line: kind, start, end[, note]. Times may be HH:MM:SS, YYYY-MM-DD HH:MM:SS, or plain seconds. Record order is the true order; a record that starts before its predecessor marks a clock rollback. Lines starting with # are ignored. Re-parse after editing the log: anchors follow the parsed record ids.'))
  const logArea = element('textarea', null, 'log-text')
  logArea.value = draft.log_text
  logArea.rows = 7
  logArea.maxLength = 30000
  logArea.placeholder = 'pressure test, 08:12:04, 08:12:40, first test'
  logArea.addEventListener('input', () => { draft.log_text = logArea.value })
  body.append(logArea)
  const tools = element('div', null, 'field-row')
  tools.append(button('Parse log', guarded(async () => {
    chosen.parsed = await send(base() + '/alignment/parse', { log_text: draft.log_text })
    renderInspector()
  }), 'small'))
  body.append(tools)
  if (chosen.parsed) renderParsed(body)
  body.append(title('Anchors'))
  body.append(copy('Tie known operations to their narrow station ranges on the image. Each clock leg needs at least two anchors before its other records can be placed.'))
  const events = chosen.parsed?.events || saved?.events || []
  draft.anchors.forEach((anchor, index) => body.append(anchorCard(anchor, index, events)))
  body.append(button('Add anchor', () => {
    if (!events.length) {
      notify('Parse the log before adding anchors.', true)
      return
    }
    chosen.draft.anchors.push({ event_id: events[0].id, station_start_mm: null, station_end_mm: null })
    renderInspector()
  }, 'small'))
  const certainty = field('Certainty band (mm)', 'certainty_mm', draft.certainty_mm, { type: 'number', required: true })
  certainty.querySelector('input').addEventListener('input', (event) => {
    draft.certainty_mm = Number(event.target.value)
  })
  body.append(certainty)
  const actions = element('div', null, 'field-row')
  const save = element('button', 'Save alignment', 'button primary small')
  save.type = 'button'
  save.addEventListener('click', guarded(async () => {
    const payload = {
      log_text: draft.log_text,
      anchors: draft.anchors,
      certainty_mm: draft.certainty_mm,
      revision: revision(),
    }
    const result = await send(base() + '/alignment', payload, 'PUT')
    chosen.capture.state = result.state
    chosen.draft = null
    chosen.parsed = null
    chosen.pickTarget = null
    chosen.alignmentHit = null
    await renderView()
    notify(result.warnings?.length ? result.warnings[0] : 'Alignment saved.', Boolean(result.warnings?.length))
  }))
  actions.append(save)
  if (saved) {
    actions.append(button('Remove alignment', guarded(async () => {
      if (!window.confirm('Remove the saved log alignment from this capture?')) return
      const result = await send(base() + '/alignment', { revision: revision() }, 'DELETE')
      chosen.capture.state = result.state
      chosen.draft = null
      chosen.parsed = null
      chosen.pickTarget = null
      chosen.alignmentHit = null
      await renderView()
      notify('Alignment removed.')
    }), 'small danger'))
  }
  body.append(actions)
}

function renderParsed(body) {
  const parsed = chosen.parsed
  for (const warning of parsed.warnings) {
    body.append(element('p', warning, 'result-callout problem'))
  }
  const legs = parsed.rollback_count + 1
  body.append(copy(`${parsed.events.length} records · ${parsed.rollback_count ? `${parsed.rollback_count} rollback(s) → ${legs} clock legs` : 'single clock leg'}`))
  const rows = parsed.events.map((event) => ({
    id: event.id,
    kind: event.kind,
    start: event.start_text,
    end: event.end_text,
    leg: event.leg + 1,
  }))
  body.append(table(rows, null, 'table-scroll mini-table'))
}

function anchorCard(anchor, index, events) {
  const picking = chosen.pickTarget?.index === index
  const card = element('div', null, 'anchor-card' + (picking ? ' picking' : ''))
  const select = element('select')
  for (const event of events) {
    const leg = event.leg != null ? ` · leg ${event.leg + 1}` : ''
    const option = element('option', `${event.id} · ${event.kind} · ${event.start_text}–${event.end_text}${leg}`)
    option.value = event.id
    select.append(option)
  }
  select.value = anchor.event_id
  select.addEventListener('change', () => { anchor.event_id = select.value })
  card.append(select)
  const range = element('div', null, 'field-grid')
  const start = field('Start station (mm)', `a${index}_start`, anchor.station_start_mm, { type: 'number' })
  const end = field('End station (mm)', `a${index}_end`, anchor.station_end_mm, { type: 'number' })
  start.querySelector('input').addEventListener('input', (event) => {
    anchor.station_start_mm = Number(event.target.value)
  })
  end.querySelector('input').addEventListener('input', (event) => {
    anchor.station_end_mm = Number(event.target.value)
  })
  range.append(start, end)
  card.append(range)
  const tools = element('div', null, 'field-row')
  tools.append(
    button('Pick start', () => {
      chosen.pickTarget = { index, key: 'station_start_mm' }
      renderInspector()
      notify('Click the image to set the anchor start station.')
    }, 'small'),
    button('Pick end', () => {
      chosen.pickTarget = { index, key: 'station_end_mm' }
      renderInspector()
      notify('Click the image to set the anchor end station.')
    }, 'small'),
    button('Remove', () => {
      chosen.draft.anchors.splice(index, 1)
      chosen.pickTarget = null
      renderInspector()
    }, 'small danger'),
  )
  card.append(tools)
  return card
}

function alignmentPick(coords) {
  const extent = chosen.extent
  const station = extent.y_from + coords.y * (extent.y_to - extent.y_from)
  markPoint(coords)
  if (chosen.pickTarget) {
    const anchor = chosen.draft?.anchors[chosen.pickTarget.index]
    if (anchor) {
      anchor[chosen.pickTarget.key] = Math.round(station * 10) / 10
      const end = chosen.pickTarget.key === 'station_end_mm'
      notify(`Anchor ${end ? 'end' : 'start'} set to ${format(anchor[chosen.pickTarget.key])} mm.`)
    }
    chosen.pickTarget = null
    renderInspector()
    return
  }
  const report = chosen.alignment?.report
  if (!report) return
  const hits = report.placements.filter((p) =>
    p.station_start_mm != null && station >= p.station_start_mm && station <= p.station_end_mm)
  chosen.alignmentHit = { station_mm: station, event_ids: hits.map((p) => p.event_id) }
  renderInspector()
  redrawAlignment()
}

function redrawAlignment() {
  const canvas = $('#ruler-layer')
  if (chosen.section !== 'alignment') return
  const width = canvas.clientWidth
  const height = canvas.clientHeight
  const ratio = window.devicePixelRatio || 1
  canvas.width = Math.round(width * ratio)
  canvas.height = Math.round(height * ratio)
  const extent = chosen.extent
  const report = chosen.alignment?.report
  if (!extent || !report) return
  const brush = canvas.getContext('2d')
  brush.scale(ratio, ratio)
  const span = extent.y_to - extent.y_from || 1
  const py = (station) => (station - extent.y_from) / span * height
  const band = (from, to, fill, stroke, dash = []) => {
    const y0 = py(from)
    const y1 = py(to)
    if (Math.min(y0, y1) > height || Math.max(y0, y1) < 0) return
    const top = Math.max(0, Math.min(y0, y1))
    const size = Math.max(2, Math.min(Math.max(y0, y1), height) - top)
    brush.fillStyle = fill
    brush.fillRect(0, top, width, size)
    brush.strokeStyle = stroke
    brush.lineWidth = 1.2
    brush.setLineDash(dash)
    brush.strokeRect(-1, top, width + 2, size)
    brush.setLineDash([])
  }
  for (const gap of report.gaps) {
    band(gap.from_mm, gap.to_mm, 'rgba(86, 99, 168, 0.5)', 'rgba(150, 162, 220, 0.8)')
  }
  for (const anchor of chosen.alignment.alignment?.anchors || []) {
    band(anchor.station_start_mm, anchor.station_end_mm, 'rgba(136, 174, 242, 0.30)', '#88aef2')
  }
  for (const p of report.placements) {
    if (p.station_start_mm == null) continue
    if (p.status === 'confident') {
      band(p.station_start_mm, p.station_end_mm, 'rgba(114, 229, 204, 0.22)', '#72e5cc')
    } else if (p.status === 'ambiguous') {
      band(p.station_start_mm, p.station_end_mm, 'rgba(240, 177, 111, 0.20)', '#f0b16f', [6, 4])
    }
  }
  if (chosen.alignmentHit) {
    brush.strokeStyle = '#f6d46d'
    brush.lineWidth = 2
    for (const id of chosen.alignmentHit.event_ids) {
      const p = report.placements.find((item) => item.event_id === id)
      if (!p || p.station_start_mm == null) continue
      const top = Math.max(0, Math.min(py(p.station_start_mm), py(p.station_end_mm)))
      const size = Math.max(2, Math.abs(py(p.station_end_mm) - py(p.station_start_mm)))
      brush.strokeRect(1, top, width - 2, size)
    }
  }
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
  window.addEventListener('resize', () => {
    redrawRuler()
    redrawAlignment()
  })
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
