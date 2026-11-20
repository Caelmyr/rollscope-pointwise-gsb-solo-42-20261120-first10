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
  grain: ['Grain conformance', 'Grain direction and coarseness against two signed-off windows', 'grain'],
  illumination: ['Reflectance trend', 'Across-strip and along-roll material reflectance', 'illumination'],
  windows: ['Review windows', 'Saved bounds in physical station coordinates', 'station'],
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
  grain: [
    ['tile_size_mm', 'Grain tile size (mm)', '0.1'],
    ['overlap_fraction', 'Tile overlap fraction'],
    ['minimum_wave_mm', 'Shortest grain wavelength (mm)'],
    ['maximum_wave_mm', 'Longest grain wavelength (mm)'],
    ['minimum_coherence', 'Minimum grain coherence'],
    ['envelope_margin', 'Reference envelope width (sigma)'],
    ['outlier_sigmas', 'Reference outlier cutoff (sigma)'],
    ['angle_tolerance_deg', 'Minimum direction tolerance (deg)'],
    ['coarseness_ratio', 'Minimum coarseness tolerance (ratio)'],
    ['minimum_reference_tiles', 'Minimum reference tiles', '1'],
    ['maximum_feature_share', 'Maximum feature share per tile'],
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
  grain: [['both', 'Both references'], ['only-a', 'Only reference A'], ['only-b', 'Only reference B'], ['neither', 'Neither reference'], ['gap', 'Encoder gap'], ['bad', 'Unknown, feature or unavailable']],
  illumination: [['reflectance', 'Reflectance'], ['gap', 'Encoder gap'], ['bad', 'Invalid sample']],
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
  grain: null,
  grainTile: null,
  grainTileKey: '',
  snapshot: null,
  compare: null,
  probe: null,
  profile: null,
  rulerPoints: [],
  ruler: null,
  request: 0,
  load: 0,
  probeRequest: 0,
}

let toastTimer
const regularView = () => !['windows', 'ruler', 'snapshots', 'grain'].includes(chosen.section)
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
      grain: null, grainTile: null, grainTileKey: '',
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
  $('#window-picker').disabled = raw || chosen.section === 'ruler' || chosen.section === 'grain'
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
    : 'Select a pixel to inspect its source lines.'
  const sectionLink = $('#section-download')
  sectionLink.hidden = false
  sectionLink.textContent = chosen.section === 'grain' ? 'Export grain evidence ↗' : 'Export ZIP ↗'
  if (chosen.section === 'grain') {
    sectionLink.href = base() + '/grain/export'
  } else if (regularView()) {
    sectionLink.href = anchorURL(base() + `/analysis/${chosen.section}/export`, { revision: revision() })
  } else {
    sectionLink.hidden = true
  }
  const parameters = viewParams({ overlay: $('#overlay-toggle').checked ? 1 : 0 })
  const viewURL = anchorURL(base() + '/view', parameters)
  const imageURL = anchorURL(base() + '/preview.png', parameters)
  let grainUnready = chosen.section === 'grain' && !chosen.grain?.configured
  let grainOffCapture = false
  const analysisURL = regularView()
    ? anchorURL(base() + '/analysis/' + chosen.section, { revision: revision() })
    : chosen.section === 'grain'
      ? base() + '/grain'
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
    if (chosen.section === 'grain') {
      chosen.grain = result[1]
      if (result[1]?.selection) chosen.grainSelection = result[1].selection
      grainUnready = !result[1]?.configured
      grainOffCapture = Boolean(result[1]?.configured && result[1].target_capture !== chosen.capture.id)
    } else {
      chosen.analysis = result[1]
    }
  } catch (error) {
    if (token === chosen.request) showErrorImage(error)
    throw error
  }
  const image = $('#raster')
  if (grainUnready || grainOffCapture) {
    $('#raster').removeAttribute('src')
    $('#empty-image').textContent = grainUnready
      ? 'Choose two signed-off windows and a target window to draw the grain classification.'
      : `The target window is on ${chosen.grain.target_capture_name || 'another capture'}. Open that capture to see the classification map.`
    $('#empty-image').hidden = false
  } else {
    image.src = imageURL
    image.alt = title + ' image'
    $('#empty-image').hidden = true
  }
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
  if (chosen.section === 'windows') return renderWindows(body)
  if (chosen.section === 'ruler') return renderRuler(body)
  if (chosen.section === 'snapshots') return renderSnapshots(body)
  if (chosen.section === 'grain') return renderGrain(body)
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
  if (chosen.section === 'grain') return grainTables()
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
    const picker = chosen.section === 'grain' && report.scope === 'target'
      ? guarded(locateGrainTile)
      : guarded(locate)
    group.append(element('h3', report.title, 'table-title'), table(report.rows, picker))
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


const GRAIN_CLASSES = [
  ['both_references', 'Supports both references'],
  ['only_reference_a', 'Only reference A'],
  ['only_reference_b', 'Only reference B'],
  ['neither_reference', 'Neither reference'],
  ['unknown', 'Unknown'],
]

const GRAIN_ROLES = [
  ['reference_a', 'Reference A (signed off)'],
  ['reference_b', 'Reference B (signed off)'],
  ['target', 'New roll (target)'],
]

function grainKey(reference) {
  return reference ? reference.capture + ':' + reference.window : ''
}

function grainChoice(reference) {
  if (!reference) return ''
  return `${reference.capture}|${reference.window}`
}

function splitChoice(value) {
  if (!value) return null
  const [capture, window] = value.split('|')
  return capture && window ? { capture, window } : null
}

function grainCounts() {
  if (!chosen.grain?.matrix) return null
  return chosen.grain.matrix.target
}

function renderGrain(body) {
  if (!chosen.grain) {
    body.append(copy('Comparing grain against the signed-off windows…'))
    return
  }
  const report = chosen.grain
  const configured = Boolean(report?.configured)
  body.append(title('Signed-off windows'), copy(
    'Choose the two approved windows as references and the window on the new roll to test. ' +
    'Direction and coarseness are compared in millimetres; lighting is normalized out.'
  ))
  body.append(grainChooser(report?.catalog || [], report?.selection || chosen.grainSelection))
  if (!configured) {
    body.append(copy('Both signed-off windows are required before a comparison can run.'))
    return
  }
  body.append(title('Result'), keyValues(grainSummaryValues(report.summary)))
  body.append(grainClassList(report.summary, grainCounts()))
  body.append(title('Signed-off envelopes'), keyValues(grainEnvelopeValues(report)))
  const rows = grainTileRows(report).slice(0, 40)
  body.append(title('Inspect a tile'), copy(
    'Pick a row to inspect its direction, coarseness and the raw scan lines behind it. ' +
    'The measurement tables below also locate tiles on the image.'
  ))
  body.append(table(rows, guarded(locateGrainTile), 'table-scroll mini-table'))
  if (chosen.grainTile) body.append(grainTilePanel(chosen.grainTile))
  else body.append(copy('No tile selected yet.'))
}

function grainSummaryValues(summary) {
  const values = {
    target_window: summary.target_window,
    reference_a: summary.reference_a,
    reference_b: summary.reference_b,
    tile_size_mm: `${format(summary.tile_height_mm)} × ${format(summary.tile_width_mm)}`,
    both_references_tiles: summary.both_references_tiles,
    only_reference_a_tiles: summary.only_reference_a_tiles,
    only_reference_b_tiles: summary.only_reference_b_tiles,
    neither_reference_tiles: summary.neither_reference_tiles,
    unknown_tiles: summary.unknown_tiles,
    unknown_feature_tiles: summary.unknown_feature_tiles,
    unknown_unsupported_tiles: summary.unknown_unsupported_tiles,
    unknown_outside_tiles: summary.unknown_outside_tiles,
    unknown_faint_tiles: summary.unknown_faint_tiles,
    both_references_fraction: summary.both_references_fraction,
    outside_normal_fraction: summary.outside_normal_fraction,
    target_direction_median_deg: summary.target_direction_median_deg,
    target_coarseness_median_mm: summary.target_coarseness_median_mm,
  }
  return values
}

function grainEnvelopeValues(report) {
  const values = {}
  for (const envelope of report.references || []) {
    const label = envelope.key === 'a' ? 'Reference A' : 'Reference B'
    values[label] = {
      window: envelope.window,
      status: envelope.status,
      direction_center_deg: envelope.direction_center_deg,
      direction_tolerance_deg: envelope.direction_tolerance_deg,
      coarseness_center_mm: envelope.coarseness_center_mm,
      coarseness_tolerance_ratio: envelope.coarseness_tolerance_ratio,
      measured_tiles: envelope.measured_tiles,
      trimmed_tiles: envelope.trimmed_tiles,
    }
  }
  return values
}

function grainChooser(catalog, selection) {
  const form = element('form')
  const choices = [['', '— choose a window —']].concat(catalog.map((item) => [
    `${item.capture_ref}|${item.window_id}`,
    `${item.capture_name} · ${item.label}`,
  ]))
  for (const [role, label] of GRAIN_ROLES) {
    form.append(field(label, role, grainChoice(selection?.[role]), { choices }))
  }
  const actions = element('div', null, 'field-row')
  const submit = element('button', 'Compare grain', 'button primary small')
  submit.type = 'submit'
  actions.append(submit)
  form.append(actions)
  form.addEventListener('submit', guarded(async (event) => {
    event.preventDefault()
    const entries = new FormData(form)
    const payload = { revision: revision() }
    for (const [role] of GRAIN_ROLES) payload[role] = splitChoice(entries.get(role))
    chosen.grainSelection = payload
    const result = await send(base() + '/grain', payload, 'PUT')
    chosen.capture.state = result.state
    chosen.grainTile = null
    chosen.grainTileKey = ''
    await renderView()
    notify('Grain comparison saved.')
  }))
  return form
}

function percentage(value) {
  return typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : '—'
}

function grainClassList(summary, counts) {
  const list = element('dl', null, 'kv')
  const total = summary.measured_tiles || 0
  for (const [key, label] of GRAIN_CLASSES) {
    const value = counts?.[key] ?? 0
    const share = total && key !== 'unknown' ? ` (${(100 * value / total).toFixed(1)}%)` : ''
    if (key === 'unknown') {
      list.append(element('dt', label), element('dd', String(summary.unknown_tiles ?? 0)))
    } else {
      list.append(element('dt', label), element('dd', `${value}${share}`))
    }
  }
  if (!total) {
    list.append(element('dt', 'Measured tiles'), element('dd', 'None'))
  }
  return list
}

function grainTileRows(report) {
  return (report.tables || [])
    .filter((entry) => entry.key === 'grain_target_tiles')
    .flatMap((entry) => entry.rows)
}

function grainTables() {
  if (!chosen.grain?.configured) return []
  const rows = grainTileRows(chosen.grain)
  const measured = chosen.grain.summary.measured_tiles || 0
  const extents = chosen.grain.extents || {}
  const byClass = GRAIN_CLASSES.map(([key, label]) => {
    const extent = extents[key] || {}
    const count = rows.filter((row) => row.classification === key).length
    return {
      class: label,
      tiles: count,
      share_of_measured: key === 'unknown' || !measured ? null : count / measured,
      station_from_mm: extent.station_from_mm ?? null,
      station_to_mm: extent.station_to_mm ?? null,
      transverse_from_mm: extent.transverse_from_mm ?? null,
      transverse_to_mm: extent.transverse_to_mm ?? null,
    }
  })
  const tables = [
    { title: 'Tiles by conformance', rows: byClass },
    { title: 'Target tile measurements', rows, scope: 'target' },
  ]
  for (const [key, label] of [['grain_reference_a_tiles', chosen.grain.summary.reference_a],
                              ['grain_reference_b_tiles', chosen.grain.summary.reference_b]]) {
    const entry = (chosen.grain.tables || []).find((item) => item.key === key)
    if (entry) tables.push({ title: `${label} tile measurements`, rows: entry.rows })
  }
  const envelopes = (chosen.grain.tables || []).find((item) => item.key === 'grain_envelopes')
  if (envelopes) tables.push({ title: 'Signed-off grain envelopes', rows: envelopes.rows })
  return tables
}

async function locateGrainTile(data) {
  const extent = chosen.extent
  if (!extent || !data) return
  const yCoord = (data.station_mm - extent.y_from) / (extent.y_to - extent.y_from || 1)
  const xCoord = (data.transverse_mm - extent.x_from_mm) / (extent.x_to_mm - extent.x_from_mm || 1)
  if (!(yCoord >= 0 && yCoord <= 1 && xCoord >= 0 && xCoord <= 1)) {
    notify('That tile lies outside the current view.')
    return
  }
  const key = `target:${data.tile}`
  chosen.grainTileKey = key
  chosen.grainTile = null
  markPoint({ x: xCoord, y: yCoord })
  const result = await json(`${base()}/grain/tiles/target/${data.tile}`)
  if (chosen.grainTileKey !== key) return
  chosen.grainTile = result
  renderInspector()
}

function grainTilePanel(tile) {
  const wrapper = element('section')
  wrapper.append(title(`Tile ${tile.tile.tile} · ${tile.classification_title}`))
  wrapper.append(keyValues({
    station_mm: tile.tile.station_mm,
    transverse_mm: tile.tile.transverse_mm,
    direction_deg: tile.tile.direction_deg,
    coherence: tile.tile.coherence,
    coarseness_mm: tile.tile.coarseness_mm,
    relative_amplitude: tile.tile.relative_amplitude,
    state: tile.tile.state,
    unknown_reason: tile.unknown_reason,
    nearest_reference: tile.nearest_reference,
    limiting_feature: tile.limiting_feature,
    size_mm: `${format(tile.tile.height_mm)} × ${format(tile.tile.width_mm)}`,
  }))
  const bands = { title: 'Wavelength share', x: 'from_mm', y: ['fraction'], unit: 'fraction', records: tile.bands }
  wrapper.append(chart(bands))
  const lines = tile.source_lines || []
  if (lines.length) {
    wrapper.append(title('Source scan lines'))
    wrapper.append(table(lines, null, 'table-scroll mini-table'))
  }
  return wrapper
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
