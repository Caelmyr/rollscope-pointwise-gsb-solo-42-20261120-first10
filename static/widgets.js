export const $ = (selector) => document.querySelector(selector)

export function element(tag, content, className) {
  const node = document.createElement(tag)
  if (content !== undefined && content !== null) node.textContent = content
  if (className) node.className = className
  return node
}

export function labelFor(value) {
  const abbrev = { mm: 'mm', mm2: 'mm²', rms: 'RMS', id: 'ID', px: 'px', s: 's',
    ms: 'ms', p10: 'P10', p90: 'P90', csv: 'CSV', sd: 'SD',
  }
  const words = value.split('_').map((word) => abbrev[word] || word)
  const result = words.join(' ')
  return result.charAt(0).toUpperCase() + result.slice(1)
}

export const whole = new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 })
const precision = new Intl.NumberFormat('en-US', { maximumFractionDigits: 4 })
const fine = new Intl.NumberFormat('en-US', { maximumSignificantDigits: 4 })

export function format(value) {
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) return '—'
    return Math.abs(value) > 0 && Math.abs(value) < 0.001 ? fine.format(value) : precision.format(value)
  }
  if (value == null) return '—'
  if (value === true) return 'Yes'
  if (value === false) return 'No'
  if (Array.isArray(value)) return value.length ? value.map(format).join(', ') : '—'
  if (typeof value === 'object') {
    return Object.entries(value).map(([key, quantity]) => `${labelFor(key)} ${format(quantity)}`).join(' · ')
  }
  return String(value).replaceAll('_', ' ')
}

export function keyValues(data) {
  const list = element('dl', null, 'kv')
  for (const [name, value] of Object.entries(data || {})) {
    if (value && typeof value === 'object' && !Array.isArray(value)) {
      list.append(element('div', labelFor(name), 'kv-section'))
      for (const [key, inner] of Object.entries(value)) addKey(list, key, inner)
    } else {
      addKey(list, name, value)
    }
  }
  return list
}

function addKey(list, name, value) {
  list.append(element('dt', labelFor(name)), element('dd', format(value)))
}

export function table(rows, onSelect, className) {
  if (!rows.length) return element('p', 'No measurements in this table.', 'empty-table')
  const wrapper = element('div', null, className || 'table-scroll')
  const node = element('table')
  const columns = Object.keys(rows[0])
  const head = node.createTHead().insertRow()
  for (const key of columns) {
    const cell = element('th', labelFor(key))
    cell.scope = 'col'
    head.append(cell)
  }
  const tbody = node.createTBody()
  for (const data of rows) {
    const tr = tbody.insertRow()
    for (const key of columns) {
      const value = data[key]
      tr.append(element('td', format(value), typeof value === 'number' ? 'tabular' : ''))
    }
    const hasLocator = ['station_mm', 'from_mm', 'station_after_mm', 'source_line'].some(
      (key) => typeof data[key] === 'number'
    )
    if (onSelect && hasLocator) {
      tr.dataset.station = 'true'
      tr.tabIndex = 0
      tr.title = 'Locate this row in the image'
      tr.addEventListener('click', () => onSelect(data))
      tr.addEventListener('keydown', (event) => {
        if (event.key === 'Enter') onSelect(data)
      })
    }
  }
  wrapper.append(node)
  return wrapper
}

export function field(label, name, value, options = {}) {
  let input
  if (options.choices) {
    input = element('select')
    for (const [key, title] of options.choices) {
      const option = element('option', title)
      option.value = key
      input.append(option)
    }
  } else if (options.long) {
    input = element('textarea')
  } else {
    input = element('input')
    input.type = options.type || 'text'
  }
  if (input.type === 'number') input.step = options.step || 'any'
  if (options.max) input.maxLength = options.max
  if (options.placeholder) input.placeholder = options.placeholder
  if (options.required) input.required = true
  input.name = name
  input.value = value ?? ''
  const wrapper = element('label', label, 'field')
  wrapper.append(input)
  return wrapper
}

export function button(name, action, className = '') {
  const control = element('button', name, `button ${className}`)
  control.type = 'button'
  control.addEventListener('click', action)
  return control
}

export function link(name, href, className = 'button small') {
  const anchor = element('a', name, className)
  anchor.href = href
  return anchor
}

const COLORS = ['#72e5cc', '#e4ad6d', '#9aa8f5', '#d6a0cf']
const plots = new WeakMap()

function axisLabel(value) {
  return Math.abs(value) > 100 ? whole.format(value) : format(value)
}

export function chart(specification) {
  const wrapper = element('section', null, 'plot-card')
  wrapper.append(element('h3', specification.title))
  const canvas = element('canvas')
  canvas.setAttribute('aria-label', specification.title)
  canvas.setAttribute('role', 'img')
  canvas.dataset.chart = 'true'
  plots.set(canvas, specification)
  wrapper.append(canvas)
  if (specification.y.length > 1) {
    const keys = element('div', null, 'plot-keys')
    specification.y.forEach((key, index) => {
      const item = element('span', '● ' + labelFor(key))
      item.style.color = COLORS[index]
      keys.append(item)
    })
    wrapper.append(keys)
  }
  requestAnimationFrame(() => draw(canvas))
  return wrapper
}

function limits(values) {
  if (!values.length) return [0, 1]
  const low = Math.min(...values)
  const high = Math.max(...values)
  if (low === high) {
    const difference = Math.max(0.01, Math.abs(low) * 0.1)
    return [low - difference, low + difference]
  }
  return [low, high]
}

function draw(canvas) {
  const width = canvas.clientWidth
  const height = canvas.clientHeight
  if (!width || !height) return
  const specification = plots.get(canvas)
  const ratio = window.devicePixelRatio || 1
  canvas.width = Math.round(width * ratio)
  canvas.height = Math.round(height * ratio)
  const brush = canvas.getContext('2d')
  brush.scale(ratio, ratio)
  const { records, x, y, unit } = specification
  const xs = records.map((record) => record[x]).filter(Number.isFinite)
  const ys = records.flatMap((record) => y.map((key) => record[key])).filter(Number.isFinite)
  const [x0, x1] = limits(xs)
  const [y0, y1] = limits(ys)
  const start = 43
  const final = width - 7
  const top = 8
  const base = height - 19
  const transX = (value) => start + (value - x0) * (final - start) / (x1 - x0)
  const transY = (value) => base - (value - y0) * (base - top) / (y1 - y0)
  brush.strokeStyle = '#2a394f'
  brush.fillStyle = '#91a4bc'
  brush.font = '9px ui-monospace, monospace'
  for (let tick = 0; tick <= 3; tick++) {
    const position = top + tick * (base - top) / 3
    const yValue = y1 - tick * (y1 - y0) / 3
    brush.beginPath()
    brush.moveTo(start, position)
    brush.lineTo(final, position)
    brush.stroke()
    brush.textAlign = 'right'
    brush.fillText(axisLabel(yValue), start - 5, position + 3)
  }
  brush.textAlign = 'left'
  brush.fillText(axisLabel(x0), start, height - 4)
  brush.textAlign = 'right'
  brush.fillText(axisLabel(x1), final, height - 4)
  brush.textAlign = 'center'
  brush.fillText(unit, (start + final) / 2, height - 4)
  if (!ys.length) {
    brush.fillText('No supported values', (start + final) / 2, (top + base) / 2)
    return
  }

  y.forEach((key, index) => {
    brush.strokeStyle = COLORS[index]
    brush.lineWidth = 1.4
    brush.beginPath()
    let engaged = false
    for (const record of records) {
      const xValue = record[x]
      const yValue = record[key]
      if (!Number.isFinite(xValue) || !Number.isFinite(yValue)) {
        engaged = false
        continue
      }
      const point = [transX(xValue), transY(yValue)]
      if (engaged) brush.lineTo(...point)
      else brush.moveTo(...point)
      engaged = true
    }
    brush.stroke()
  })
}

window.addEventListener('resize', () => document.querySelectorAll('[data-chart]').forEach(draw))
