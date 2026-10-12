-- Give Pandoc's LaTeX writer bounded column widths when an HTML table has no
-- explicit colspec. Without widths it emits l/r/c columns that can extend past
-- the page instead of wrapping. Also add legal breakpoints after hyphens in
-- table code spans, where LaTeX's \texttt normally refuses to wrap identifiers.
-- Explicit author-supplied widths remain intact.

local function wrap_hyphenated_code(code)
  if not string.find(code.text, "-", 1, true) then
    return nil
  end

  local result = pandoc.Inlines({})
  local cursor = 1
  while true do
    local boundary = string.find(code.text, "-", cursor, true)
    if boundary == nil then
      result:insert(pandoc.Code(string.sub(code.text, cursor), code.attr))
      break
    end
    result:insert(pandoc.Code(string.sub(code.text, cursor, boundary), code.attr))
    result:insert(pandoc.RawInline("latex", "\\allowbreak{}"))
    cursor = boundary + 1
  end
  return result
end

function Table(table_element)
  table_element = table_element:walk({Code = wrap_hyphenated_code})
  local column_count = #table_element.colspecs
  if column_count == 0 then
    return table_element
  end

  local has_explicit_width = false
  for _, column_spec in ipairs(table_element.colspecs) do
    local width = column_spec[2]
    if type(width) == "number" and width > 0 then
      has_explicit_width = true
      break
    end
  end

  if has_explicit_width then
    return table_element
  end

  local width = 1 / column_count
  for index, column_spec in ipairs(table_element.colspecs) do
    table_element.colspecs[index] = {column_spec[1], width}
  end
  return table_element
end
