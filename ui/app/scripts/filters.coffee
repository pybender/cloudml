'use strict'

### Filters ###

angular.module('app.filters', [])

.filter('interpolate', [
  'version',

(version) ->
  (text) ->
    String(text).replace(/\%VERSION\%/mg, version)
])

.filter('capfirst', [() ->
  (text) ->
    t = String(text)
    return t[0].toUpperCase() + t.slice(1)
])

.filter('format_date', [() ->
  (text) ->
    dt = new Date(text)
    d = dt.getDate()
    if d < 10
        d = '0' + d
    m = dt.getMonth() + 1
    if m < 10
        m = '0' + m
    y = dt.getFullYear()
    h = dt.getHours()
    mm = dt.getMinutes()
    return d + "-" + m + "-" + y + ' ' + h + ':' + mm
])